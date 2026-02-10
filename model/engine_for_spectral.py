# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    model_without_ddp,
                    log_writer=None,
                    opts=None,
                    start_steps=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    optimizer.zero_grad()
    accum_iter = opts['accum_iter']
    display_iter = opts['display_iter']

    # mean, std = load_mean_std(opts['mean_std_path'])
    
    mask_zoom = opts['max_mask_ratio'] - opts['min_mask_ratio']
    # band = np.arange(opts['band'][0], opts['band'][1], opts['band'][2])
    for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, opts)

        # random mask ratio
        if opts['random_mask_ratio']:
            mask_ratio = np.random.rand() * mask_zoom + opts['min_mask_ratio']
        else:
            mask_ratio = opts['mask_ratio']
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if (step + 1) % opts['save_latest_freq'] == 0:
            if opts['output_dir']:
                misc.save_model(
                    opts=opts, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_value_reduce=loss_value_reduce, head="loss_value_reduce")
            log_writer.update(lr=lr, head="lr")
            log_writer.set_step()
        if (step + len(data_loader) * epoch) % display_iter == 0:
            origin, result, rHSI, result_HSI = data_loader.dataset.random_model_test(model)
            if log_writer is not None:
                log_writer.writer.add_image("origin", origin, global_step=step + len(data_loader) * epoch,
                                            dataformats='HWC')
                log_writer.writer.add_image("result", result, global_step=step + len(data_loader) * epoch,
                                            dataformats='HWC')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}