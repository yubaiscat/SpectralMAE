import datetime
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

import util.misc as misc
import util.spectral_data as sp
from model import modeling_spectral
from model.engine_for_spectral import train_one_epoch
from util.base_options import BaseOptions
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optim_factory import create_optimizer


def get_model(opts):
    print(f"Creating model: {opts['model']}")
    if (opts['model'] == 'SpectralMAE'):
        model = modeling_spectral.SpectralMAE(
            patch_size=opts['patch_size'], embed_dim=opts['embed_dim'], depth=opts['depth'],
            num_heads=opts['num_heads'], in_chans=opts['in_chans'],
            decoder_embed_dim=opts['decoder_embed_dim'], decoder_depth=opts['decoder_depth'],
            decoder_num_heads=opts['decoder_num_heads'],
            mlp_ratio=opts['mlp_ratio'], norm_layer=partial(nn.LayerNorm, eps=1e-6),
            random_mask = opts['random_mask'], keep_list = opts['keep_list']
        )
    return model

def main(opts):
    misc.init_distributed_mode(opts)
    opts['distributed'] = False
    device = torch.device(opts['device'])
    torch.cuda.set_device(opts['gpu_ids'])
    
    # fix the seed for reproducibility
    seed = opts['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    model = get_model(opts)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    # get dataset
    dataset_train = sp.build_dataset(opts)
    train_loader = DataLoader(dataset=dataset_train, batch_size=opts['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=opts['pin_mem'], drop_last=True)
    num_training_steps_per_epoch = len(dataset_train) // opts['batch_size']
    
    # log 
    if opts['log_dir'] is not None:
        os.makedirs(opts['log_dir'], exist_ok=True)
        log_writer = misc.TensorboardLogger(log_dir=opts['log_dir'])
    else:
        log_writer = None

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = opts['batch_size'] * misc.get_world_size()
    opts['lr'] = opts['lr'] * total_batch_size / 256

    print("LR = %.8f" % opts['lr'])
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if opts['distributed']:
       model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts['gpu']], find_unused_parameters=True)
       model_without_ddp = model.module
    
    optimizer = optim_factory.create_optimizer_v2(model_without_ddp, opt=opts['opt'], weight_decay=opts['weight_decay'],
                                                  lr=opts['lr'])
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.auto_load_model(
        opts=opts, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {opts['epochs']} epochs")
    start_time = time.time()
    for epoch in range(opts['start_epoch'], opts['epochs']):
        if opts['distributed']:
            train_loader.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            opts=opts,
            start_steps=epoch * num_training_steps_per_epoch,
            model_without_ddp=model_without_ddp
        )
        if opts['output_dir']:
            if (epoch + 1) % opts['save_ckpt_freq'] == 0 or epoch + 1 == opts['epochs']:
                misc.save_model(
                    opts=opts, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if opts['output_dir'] and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(opts['output_dir'], "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts = BaseOptions().parse()
    if 'output_dir' in opts:
        Path(opts['output_dir']).mkdir(parents=True, exist_ok=True)
    main(opts)
