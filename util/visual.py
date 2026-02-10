import pathlib
import random

import cv2
import h5py
import imgvision as iv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def random_model_test(model, mean, std, opts, device: torch.device, band=np.arange(390, 700, 10)):
	hyper_list = list(pathlib.Path(opts['test_path']).glob('*.mat'))
	hyper_list.sort()
	# print(f'length of dataset:{len(hyper_list)}')
	rand = random.randint(0, len(hyper_list)-1)
	with h5py.File(hyper_list[rand], 'r') as mat:
		HSI = np.float32(np.array(mat['cube']))
		rHSI = np.einsum('chw->hwc', HSI)
		origin = HSI2RGB(rHSI, band)
		norm_HSI = rHSI
		for c in range(opts['in_chans']):
			norm_HSI[:,:,c] = (norm_HSI[:, :, c] - mean[c]) / std[c]
		
		result_HSI = run_one_image(norm_HSI, model, opts, device)
		for c in range(opts['in_chans']):
			result_HSI[:,:,c] = (result_HSI[:, :, c] * std[c]) + mean[c]

		result = HSI2RGB(result_HSI, band)
		return origin, result, rHSI, result_HSI

def visual_fake_hyper(HSI, RGB_chanels):
    R = HSI[:,:,RGB_chanels[0]]
    G = HSI[:,:,RGB_chanels[1]]
    B = HSI[:,:,RGB_chanels[2]]
    img = np.stack((R,G,B),axis=0)
    img = np.einsum('chw->hwc', img)
    img *= 255
    img = Image.fromarray(img)
    return img
	
def valid_one_image(HSI, model, opts, mask_ratio, device: torch.device):
	H = opts['display_height']
	W = opts['display_width']
	C = HSI.shape[-1]
	patch_size = opts['patch_size']

	patch_per_line = W // patch_size
	patch_per_column = H // patch_size
	assert W % patch_size == 0
	assert H % patch_size == 0

	rHSI = resize_spectral(HSI, H, W, C)
	batch_num = patch_per_line*patch_per_column
	batch = np.zeros((batch_num,patch_size,patch_size,C))
	for h_idx in range(patch_per_line):
		for w_idx in range(patch_per_column):
			batch[h_idx*patch_per_column+w_idx, :, :, :] = rHSI[h_idx * patch_size: h_idx * patch_size + patch_size, w_idx * patch_size: w_idx * patch_size + patch_size,:]

	batch_tensor = torch.tensor(batch).float().to(device, non_blocking=True)
	result_tensor = torch.zeros(rHSI.shape).float().to(device, non_blocking=True)
	mask_tensor = torch.zeros(rHSI.shape).float().to(device, non_blocking=True)

	# make it a batch-like
	result = np.zeros(batch_tensor.shape)
	mask = np.zeros(batch_tensor.shape)
	batch_tensor = torch.einsum('nhwc->nchw', batch_tensor)

	display_batch = opts['display_batch']
	batch_time = batch_num // display_batch
	assert batch_num % display_batch == 0
	for i in range(batch_time):
		batch = batch_tensor[i*display_batch:(i+1)*display_batch,:,:,:]
		loss, result_batch, mask_batch = model(batch, mask_ratio=mask_ratio)
		mask_batch = mask_batch.unsqueeze(-1)
		mask_batch = mask_batch.repeat(1, 1, result_batch.shape[-1])
		mask_batch = model.unpatchify(mask_batch)
		result_batch = model.unpatchify(result_batch)
		result_batch = torch.einsum('nchw->nhwc', result_batch)
		mask_batch = torch.einsum('nchw->nhwc', mask_batch)
		result[i * display_batch:(i + 1) * display_batch, :, :, :] = result_batch.cpu().detach().numpy()
		mask[i * display_batch:(i + 1) * display_batch , :, :, :] = mask_batch.cpu().detach().numpy()

	for h_idx in range(patch_per_line):
		for w_idx in range(patch_per_column):
			result_tensor[h_idx * patch_size: h_idx * patch_size + patch_size, w_idx * patch_size: w_idx * patch_size + patch_size,:] = result[h_idx*patch_per_column+w_idx, :, :, :]
			mask_tensor[h_idx * patch_size: h_idx * patch_size + patch_size, w_idx * patch_size: w_idx * patch_size + patch_size,:] = mask[h_idx*patch_per_column+w_idx, :, :, :]
	return result_HSI, mask_HSI, rHSI


def run_one_image(HSI, model, opts, device: torch.device):
	H = opts['display_height']
	W = opts['display_width']
	C = HSI.shape[-1]
	patch_size = opts['patch_size']

	patch_per_line = W // patch_size
	patch_per_column = H // patch_size
	assert W % patch_size == 0
	assert H % patch_size == 0

	rHSI = resize_spectral(HSI, H, W, C)
	batch_num = patch_per_line * patch_per_column
	batch = np.zeros((batch_num, patch_size, patch_size, C))
	for h_idx in range(patch_per_line):
		for w_idx in range(patch_per_column):
			batch[h_idx * patch_per_column + w_idx, :, :, :] = rHSI[h_idx * patch_size: h_idx * patch_size + patch_size,
															   w_idx * patch_size: w_idx * patch_size + patch_size, :]

	batch_tensor = torch.tensor(batch).float().to(device, non_blocking=True)
	result_tensor = np.zeros(rHSI.shape)

	# make it a batch-like
	result = np.zeros(batch_tensor.shape)
	batch_tensor = torch.einsum('nhwc->nchw', batch_tensor)

	display_batch = opts['display_batch']
	batch_time = batch_num // display_batch
	assert batch_num % display_batch == 0
	# result_tensor = result_tensor
	# mask_tensor = mask_tensor.to(device, non_blocking=True)
	# batch_tensor = batch_tensor.to(device, non_blocking=True)
	for i in range(batch_time):
		batch = batch_tensor[i * display_batch:(i + 1) * display_batch, :, :, :]
		loss, result_batch, mask_batch = model(batch, mask_ratio=opts['mask_ratio'])
		result_batch = model.unpatchify(result_batch)
		result_batch = torch.einsum('nchw->nhwc', result_batch)
		result[i * display_batch:(i + 1) * display_batch, :, :, :] = result_batch.cpu().detach().numpy()

	for h_idx in range(patch_per_line):
		for w_idx in range(patch_per_column):
			result_tensor[h_idx * patch_size: h_idx * patch_size + patch_size,
			w_idx * patch_size: w_idx * patch_size + patch_size, :] = result[h_idx * patch_per_column + w_idx, :, :, :]
	# result_HSI = result_tensor
	return result_tensor

def resize_spectral(HSI, H, W, C):
	resize_HSI = np.zeros((H, W, C))
	for i in range(C):
		resize_HSI[:, :, i] = cv2.resize(HSI[:, :, i], dsize=(H, W), interpolation=cv2.INTER_CUBIC)
	return resize_HSI


def HSI2RGB(HSI, band):
	# 光谱图像的RGB显示
	# (225,246,87)  该光谱图像是 空间维度225×246，光谱维度87（370nm~800nm 间隔5nm）
	# 创建转换器   illuminant='D50'表示D50下显示。支持光源包括A/B/C/D50~75，以及自定义光源
	# band 为波段参数，如370~800nm.间隔5nm 即 band = np.arange(370,805,5)； 若370~702nm.间隔4nm 即 band = np.arange(370,706,4)
	convertor = iv.spectra(illuminant='D50', band=band)
	Image = convertor.space(HSI, space='srgb')

	# 图像显示
	# plt.imshow(Image)
	# plt.show()
	return Image

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

