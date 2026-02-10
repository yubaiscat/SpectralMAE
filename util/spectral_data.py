import pathlib
import random

import cv2
import h5py
import imgvision as iv
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class NTIREDataset(Dataset):
    def __init__(self, opts):
        self.arg = True
        self.opts = opts
        self.patch_size = opts['patch_size']
        self.stride = opts['stride']
        hyper_list = list(pathlib.Path(opts['data_path']).glob('*.mat'))
        hyper_list.sort()
        print(f'length of dataset:{len(hyper_list)}')
        self.hypers = []
        self.mean, self.std = load_mean_std(opts['mean_std_path'])
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                for c in range(opts['in_chans']):
                    hyper[c, :, :] = (hyper[c, :, :] - self.mean[c]) / self.std[c]
                self.hypers.append(hyper)
                mat.close()
                print('Load hyper image ', i)
        self.img_num = len(self.hypers)

        if opts['load_state']:
            self.height = opts['height']
            self.width = opts['width']
            self.in_chans = opts['in_chans']
            self.patch_per_line = (self.width - self.patch_size) // self.stride + 1
            self.patch_per_colum = (self.height - self.patch_size) // self.stride + 1
            self.patch_per_img = self.patch_per_line * self.patch_per_colum
            self.band = np.arange(opts['band'][0], opts['band'][1], opts['band'][2])
        else:
            self.get_spectral_state()
            
        self.load_test_data()

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        patch_size = self.patch_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_colum, patch_idx % self.patch_per_colum

        hyper = self.hypers[img_idx]
        hyper = hyper[:, h_idx * stride: h_idx * stride + patch_size, w_idx * stride: w_idx * stride + patch_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img * self.img_num

    def get_spectral_state(self):
        hyper = self.hypers[0]
        self.in_chans = hyper.shape[0]
        self.width = hyper.shape[1]
        self.height = hyper.shape[2]
        self.patch_per_line = (self.width - self.patch_size) // self.stride + 1
        self.patch_per_colum = (self.height - self.patch_size) // self.stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum
        self.band = np.arange(self.opts['band'][0], self.opts['band'][1], self.opts['band'][2])
        
    def get_mean_std(self):
        '''
        Compute mean and variance for training data
        :return: (mean, std)
        '''

        print('Compute mean and variance for training data.')
        train_loader = DataLoader(dataset=self.hypers, batch_size=1, shuffle=False, pin_memory=True)
        mean = torch.zeros(self.in_chans)
        std = torch.zeros(self.in_chans)
        for X in train_loader:
            for c in range(self.in_chans):
                mean[c] += X[:, c, :, :].mean()
                std[c] += X[:, c, :, :].std()
        mean.div_(self.img_num)
        std.div_(self.img_num)
        self.mean = list(mean.numpy())
        self.std = list(std.numpy())
        print(self.mean)
        print(self.std)
    
    def load_test_data(self):
        hyper_list = list(pathlib.Path(self.opts['test_path']).glob('*.mat'))
        hyper_list.sort()
        self.test_hypers = []
        print(f'length of test dataset:{len(hyper_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                for c in range(self.in_chans):
                    hyper[c, :, :] = (hyper[c, :, :] - self.mean[c]) / self.std[c]
                self.test_hypers.append(hyper)
                mat.close()
                print('Load hyper image ', i)
        
        self.img_test_num = len(self.test_hypers)
        self.display_height = self.opts['display_height']
        self.display_width = self.opts['display_width']
        self.display_patch_per_column = self.display_height // self.patch_size
        self.display_patch_per_line = self.display_width// self.patch_size
        assert self.display_width % self.patch_size == 0
        assert self.display_height % self.patch_size == 0
        self.diplay_batch_num = self.display_patch_per_line * self.display_patch_per_column
        self.display_batch = self.opts['display_batch']
        assert self.diplay_batch_num % self.display_batch == 0
        
    def random_model_test(self, model):
        model.eval()
        with torch.no_grad():
            rand = random.randint(0, len(self.test_hypers)-1)
            HSI = self.test_hypers[rand].copy()
            HSI = np.einsum('chw->hwc', HSI)
            result_HSI = self.run_one_image(HSI, model)
            for c in range(self.in_chans):
                result_HSI[:,:,c] = (result_HSI[:, :, c] * self.std[c] + self.mean[c]) 
            result = HSI2RGB(result_HSI, self.band)

            for c in range(self.in_chans):
                HSI[:, :, c] = (HSI[:, :, c] * self.std[c] + self.mean[c])
            origin = HSI2RGB(HSI, self.band)
            return origin, result, HSI, result_HSI
            
    def run_one_image(self, HSI, model):
        rHSI = resize_spectral(HSI, self.display_height, self.display_width, self.in_chans)
        batch = np.zeros((self.diplay_batch_num, self.patch_size, self.patch_size, self.in_chans))
        for h_idx in range(self.display_patch_per_line):
            for w_idx in range(self.display_patch_per_column):
                batch[h_idx * self.display_patch_per_column + w_idx, :, :, :] = rHSI[h_idx * self.patch_size: h_idx * self.patch_size + self.patch_size,
															   w_idx * self.patch_size: w_idx * self.patch_size + self.patch_size, :]
        batch_tensor = torch.tensor(batch).float().to(self.opts['device'], non_blocking=True)
        result = np.zeros(batch_tensor.shape)
        result_tensor = np.zeros(rHSI.shape)
        batch_tensor = torch.einsum('nhwc->nchw', batch_tensor)

	    # make it a batch-like
        batch_time = self.diplay_batch_num // self.display_batch 
        for i in range(batch_time):
            batch = batch_tensor[i * self.display_batch:(i + 1) * self.display_batch, :, :, :]
            loss, result_batch, mask_batch = model(batch, mask_ratio=self.opts['mask_ratio'])
            result_batch = model.unpatchify(result_batch)
            result_batch = torch.einsum('nchw->nhwc', result_batch)
            result[i * self.display_batch:(i + 1) * self.display_batch, :, :, :] = result_batch.cpu().detach().numpy()
        
        for h_idx in range(self.display_patch_per_line):
            for w_idx in range(self.display_patch_per_column):
                result_tensor[h_idx * self.patch_size: h_idx * self.patch_size + self.patch_size,
                              w_idx * self.patch_size: w_idx * self.patch_size + self.patch_size, :] = result[h_idx * self.display_patch_per_column + w_idx, :, :, :]
        
        return result_tensor 

class SpectralDataset(Dataset):
    def __init__(self, opts):
        self.arg = True
        self.opts = opts
        self.patch_size = opts['patch_size']
        self.stride = opts['stride']
        self.order = opts['order']
        self.mean =  opts['mean']
        self.std =  opts['std']
        hyper_list = list(pathlib.Path(opts['data_path']).glob('*.mat'))
        hyper_list.sort()
        print(f'length of dataset:{len(hyper_list)}')
        self.hypers = []
        # self.mean, self.std = load_mean_std(opts['mean_std_path'])
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            mat = sio.loadmat(hyper_path)
            hyper = mat.get(opts['data_name'])
            hyper = np.float32(hyper)
            if self.order != 'chw':
                trans = self.order + '->chw'
                hyper = np.einsum(trans, hyper)
            hyper = (hyper - self.mean) * self.std
            self.hypers.append(hyper)
            print('Load hyper image ', i)
        self.img_num = len(self.hypers)

        if opts['load_state']:
            self.height = opts['height']
            self.width = opts['width']
            self.in_chans = opts['in_chans']
        else:
            hyper = self.hypers[0]
            self.in_chans = hyper.shape[0]
            self.width = hyper.shape[1]
            self.height = hyper.shape[2]
    
        self.patch_per_line = (self.width - self.patch_size) // self.stride + 1
        self.patch_per_colum = (self.height - self.patch_size) // self.stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum
        self.band = np.arange(opts['band'][0], opts['band'][1], opts['band'][2])
        self.load_test_data()

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1]
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :]
        return img.copy()

    def __getitem__(self, idx):
        stride = self.stride
        patch_size = self.patch_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_colum, patch_idx % self.patch_per_colum
        hyper = self.hypers[img_idx]
        hyper = hyper[:, h_idx * stride: h_idx * stride + patch_size, w_idx * stride: w_idx * stride + patch_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(hyper)


    def __len__(self):
        return self.patch_per_img * self.img_num

    def get_spectral_state(self):
        hyper = self.hypers[0]
        self.in_chans = hyper.shape[0]
        self.width = hyper.shape[1]
        self.height = hyper.shape[2]
        self.patch_per_line = (self.width - self.patch_size) // self.stride + 1
        self.patch_per_colum = (self.height - self.patch_size) // self.stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum
        self.band = np.arange(self.opts['band'][0], self.opts['band'][1], self.opts['band'][2])
        self.RGB = self.opts['RGB']
    
    def load_test_data(self):
        hyper_list = list(pathlib.Path(self.opts['test_path']).glob('*.mat'))
        hyper_list.sort()
        self.test_hypers = []
        print(f'length of test dataset:{len(hyper_list)}')

        self.RGB = self.opts['RGB']
        for i in range(len(hyper_list)):
            hyper_path = hyper_list[i]
            mat = sio.loadmat(hyper_path)
            hyper = mat.get(self.opts['data_name'])
            hyper = np.float32(hyper)
            # hyper = resize_spectral(hyper, self.display_height, self.display_width, self.in_chans)
            if self.order != 'chw':
                trans = self.order + '->chw'
                hyper = np.einsum(trans, hyper)
            hyper = (hyper - self.mean) * self.std
            
            self.test_hypers.append(hyper)
            print('Load test hyper image ', i)
            
        self.img_test_num = len(self.test_hypers)
        # self.display_height = self.opts['display_height']
        # self.display_width = self.opts['display_width']
        self.display_height = hyper.shape[1]
        self.display_width = hyper.shape[2]
        self.display_patch_per_line = self.display_height // self.patch_size
        self.display_patch_per_column = self.display_width // self.patch_size
        assert self.display_width % self.patch_size == 0
        assert self.display_height % self.patch_size == 0
        self.diplay_batch_num = self.display_patch_per_line * self.display_patch_per_column
        self.display_batch = self.opts['display_batch']
        assert self.diplay_batch_num % self.display_batch == 0

        
    def random_model_test(self, model):
        model.eval()
        with torch.no_grad():
            rand = random.randint(0, len(self.test_hypers)-1)
            HSI = self.test_hypers[rand]

            result_HSI = self.run_one_image(HSI, model)
            result = visual_fake_hyper(result_HSI, self.RGB)

            HSI = HSI / self.std + self.mean
            origin = visual_fake_hyper(HSI, self.RGB)
            return origin, result, HSI, result_HSI
            
    def run_one_image(self, HSI, model):
        batch = np.zeros((self.diplay_batch_num, self.in_chans, self.patch_size, self.patch_size))
        for h_idx in range(self.display_patch_per_line):
            for w_idx in range(self.display_patch_per_column):
                batch[h_idx * self.display_patch_per_column + w_idx, :, :, :] = HSI[:, h_idx * self.patch_size: h_idx * self.patch_size + self.patch_size,
															   w_idx * self.patch_size: w_idx * self.patch_size + self.patch_size]
        batch_tensor = torch.tensor(batch).float().to(self.opts['device'], non_blocking=True)
        result = np.zeros(batch_tensor.shape)
        result_tensor = np.zeros(HSI.shape)

	    # make it a batch-like
        batch_time = self.diplay_batch_num // self.display_batch 
        for i in range(batch_time):
            batch = batch_tensor[i * self.display_batch:(i + 1) * self.display_batch, :, :, :]
            loss, result_batch, mask_batch = model(batch, mask_ratio=self.opts['mask_ratio'])
            result_batch = model.unpatchify(result_batch)
            result_batch = result_batch / self.std + self.mean
            result[i * self.display_batch:(i + 1) * self.display_batch, :, :, :] = result_batch.cpu().detach().numpy()
        
        for h_idx in range(self.display_patch_per_line):
            for w_idx in range(self.display_patch_per_column):
                result_tensor[:, h_idx * self.patch_size: h_idx * self.patch_size + self.patch_size,
                              w_idx * self.patch_size: w_idx * self.patch_size + self.patch_size] = result[h_idx * self.display_patch_per_column + w_idx, :, :, :]
        
        return result_tensor

def build_dataset(opts):
    print("---------------------------")
    if opts['data_set'] == "NTIRE":
        print("Lode from folder %s", opts['data_path'])
        dataset = NTIREDataset(opts)
    elif opts['data_set'] == "Spectral":
        print("Lode from folder %s", opts['data_path'])
        dataset = SpectralDataset(opts)
    else:
        raise NotImplementedError()
    return dataset

def load_mean_std(filepath):
    norm_data = np.loadtxt(filepath)
    mean = norm_data[:, 0]
    std = norm_data[:, 1]
    return mean, std


def load_wavelength(filepath):
    wavelength_data = np.loadtxt(filepath)
    idx = wavelength_data[:, 0]
    wavelength = wavelength_data[:, 1]
    return idx, wavelength

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
    
def visual_fake_hyper(HSI, RGB_chanels):
    R = HSI[RGB_chanels[0], :, :]
    G = HSI[RGB_chanels[1], :, :]
    B = HSI[RGB_chanels[2], :, :]
    img = np.stack((R, G, B), axis=0)
    img = np.einsum('chw->hwc', img)
    # img *= 255
    # img = Image.fromarray(np.uint8(img))
    return img