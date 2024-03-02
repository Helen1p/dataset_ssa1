# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import math
import random

# from utils import utils_image as util

from scipy import ndimage
import scipy
from scipy import special


###############加雾##############
# def add_fog(img, intensity=0.8):
#     '''
#     添加雾效果函数：
#     - img 是输入的原始图像。
#     - intensity 表示雾的浓度，取值范围为 0 至 1 的浮点数，表示雾层的不透明度。
#     '''
#     # img: float32
#     # 
#     if intensity < 0: intensity = 0
#     if intensity > 1: intensity = 1

#     # 创建与原始图像相同大小的雾层，这里我们用白色来模拟雾
#     # fog_layer = np.ones_like(img, dtype=np.uint8) * 255
#     fog_layer = np.ones_like(img, dtype=np.uint8)
#     # 根据雾的浓度调整雾层的不透明度
#     alpha = intensity  # alpha 是雾层的不透明度
#     beta = 1.0 - alpha  # beta 是原始图像的不透明度

#     # 叠加雾层到原始图像
#     # print(img.dtype)
#     fog_layer=fog_layer.astype(np.float32)

#     foggy_img = cv2.addWeighted(img, beta, fog_layer, alpha, 0.0)
#     # print(foggy_img.dtype)
#     return foggy_img


def add_fog(img):
    '''
    添加雾效果函数：
    - img 是输入的原始图像。
    - intensity 表示雾的浓度，取值范围为 0 至 1 的浮点数，表示雾层的不透明度。
    '''
    intensity = random.uniform(0,0.6)
    # img: float32
    img = img.astype(np.float32)
    if intensity < 0: intensity = 0
    if intensity > 1: intensity = 1

    # 创建与原始图像相同大小的雾层，这里我们用白色来模拟雾
    # fog_layer = np.ones_like(img, dtype=np.uint8) * 255
    fog_layer = np.ones_like(img, dtype=np.float32)
    # 根据雾的浓度调整雾层的不透明度
    alpha = intensity  # alpha 是雾层的不透明度
    beta = 1.0 - alpha  # beta 是原始图像的不透明度

    # 叠加雾层到原始图像
    # print(img.dtype)
    # fog_layer=fog_layer.astype(np.float32)

    foggy_img = cv2.addWeighted(img, beta, fog_layer, alpha, 0.0)
    # print(foggy_img.dtype)
    return foggy_img


##############加雨############
degtorad = 0.01745329252  # 定义角度转弧度的转换常数
radtodeg = 1/degtorad;  # 定义弧度转角度的转换常数

# 返回线条列表
def rand_lines(w,h,a,l,nrs):
    lines = []  # 初始化线条列表
    
    for i in range(nrs):  # 随机生成指定数量的线条
        # 随机化二维线条的起点和终点坐标
        sx = random.randint(0,w-1)
        sy = random.randint(0,h-1)
        
        le = random.randint(1,l)
        ang = a + random.randint(0,10)
        # le = 1时，sx=ex, sy=ey
        ex = sx + int(le * math.sin(ang * degtorad))
        ey = sy + int(le * math.cos(ang * degtorad))
        
        # 确保线条终点坐标在图像框架内
        if ex<0: 
            ex = 0
        if ex>w-1: 
            ex=w-1
        if ey<0: 
            ey = 0
        if ey>h-1: 
            ey=h-1
        
        # 将线条添加到列表中
        lines.append({'sx':sx,'sy':sy,'ex':ex,'ey':ey})
        
    return lines  # 返回线条列表

def add_rain(img):
    '''
    添加雨效果函数：
    - angle 应为 -90 至 90 之间的整数，表示雨滴下落的角度。
    - drop_length 为正值，表示雨滴的最大长度（像素单位），实际长度随机但不超过此长度，应与图像分辨率相匹配。
    - drop_thickness 为正值，表示雨滴的宽度（像素单位）。
    - drop_nrs 为正整数，表示要添加的雨滴数量。
    - blur 为模糊过滤的参数，取值范围为 1 至 10 的整数。
    - intensity 表示雨滴条纹的灰度强度，取值范围为 0 至 255 的整数。
    '''
    angle = random.randint(-30, 30)
    drop_length = random.randint(5, 70)
    # drop_length = 20
    drop_thickness = random.randint(5, 15)
    drop_nrs = random.randint(3, 50)
    blur = random.randint(3, 11)
    intensity = random.randint(20, 100)
    # 创建雨滴效果的占位图层
    rain = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='uint16')
    
    # 生成随机线条，模拟雨滴
    lines = rand_lines(rain.shape[1], rain.shape[0], angle, drop_length, drop_nrs)
    
    # 在图像上绘制线条
    for l in lines:
        cv2.line(rain, (l['sx'], l['sy']), (l['ex'], l['ey']), (intensity, intensity, intensity), drop_thickness)
    
    # 对线条应用模糊效果
    rain = cv2.blur(rain, (blur, blur))
    # img=img.astype(np.uint16)
    a=rain/225. + img
    # 观察
    # a=rain + img
    # 将雨滴效果叠加到原图像上
    return a


# basicsr/utils/img_process_util.py
def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


# basicsr/data/degradations.py
def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


# basicsr/utils/img_process_util.py
def get_aniso_sigma(sigma_x, sigma_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    D = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(U, np.dot(D, U.T))


# basicsr/data/degradations.py
def get_GaussianBlur_kernel(kernel_size, sig_x, sig_y=None, theta=None, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = get_aniso_sigma(sig_x, sig_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

    kernel = kernel / np.sum(kernel)
    return kernel


# basicsr/data/degradations.py
def get_generalized_kernel(kernel_size, sig_x, sig_y=None, theta=None, beta=1., grid=None, isotropic=True):
    """Generate a bivariate generalized Gaussian kernel.
    ``Paper: Parameter Estimation For Multivariate Generalized Gaussian Distributions``
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = get_aniso_sigma(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))

    kernel = kernel / np.sum(kernel)
    return kernel


# basicsr/data/degradations.py
def get_plateau_kernel(kernel_size, sig_x, sig_y=None, theta=None, beta=1., grid=None, isotropic=True):
    """Generate a plateau-like anisotropic kernel.
    1 / (1+x^(beta))
    Reference: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = get_aniso_sigma(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)

    kernel = kernel / np.sum(kernel)
    return kernel


# basicsr/data/degradations.py
def get_circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter
    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    np.seterr(divide='ignore', invalid='ignore')
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def add_iso_gaussian_blur(img, sigma_min=0.2, sigma_max=4., random=True, ksize=None):
    # RGB, 0-1, HxWxC
    if random:
        ksize = 2 * np.random.randint(2, 11) + 1
        sigma = np.random.uniform(sigma_min, sigma_max)
        kernel = get_GaussianBlur_kernel(ksize, sigma)
    else:
        assert ksize is not None, 'Please specific kernel size.'
        kernel = get_GaussianBlur_kernel(ksize, sigma)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_aniso_gaussian_blur(img, sigma_min=0.2, sigma_max=4., random=True, ksize=None, theta=None):
    # RGB, 0-1, HxWxC
    if random:
        ksize = 2 * np.random.randint(2, 11) + 1
        sigma = np.random.uniform(sigma_min, sigma_max)
        sigma2 = np.random.uniform(sigma_min, sigma_max)
        theta = np.random.uniform(-np.pi, np.pi)
        kernel = get_GaussianBlur_kernel(ksize, sigma, sigma2, theta, isotropic=False)
    else:
        assert ksize is not None, 'Please specific kernel size.'
        kernel = get_GaussianBlur_kernel(ksize, sigma_min, sigma_max, theta, isotropic=False)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_iso_generalized_blur(img, sigma_min=0.2, sigma_max=4., g_min=0.5, g_max=4.):
    # RGB, 0-1, HxWxC
    ksize = 2 * np.random.randint(2, 11) + 1
    sigma = np.random.uniform(sigma_min, sigma_max)
    beta = np.random.uniform(g_min, g_max)
    kernel = get_generalized_kernel(ksize, sigma, beta=beta)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_aniso_generalized_blur(img, sigma_min=0.2, sigma_max=4., g_min=0.5, g_max=4.):
    # RGB, 0-1, HxWxC
    ksize = 2 * np.random.randint(2, 11) + 1
    sigma = np.random.uniform(sigma_min, sigma_max)
    sigma2 = np.random.uniform(sigma_min, sigma_max)
    theta = np.random.uniform(-np.pi, np.pi)
    beta = np.random.uniform(g_min, g_max)
    kernel = get_generalized_kernel(ksize, sigma, sigma2, theta, beta=beta, isotropic=False)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_iso_plateau_blur(img, sigma_min=0.2, sigma_max=4., p_min=1., p_max=2.):
    # RGB, 0-1, HxWxC
    ksize = 2 * np.random.randint(2, 11) + 1
    sigma = np.random.uniform(sigma_min, sigma_max)
    beta = np.random.uniform(p_min, p_max)
    kernel = get_plateau_kernel(ksize, sigma, beta=beta)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_aniso_plateau_blur(img, sigma_min=0.2, sigma_max=4., p_min=1., p_max=2.):
    # RGB, 0-1, HxWxC
    ksize = 2 * np.random.randint(2, 11) + 1
    sigma = np.random.uniform(sigma_min, sigma_max)
    sigma2 = np.random.uniform(sigma_min, sigma_max)
    theta = np.random.uniform(-np.pi, np.pi)
    beta = np.random.uniform(p_min, p_max)
    kernel = get_plateau_kernel(ksize, sigma, sigma2, theta, beta=beta, isotropic=False)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


def add_sinc(img):
    kernel_size = 2 * np.random.randint(2, 11) + 1
    if kernel_size < 13:
        omega_c = np.random.uniform(np.pi / 3, np.pi)
    else:
        omega_c = np.random.uniform(np.pi / 5, np.pi)
    kernel = get_circular_lowpass_kernel(omega_c, kernel_size)

    img = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return img


## real-sr ##
def resize1(img):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 1.5)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.15, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def resize2(img):
    rnum = np.random.rand()
    if rnum > 0.7:  # up
        sf1 = random.uniform(1, 1.2)
    elif rnum < 0.4:  # down
        sf1 = random.uniform(0.3, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def task(img, order):
    # second-order
    # RGB, 0-1, HxWxC
    ori_h, ori_w, _ = img.shape
    distortion_type=[]
    if not order[0]:
        img = add_iso_gaussian_blur(img)
    else:
        rnum = np.random.rand()
        ## prob: [0.46, 0.22, 0.05, 0.22, 0.05] for other types
        if rnum < 0.46:
            img = add_aniso_gaussian_blur(img)
        elif rnum < 0.68:
            img = add_iso_generalized_blur(img)
        elif rnum < 0.73:
            img = add_aniso_generalized_blur(img)
        elif rnum < 0.95:
            img = add_iso_plateau_blur(img)
        else:
            img = add_aniso_plateau_blur(img)
    distortion_type.append('blur')
    
    img = resize1(img)
    distortion_type.append('resize')

    rnum = np.random.rand()
    if not order[1]:
        img = add_gaussian_noise(img)
    else:
        rnum = np.random.rand()
         # poisson : speckle = 0.8 : 0.2
        if rnum < 0.8:
            img = add_poisson_noise(img)
        else:
            img = add_speckle_noise(img)
    distortion_type.append('noise')

    img = add_JPEG_noise(img)
    distortion_type.append('jpeg_compression')

    if np.random.rand() < 0.8:
        if not order[2]:
            img = add_iso_gaussian_blur(img)
        else:
            rnum = np.random.rand()
            ## prob: [0.46, 0.22, 0.05, 0.22, 0.05] for other types
            if rnum < 0.46:
                img = add_aniso_gaussian_blur(img)
            elif rnum < 0.68:
                img = add_iso_generalized_blur(img)
            elif rnum < 0.73:
                img = add_aniso_generalized_blur(img)
            elif rnum < 0.95:
                img = add_iso_plateau_blur(img)
            else:
                img = add_aniso_plateau_blur(img)

        img = resize2(img)

        if not order[3]:
            img = add_gaussian_noise(img)
        else:
            rnum = np.random.rand()
            # poisson : speckle = 0.8 : 0.2
            if rnum < 0.8:
                img = add_poisson_noise(img)
            else:
                img = add_speckle_noise(img)

        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter + JPEG compression
            img = cv2.resize(img, (int(ori_w), int(ori_h)), interpolation=random.choice([1, 2, 3]))
            if np.random.rand() < 0.8:
                img = add_sinc(img)
                distortion_type.append('sinc')
            img = add_JPEG_noise(img)
        else:
            # JPEG compression + resize back + sinc
            img = add_JPEG_noise(img)
            img = cv2.resize(img, (int(ori_w), int(ori_h)), interpolation=random.choice([1, 2, 3]))
            if np.random.rand() < 0.8:
                img = add_sinc(img)
                distortion_type.append('sinc')

    else:
        img = cv2.resize(img, (int(ori_w), int(ori_h)), interpolation=random.choice([1, 2, 3]))

    ran = np.random.rand()
    # rain : fog : none = 0.2 : 0.2 : 0.6
    if ran < 0.2:
        img = add_rain(img)
        distortion_type.append('rain')
    elif ran < 0.4:
        img = add_fog(img)
        distortion_type.append('fog')
    else:
        pass
    
    # rounding
    img = np.clip((img * 255.).round(), 0, 255) / 255.
    type_ = (',').join(map(str, distortion_type))

    return img, type_


def add_gaussian_noise(img, sigma_min=5, sigma_max=30):
    # RGB, 0-1, HxWxC, only color noise
    sigma = np.random.randint(sigma_min, sigma_max)

    img += np.random.randn(*img.shape) * sigma / 255.
    img = img.astype(np.float32)

    img = np.clip(img, 0.0, 1.0)

    return img


def add_poisson_noise(img, scale_min=0.05, scale_max=3.):
    """Add poisson noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    scale = np.random.uniform(scale_min, scale_max)
    rnum = np.random.rand()
    img_ori = img.copy()
    if rnum < 0.4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = out - img
    if rnum < 0.4:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    
    noise *= scale

    img_ori += noise

    return np.clip(img_ori, 0, 1.)


def add_speckle_noise(img, sigma_min=5, sigma_max=30):
    noise_level = random.randint(sigma_min, sigma_max)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.4:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    else:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img, jpeg_min=30, jpeg_max=95):
    # RGB, 0-1, HxWxC
    quality_factor = random.randint(jpeg_min, jpeg_max)
    # quality_factor = jpeg
    img = np.clip((img*255.).round(), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h-lq_patchsize)
    rnd_w = random.randint(0, w-lq_patchsize)
    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
    return lq, hq


## hybrid distortion synthesis

def distortion_combination(num, min_l, max_l):
    """
    :param num: number of noise level for each distortion
    :param min_l: min sum of levels
    :param max_l: max sum of levels
    :return: a matrix, each row of which is a possible combination
    """

    vec = []
    for k in range(1, num+1):
        for m in range(k, num+1):
            for n in range(m, num+1):
                if min_l <= k + m + n - 2 <= max_l:
                    if k == n:
                        # Here you need to append new combinations to vec (the existing array),
                        # rather than use np.zeros() to build a new array.
                        vec.append([k, m, n])
                    elif k == m:
                        vec += [[k, m, n], [k, n, m], [n, k, m]]
                    elif m == n:
                        vec += [[k, m, n], [m, k, n], [m, n, k]]
                    else:
                        vec += [[k, m, n], [k, n, m], [m, k, n], [m, n, k], [n, k, m], [n, m, k]]

    return np.asarray(vec)


def hybrid_distortion(level, img, ksize):
    '''
    level: hybrid distortion level, should be an arraylike with shape 1x3
    img: input img, should be 0-1, RGB, HxWxC
    ksize: gaussian blur kernel size
    '''
    blur_sig = np.arange(0, 5.1, 0.5)
    noi_sig = np.arange(0, 51, 5)
    jpg_q = np.array([100, 80, 60, 50, 40, 35, 30, 25, 20, 15, 10])

    img_dis = img.copy()

    ## ------ add isotropic gaussian blur ------
    blur_idx = level[0] - 1  # there is a "-1" because python index starts from 0
    blur_sigma = blur_sig[blur_idx] + np.random.rand() * (blur_sig[blur_idx + 1] - blur_sig[blur_idx])
    kernel = get_GaussianBlur_kernel(ksize, blur_sigma)
    img_dis = cv2.filter2D(img_dis, -1, kernel).astype(np.float32)

    ## ------ add gaussian noise ------ 
    noi_idx = level[1] - 1
    noi_sigma = noi_sig[noi_idx] + np.random.rand() * ((noi_sig[noi_idx + 1]) - noi_sig[noi_idx])
    # color noise in hybrid distortion
    noise = np.random.randn(*img_dis.shape) * noi_sigma / 255.
    img_dis += noise
    # follow RealESRGAN, no rounding
    img_dis = np.clip(img_dis, 0, 1).astype(np.float32)

    ## ------ add JPEG -------
    jpg_idx = level[2] - 1
    quality = jpg_q[jpg_idx] + np.random.rand() * (jpg_q[jpg_idx + 1] - jpg_q[jpg_idx])
    img_dis = (img_dis * 255.).astype(np.uint8)
    img_dis = cv2.cvtColor(img_dis, cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_dis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    img_dis = cv2.imdecode(encimg, 1)
    img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)
    img_dis = img_dis.astype(np.float32) / 255.

    return img_dis


if __name__ == '__main__':
    img = np.random.uniform(0, 1, (32, 32, 3))
    img = cv2.imread('./EIQA/000002.png').astype(np.float32)[:, :, ::-1] / 255.
    imgt = img.copy()
    # imgt = add_Gaussian_noise(imgt)
    # print(np.max(imgt), np.min(imgt))
    # imgt = img.copy()
    # imgt = add_iso_gaussian_blur(imgt)
    # print(np.max(imgt), np.min(imgt))
    # imgt = img.copy()
    # imgt = add_aniso_gaussian_blur(imgt)
    # print(np.max(imgt), np.min(imgt))
    # imgt = img.copy()
    # imgt = add_mix_mild(imgt)
    # print(np.max(imgt), np.min(imgt))

    imgt = task(img, (1, 0, 1, 1))
    print(np.max(imgt), np.min(imgt))
    imgt = np.clip(imgt*255, 0, 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite('test.png', imgt)
