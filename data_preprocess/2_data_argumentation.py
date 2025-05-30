import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from random import choice
import pickle

cla_tmp = 'OBC'
cla = 'OBC'
save_dir = 'data_argumentation'
dir_path = 'data_train_val_split'

if cla == 'OLP':
    overlap_threshold = 0.5
elif cla == 'OLK':
    overlap_threshold = 0.5
elif cla == 'OBC':
    overlap_threshold = 0.09
    trans_num = 2

image_save_path_OLP = f'../{dir_path}/{save_dir}/imgs/OLP'
mask_save_path_OLP = f'../{dir_path}/{save_dir}/masks/OLP'

image_save_path_OLK = f'../{dir_path}/{save_dir}/imgs/OLK'
mask_save_path_OLK = f'../{dir_path}/{save_dir}/masks/OLK'

image_save_path_OBC = f'../{dir_path}/{save_dir}/imgs/OBC'
mask_save_path_OBC = f'../{dir_path}/{save_dir}/masks/OBC'

image_save_path = f'../{dir_path}/{save_dir}/imgs'
mask_save_path = f'../{dir_path}/{save_dir}/masks'

if not os.path.exists(image_save_path_OLP):
    os.makedirs(image_save_path_OLP)
if not os.path.exists(mask_save_path_OLP):
    os.makedirs(mask_save_path_OLP)

if not os.path.exists(image_save_path_OLK):
    os.makedirs(image_save_path_OLK)
if not os.path.exists(mask_save_path_OLK):
    os.makedirs(mask_save_path_OLK)

if not os.path.exists(image_save_path_OBC):
    os.makedirs(image_save_path_OBC)
if not os.path.exists(mask_save_path_OBC):
    os.makedirs(mask_save_path_OBC)


def resized_origal(image, mask, name, cla):
    height, width = image.shape[:2]

    left = int((3200 - width) / 2)
    right = 3200 - width - left
    top = int((2240 - height) / 2)
    bottom = 2240 - height - top

    # 填充为黑色
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

    resized_image = cv2.resize(padded_image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(padded_mask, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    unique_values = np.unique(resized_mask)

    _, resized_mask = cv2.threshold(resized_mask, unique_values[-2], 255, cv2.THRESH_BINARY)
    unique_values = np.unique(resized_mask)
    cv2.imwrite(os.path.join(image_save_path, cla, name + '.jpg'), resized_image)
    cv2.imwrite(os.path.join(mask_save_path, cla, name + '.png'), resized_mask)

    if name != 'OBC':
        Rotation(resized_image, resized_mask, f'{name}', cla)
        ScaleAbs(resized_image, resized_mask, f'{name}', cla)


def cut_image(image, mask):
    h, w = image.shape[:2]
    win_h, win_w = 448, 640
    stride = 100
    crop_image_all = []
    binary_image_all = []

    # 遍历图像并提取子图像
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            window = (y, x, win_h, win_w)
            overlap_ratio = calculate_overlap(window, mask)

            if overlap_ratio > overlap_threshold:
                crop_image = image[y:y + win_h, x:x + win_w]
                crop_label = mask[y:y + win_h, x:x + win_w]
                unique_values = np.unique(crop_label)
                _, binary_image = cv2.threshold(crop_label, unique_values[-2], unique_values[-1], cv2.THRESH_BINARY)
                unique_values = np.unique(binary_image)
                crop_image_all.append(crop_image)
                binary_image_all.append(binary_image)

    if len(crop_image_all) > 0:
        r = random.randint(0, len(crop_image_all) - 1)
        max_img = np.max(binary_image_all[r])
        unique_values = np.unique(binary_image_all[r])
        return crop_image_all[r], binary_image_all[r]
    else:
        return None, None

def calculate_overlap(window, label):
    # 获取窗口和目标区域的坐标
    win_y, win_x, win_h, win_w = window
    window_image = label[win_y:win_y + win_h, win_x:win_x + win_w]
    unique_values = np.unique(window_image)
    classes_nums = np.zeros([256], int)
    classes_nums += np.bincount(np.reshape(window_image, [-1]), minlength=256)
    pixel = []
    for x in classes_nums:
        if x !=0 :
            pixel.append(x)
    if len(pixel) == 2:
        num_0 = pixel[0]
        num_255 = pixel[1]
        ratio = num_255 / num_0
    else:
        ratio = 0
    return ratio

def ScaleAbs(image, mask, name, cla):
    for i in range(2):
        beta = random.randint(-50, 50)
        # 生成增强图像
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        unique_values = np.unique(mask)
        cv2.imwrite(os.path.join(image_save_path, cla, f'{name}_s{i}.jpg'), enhanced_image)
        cv2.imwrite(os.path.join(mask_save_path, cla, f'{name}_s{i}.png'), mask)

def Rotation(image, mask, name, cla):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    i = 1
    unique_values = np.unique(mask)
    for angle in [45, 135]:
        scale = 1.5  # 缩放因子 (1.0表示不缩放)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (new_w, new_h))
        rotated_image, rotated_mask = cut_image(rotated_image, rotated_mask)
        if rotated_image is not None:
            unique_values = np.unique(rotated_mask)
            cv2.imwrite(os.path.join(image_save_path, cla, f'{name}_r{i}.jpg'), rotated_image)
            cv2.imwrite(os.path.join(mask_save_path, cla, f'{name}_r{i}.png'), rotated_mask)
        i = i + 1

def Translation(image, mask, name, cla):
    i = 1
    txs = [-400, -200, 0, 200, 400]
    tys = [-200, -100, 0, 100, 200]
    while i < trans_num:
        tx = choice(txs)  # 水平平移
        ty = choice(tys)  # 垂直平移

        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        translated_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))
        overlap_ratio = calculate_overlap([0, 0, 448, 640], translated_mask)

        if overlap_ratio > overlap_threshold:
            unique_values = np.unique(translated_mask)
            cv2.imwrite(os.path.join(image_save_path, cla, f'{name}_t{i}.jpg'), translated_image)
            cv2.imwrite(os.path.join(mask_save_path, cla, f'{name}_t{i}.png'), translated_mask)
        i = i + 1

def sliding_window_crop_with_filter(image, label, name, window_size, stride, cla):
    h, w = image.shape[:2]
    win_h, win_w = window_size

    unique_values = np.unique(label)
    # 遍历图像并提取子图像
    i = 1
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            window = (y, x, win_h, win_w)
            overlap_ratio = calculate_overlap(window, label)

            # 只有重叠占比大于阈值的窗口才会被保留下来
            if overlap_ratio > overlap_threshold:
                crop_image = image[y:y + win_h, x:x + win_w]
                crop_label = label[y:y + win_h, x:x + win_w]

                resized_crop_image = cv2.resize(crop_image, (640, 448))
                resized_binary_image = cv2.resize(crop_label, (640, 448))
                unique_values = np.unique(resized_binary_image)
                _, resized_binary_image = cv2.threshold(resized_binary_image, unique_values[-2], unique_values[-1], cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(image_save_path, cla, f'{name}_{i}.jpg'), resized_crop_image)
                cv2.imwrite(os.path.join(mask_save_path, cla, f'{name}_{i}.png'), resized_binary_image)

                if cla == 'OBC':
                    Rotation(resized_crop_image, resized_binary_image, f'{name}_{i}', cla)
                    # Translation(resized_crop_image, resized_binary_image, f'{name}_{i}', cla)
                    ScaleAbs(resized_crop_image, resized_binary_image, f'{name}_{i}', cla)
                i = i + 1


if __name__ == '__main__':
    # 读取图像
    dir_path = '../datasets'
    images = os.listdir(f'{dir_path}/JPEGimages/{cla_tmp}')

    for image in images:
        print(image)
        img = cv2.imread(f'./{dir_path}/JPEGimages/{cla_tmp}/{image}')
        label = cv2.imread(f'./{dir_path}/SegmentationClass/{cla_tmp}/{image[:-4]}.png')
        unique_values = np.unique(label)
        if img.shape[0]!=3008 and img.shape[1]!=3008:
            if img.shape[0] > img.shape[1]:
                new_size = (2000, 3008)
            else:
                new_size = (3008, 2000)
            img = cv2.resize(img, new_size)
            label = cv2.resize(label, new_size)
            unique_values = np.unique(label)
            _, label = cv2.threshold(label, unique_values[-2], unique_values[-1], cv2.THRESH_BINARY)
        if img.shape[0]==3008:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
        unique_values = np.unique(label)
        resized_origal(img, label, image[:-4], cla)
        # 设置窗口大小 (高度, 宽度) 和步长
        if cla == 'OBC':
            window_size = (1344, 1920)
            stride = 600
        else:
            window_size = (1972, 2560)
            stride = 200
        sliding_window_crop_with_filter(img, label, image[:-4], window_size, stride, cla)