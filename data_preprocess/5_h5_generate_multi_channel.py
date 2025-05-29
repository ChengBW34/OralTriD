import h5py
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def jpg_to_h5_single(image_path, h5_path, dataset_name='image'):
    # 读取图片并转换为 RGB
    # img = Image.open(image_path).convert('RGB')
    img = Image.open(image_path)
    # 转换为 NumPy 数组
    img_array = np.array(img)
    # plt.imshow(img_array)
    # plt.show()
    # 保存到 H5 文件
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=img_array, compression="gzip")

    print(f"已将 {image_path} 保存为 {h5_path}")

def images_to_h5(dir_path, image_list, h5_path, dataset_name='image'):
    # 获取所有图片路径（支持 JPG/PNG）
    image_files = [os.path.join(dir_path, f) for f in image_list]
    num_images = len(image_files)
    img = Image.open(image_files[0]).convert('L')
    width, height = img.size
    # 预分配数组 (N, H, W, C)
    images_array = np.zeros((height, width, num_images), dtype=np.uint8)

    # 读取图片并填入数组
    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path).convert('L')
        images_array[:, :, idx] = np.array(img)

    # 写入 H5 文件
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=images_array, compression="gzip")

    print(f"✅ 成功将 {num_images} 张图片保存到 {h5_path}")


# ✅ 示例：将单张 JPG 转换为 H5
if __name__ == '__main__':
    dir_name = 'data_train_val_808'
    dir_path_img = f'../{dir_name}/data_argumentation/imgs'
    dir_path_mask = f'../{dir_name}/data_argumentation/multi_channel_mask'
    h5_path = f'../{dir_name}/glossopathy_h5'
    segments = ['OLP', 'OLK', 'OBC']
    if not os.path.exists(h5_path):
        os.makedirs(h5_path)

    for segment in segments:
        filenames = os.listdir(os.path.join(dir_path_img, segment))
        for filename in filenames:
            # if os.path.exists(os.path.join(h5_path, filename[:-4]+'.h5')):
            #     continue
            if filename.lower().endswith(('.jpg')):
                jpg_to_h5_single(os.path.join(dir_path_img, segment, filename), os.path.join(h5_path, filename[:-4]+'.h5'))

            filename1 = filename[:-4] + '_OLP.png'
            filename2 = filename[:-4] + '_OLK.png'
            filename3 = filename[:-4] + '_OBC.png'
            file_list = [filename1, filename2, filename3]
            images_to_h5(os.path.join(dir_path_mask, segment), file_list,
                         os.path.join(h5_path, filename[:-4] + '_all.h5'))