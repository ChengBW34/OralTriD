import h5py
import numpy as np
from PIL import Image
import os

def jpg_to_h5_single(image_path, h5_path, dataset_name='image'):
    img = Image.open(image_path)
    img_array = np.array(img)

    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=img_array, compression="gzip")

    print(f"{image_path} saved as {h5_path}")

def images_to_h5(dir_path, image_list, h5_path, dataset_name='image'):
    image_files = [os.path.join(dir_path, f) for f in image_list]
    num_images = len(image_files)
    img = Image.open(image_files[0]).convert('L')
    width, height = img.size
    images_array = np.zeros((height, width, num_images), dtype=np.uint8)

    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path).convert('L')
        images_array[:, :, idx] = np.array(img)

    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=images_array, compression="gzip")

    print(f"✅ successfully saved {num_images} images to {h5_path}")


if __name__ == '__main__':
    dir_name = 'data_train_val_split'
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