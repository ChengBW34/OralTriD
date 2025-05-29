import json
import numpy as np
import cv2
import os
import shutil

cla = 'OBC'
source_image = f"../OralTriD/{cla}/Images"
jpgs_path = f"../datasets/JPEGimages/{cla}"
pngs_path = f"../datasets/SegmentationClass/{cla}"
if not os.path.exists(jpgs_path):
    os.makedirs(jpgs_path)
if not os.path.exists(pngs_path):
    os.makedirs(pngs_path)

count = os.listdir(f"../OralTriD/{cla}/Annotations")
for i in range(0, len(count)):
    print(count[i])
    path = os.path.join(f"../OralTriD/{cla}/Annotations", count[i])

    if os.path.isfile(path) and path.endswith('json'):
        data = json.load(open(path))

    # 获取图像尺寸
    image_height = data['imageHeight']
    image_width = data['imageWidth']

    # 创建空白mask图像
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 绘制多边形区域
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        value = 255
        cv2.fillPoly(mask, [points], value)

    
    # 保存Mask图像
    cv2.imwrite(f'{pngs_path}/{count[i].split(".")[0]}.png', mask)
    shutil.copy(os.path.join(source_image, count[i].split(".")[0]+'.jpg'), os.path.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
