import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

def train_test_split_add(train_val_split, train_or_val, ID, filedir):
    train_val_split["Split"].append(train_or_val)
    train_val_split["ID"].append(ID)
    train_val_split["class"].append(filedir)

    if filedir == "OLP":
        train_val_split["OLP"].append(1)
        train_val_split["OLK"].append(0)
        train_val_split["OBU"].append(0)
    elif filedir == "OLK":
        train_val_split["OLP"].append(0)
        train_val_split["OLK"].append(1)
        train_val_split["OBU"].append(0)
    elif filedir == "OBU":
        train_val_split["OLP"].append(0)
        train_val_split["OLK"].append(0)
        train_val_split["OBU"].append(1)

    return train_val_split

if __name__ == "__main__":
    random.seed(1)
    print("Generate txt in ImageSets.")

    train_percent = 0.8
    val_percent = 0.2
    Glos_path = '../datasets/JPEGimages'
    argumentation_path = '../data_train_val_split/data_argumentation/imgs'
    save_path = '../data_train_val_split'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tv_all = 0
    tr_all = 0

    temp_segs = os.listdir(Glos_path)

    classes_nums = np.zeros([256], int)
    train_val_split = {
        "ID": [],
        "class": [],
        "Split": [],
        "OLP": [],
        "OLK": [],
        "OBU": []
    }
    for filedir in temp_segs:
        argumentation_seg = os.listdir(os.path.join(argumentation_path, filedir))
        temp_seg = os.listdir(os.path.join(Glos_path, filedir))
        total_seg = []
        for seg in temp_seg:
            if seg.endswith(".jpg"):
                total_seg.append(seg)
        num = len(total_seg)
        list = range(num)
        num_train = int(num*train_percent)
        num_val = num - num_train

        train_list = random.sample(list, num_train)
        val_list = [x for x in list if x not in train_list]

        for i in tqdm(list):
            name = total_seg[i]
            if i in train_list:
                train_val_split = train_test_split_add(train_val_split, 'train', name[:-4], filedir)
                print('train ', name, 'add!')
                for a_name in argumentation_seg:
                    if '_' in a_name:
                        if a_name.split('_')[0]==name.split('.')[0]:
                            train_val_split = train_test_split_add(train_val_split, 'train', a_name[:-4], filedir)
                            print('train ', a_name, 'add!')

            elif i in val_list:
                train_val_split = train_test_split_add(train_val_split, 'val', name[:-4], filedir)
                print('val ', name, 'add!')
                for a_name in argumentation_seg:
                    if '_' in a_name:
                        if a_name.split('_')[0]==name.split('.')[0]:
                            train_val_split = train_test_split_add(train_val_split, 'val', a_name[:-4], filedir)
                            print('val ', a_name, 'add!')

    df = pd.DataFrame(train_val_split)
    train_nums = len(df[df['Split'] == 'train'].ID.values)
    val_nums = len(df[df['Split'] == 'val'].ID.values)
    print('train_nums: ', train_nums, 'val_nums: ', val_nums)
    # 保存为 pickle 文件
    df.to_pickle(f'{save_path}/train_val_split.pkl')