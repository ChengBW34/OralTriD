import numpy as np
import torch
import h5py
import torchvision
from torchvision.utils import save_image
import os
from utils.metrics import TestMetrics
import pandas as pd
from models.Unet import Unet
from models.FCN import FCNs
from models.deeplabv3 import DeepLabV3
from models.atten_unet import AttU_Net
from models.segnet import SegNet


def get_split(train_test_split_file):
    train_test_id = pd.read_csv(train_test_split_file, index_col=0)
    train_test_id['total'] = train_test_id[['OLP', 'OLK', 'OBU']].sum(axis=1)
    return train_test_id

def load_mask(image_path, img_id):
    mask_file = image_path + '%s_all.h5' % (img_id)
    f = h5py.File(mask_file, 'r')
    mask_np = f['image'][()]
    mask_np = (mask_np/255).astype('uint8')
    return mask_np

def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['image'][()]
    img_np = (img_np / 255.0).astype('float32')
    return img_np

def single_model_predict(model_name, model_saved):
    dir_name = 'data_train_val_split'
    csv_name = 'results_808.csv'
    train_test_split_file = f'./{dir_name}/train_val_split.csv'
    train_test_id = get_split(train_test_split_file)
    train_test_id = train_test_id[train_test_id['Split'] == 'val']
    train_test_id = train_test_id.ID.values
    image_path = f'./{dir_name}/glossopathy_h5/'
    mask_ind = pd.read_csv(train_test_split_file, index_col=0)
    mask_ind = pd.DataFrame(mask_ind)
    mask_ind = mask_ind.loc[mask_ind['Split'] == 'val'].values
    mask_ind = mask_ind[:, 3:]
    mask_ind = pd.DataFrame(mask_ind)
    device = torch.device('cuda')
    savefig_path = f'./predict/{model_name}/{model_saved}/predict_output'
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)

    if model_name == 'Unet':
        model = Unet(in_channels=3, classes=3).to(device)
    elif model_name == 'deeplabv3':
        model = DeepLabV3(in_class=3, class_num=3).to(device)
    elif model_name == 'FCN':
        model = FCNs(n_class=3).to(device)

    elif model_name == 'atten_unet':
        model = AttU_Net(img_ch=3, output_ch=3).to(device)

    elif model_name == 'segnet':
        model = SegNet(input_nbr=3, label_nbr=3).to(device)

    state_dict = torch.load(f'./logs/{model_saved}/best_epoch_weights.pth')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # 去除 'module.' 前缀
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    alpha = 0.3
    meter = TestMetrics()
    meter.reset()
    model.eval()

    print(f'{model_name} start predict ...')
    with torch.no_grad():
        for index in range(train_test_id.shape[0]):
            img_id = train_test_id[index]
            print(index, img_id)
            ### load image
            image_file = image_path + '%s.h5' % img_id
            img_np = load_image(image_file)
            ### load masks
            mask_np = load_mask(image_path, img_id)
            ind = mask_ind.loc[index, :].values.astype('uint8')

            train_image = np.expand_dims(img_np, 0)
            train_mask = np.expand_dims(mask_np, 0)
            train_ind = np.expand_dims(ind, 0)
            train_image = torch.from_numpy(train_image).to(device)
            train_image = train_image.permute(0, 3, 1, 2)
            train_mask = torch.from_numpy(train_mask).to(device).type(torch.cuda.FloatTensor)
            train_mask = train_mask.permute(0, 3, 1, 2)
            train_mask_ind = torch.from_numpy(train_ind).to(device).type(torch.cuda.FloatTensor)

            outputs, outputs_mask_ind1 = model(train_image)
            train_prob = torch.sigmoid(outputs)
            train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)

            meter.add(train_prob, train_mask, train_mask_ind_prob1, train_mask_ind)

            cat_mask = torch.cat([train_mask[0], train_prob[0]], dim=0)
            cat_mask = cat_mask.unsqueeze(1)
            saved_masks = torchvision.utils.make_grid(cat_mask, nrow=3, padding=10, pad_value=1)
            save_image(saved_masks, os.path.join(savefig_path, img_id + '_mask.png'))

            output_mask = train_prob[0][np.argmax(ind)].unsqueeze(0).repeat(3, 1, 1)
            output_mask[output_mask >= 0.5] = 1
            output_mask[output_mask < 0.5] = 0
            image_pre = train_image[0] * (1 - alpha) + output_mask * alpha

            gt_mask = train_mask[0][np.argmax(ind)].unsqueeze(0).repeat(3, 1, 1)
            image_gt = train_image[0] * (1 - alpha) + gt_mask * alpha

            cat_images = torch.cat([image_gt.unsqueeze(0), image_pre.unsqueeze(0)], 0)
            saved_images = torchvision.utils.make_grid(cat_images, nrow=cat_images.shape[0], padding=10, pad_value=1)
            save_image(saved_images, os.path.join(savefig_path, img_id + f'_image.png'))
    metrics = meter.value()
    print(metrics)
    jaccard = metrics['jaccard']
    miou = metrics['miou']
    Recall = metrics['Recall']
    precision = metrics['precision']
    accuracy = metrics['accuracy']
    prob1_acc = metrics['prob1_acc']
    dice = metrics['dice']
    F1_score = metrics['F1-score']
    class_acc1 = metrics['class_acc'][0]
    class_acc2 = metrics['class_acc'][1]
    class_acc3 = metrics['class_acc'][2]

    if not os.path.exists(csv_name):
        results = {
            'model_name': model_name,
            'model_saved': model_saved,
            'jaccard': jaccard,
            'miou': miou,
            'Recall': Recall,
            'precision': precision,
            'accuracy': accuracy,
            'dice': dice,
            'F1_score': F1_score,
            'prob1_acc': prob1_acc,
            'class_acc1': class_acc1,
            'class_acc2': class_acc2,
            'class_acc3': class_acc3
        }
        dataFrame = pd.DataFrame(results, index=[0])
        dataFrame.to_csv(csv_name)
        print('saved new results!')
    else:
        dataFrame = pd.read_csv(csv_name, index_col=0)
        new_result = [model_name, model_saved, jaccard,
                      miou, Recall, precision, accuracy, dice, F1_score, prob1_acc, class_acc1, class_acc2, class_acc3]
        dataFrame.loc[len(dataFrame.index)] = new_result
        dataFrame.to_csv(csv_name)
        print('saved add results!')

if __name__ == '__main__':
    models = {
        'Unet': 'unet_multi_task_1_adam_1000',
        'deeplabv3': 'deeplabv3_multi_task_1_adam_1000',
        'atten_unet': 'atten_unet_multi_task_1_adam_1000',
        'FCN': 'FCNs_multi_task_1_adam_1000',
        'segnet': 'segnet_multi_task_1_adam_1000'
    }
    for model in models.keys():
        model_name = model
        model_saved = models[model]
        single_model_predict(model_name, model_saved)