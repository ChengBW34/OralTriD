import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.cuda.empty_cache()
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import pickle
import argparse
from pathlib import Path
import pandas as pd
import time
from utils.callbacks import LossHistory
from utils.training_utils import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.training_utils import get_lr
from utils.training_utils import show_config
from utils.training_utils import write_event, write_tensorboard
from utils.training_utils import LossBinary
from utils.datasets import make_loader
from utils.metrics import AllInOneMeter
from models.Unet import Unet
from models.deeplabv3 import DeepLabV3
from models.FCN import FCNs
from models.atten_unet import AttU_Net
from models.segnet import SegNet


def get_split(train_test_split_file):
    train_test_id = pd.read_csv(train_test_split_file, index_col=0)
    train_test_id['total'] = train_test_id[['OLP', 'OLK', 'OBU']].sum(axis=1)
    return train_test_id


if __name__ == "__main__":
    dir_path = 'OralTrid'
    total_num = '808'
    dir_name = f'./data_train_val_split'
    scaler = torch.amp.GradScaler()
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name', type=str, default='unet', help='models', choices=['unet', 'deeplabv3', 'FCNs', 'atten_unet', 'segnet'])
    arg('--data-path', type=str, default=f'{dir_name}/glossopathy_h5', help='data path')
    arg('--jaccard-weight', type=float, default=1)
    arg('--checkpoint', type=str, default=f'./checkpoint', help='checkpoint path')
    arg('--save-dir', type=str, default=f'./logs', help='save dir path')
    arg('--train-test-split-file', type=str, default=f'{dir_name}/train_val_split.csv', help='train test split file path')
    arg('--batch-size-train', type=int, default=5)
    arg('--num-classes', type=int, default=3)
    arg('--batch-size-val', type=int, default=5)
    arg('--Epoch', type=int, default=1000)
    arg('--optimizer-type', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])
    arg('--resume-path', type=str, default=None)
    arg('--attribute', type=str, default='all', choices=['OLP', 'OLK', 'OBU', 'all'])
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == 'unet':
        model = Unet(in_channels=3, classes=3)
    elif model_name == 'deeplabv3':
        model = DeepLabV3(in_class=3, class_num=3)
    elif model_name == 'FCNs':
        model = FCNs(pretrained_net='VGGNet', n_class=3)
    elif model_name == 'atten_unet':
        model = AttU_Net(img_ch=3, output_ch=3)
    elif model_name == 'segnet':
        model = SegNet(input_nbr=3, label_nbr=3)

    data_path = args.data_path
    save_dir = args.save_dir
    Epoch = args.Epoch
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, f"{model_name}_loss" + str(time_str))

    input_shape = [640, 448]
    Init_lr = 2e-3
    Min_lr = Init_lr * 0.001
    momentum = 0.9
    weight_decay = 0

    lr_decay_type = 'cos'
    save_period = 5

    device = torch.device('cuda')
    model = model.to(device)
    weights_init(model)

    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    loss_fn = LossBinary(jaccard_weight=args.jaccard_weight)
    train_test_id = get_split(args.train_test_split_file)

    train_loader = make_loader(train_test_id, data_path, args, train=True, shuffle=True,
                               train_test_split_file=args.train_test_split_file)
    valid_loader = make_loader(train_test_id, data_path, args, train=False, shuffle=True,
                               train_test_split_file=args.train_test_split_file)
    print('train_loader: ', len(train_loader))
    print('valid_loader: ', len(valid_loader))

    num_train = (train_test_id['Split'] == 'train').sum()
    num_val = (train_test_id['Split'] == 'val').sum()
    show_config(
        num_classes=args.num_classes, model_name=model_name, input_shape=input_shape,
        Epoch=Epoch, batch_size_train=args.batch_size_train, batch_size_val=args.batch_size_val,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=args.optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_train=num_train, num_val=num_val
    )
    nbs = 16
    lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size_train / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size_val / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    epoch_step = num_train // args.batch_size_train + 1
    epoch_step_val = num_val // args.batch_size_val + 1

    train_sampler = None
    val_sampler = None
    shuffle = True

    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)
    log = checkpoint.joinpath('train.log').open('at', encoding='utf8')
    eval_flag = True
    eval_period = 5
    val_lines = train_test_id[train_test_id['Split'] == 'val'].ID.values

    writer = SummaryWriter(log_dir=checkpoint)
    w1 = 1.0
    w2 = 0.1
    step = 0
    meter_train = AllInOneMeter()
    meter_val = AllInOneMeter()

    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    for epoch in range(Epoch):
        start_time = time.time()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        total_loss = 0
        total_f_score = 0

        val_loss = 0
        val_f_score = 0

        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        model.train()
        meter_train.reset()
        for iteration, batch in enumerate(train_loader):
            imgs, pngs, labels = batch
            imgs = imgs.permute(0, 3, 1, 2)
            pngs = pngs.permute(0, 3, 1, 2)
            with torch.no_grad():
                imgs = imgs.to(device).type(torch.cuda.FloatTensor)
                pngs = pngs.to(device).type(torch.cuda.FloatTensor)
                labels = labels.to(device).type(torch.cuda.FloatTensor)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs, outputs_mask_ind1 = model(imgs)
                train_prob = torch.sigmoid(outputs)
                train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)
                loss1 = loss_fn(outputs, pngs)  # outputs--[48, 3 448, 640]   pngs--[48, 3, 448, 640]
                loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, labels)
                loss = loss1 * w1 + loss2 * w2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            meter_train.add(train_prob, pngs, train_mask_ind_prob1, labels, loss1.item(), loss2.item(), loss.item())
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            step += 1

        epoch_time = time.time() - start_time
        train_metrics = meter_train.value()
        train_metrics['epoch_time'] = epoch_time
        train_metrics['image'] = imgs.data
        train_metrics['mask'] = pngs.data
        train_metrics['prob'] = train_prob.data
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

        model.eval()
        strat_time = time.time()
        meter_val.reset()
        with torch.no_grad():
            for iteration, batch in enumerate(valid_loader):
                # print('valid iteration: ', iteration)
                imgs, pngs, labels = batch
                imgs = imgs.permute(0, 3, 1, 2)
                pngs = pngs.permute(0, 3, 1, 2)

                with torch.no_grad():
                    imgs = imgs.to(device).type(torch.cuda.FloatTensor)
                    pngs = pngs.to(device).type(torch.cuda.FloatTensor)
                    labels = labels.to(device).type(torch.cuda.FloatTensor)

                with torch.amp.autocast(device_type='cuda'):
                    outputs, outputs_mask_ind1, = model(imgs)
                    val_prob = torch.sigmoid(outputs)
                    val_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)
                    loss1 = loss_fn(outputs, pngs)  # outputs--[48, 3 448, 640]   pngs--[48, 3, 448, 640]
                    loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, labels)
                    loss = loss1 * w1 + loss2 * w2

                val_loss += loss.item()

                meter_val.add(val_prob, pngs, val_mask_ind_prob1, labels, loss1.item(), loss2.item(), loss.item())

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        valid_metrics = meter_val.value()
        epoch_time = time.time() - start_time
        valid_metrics['epoch_time'] = epoch_time
        valid_metrics['image'] = imgs.data
        valid_metrics['mask'] = pngs.data
        valid_metrics['prob'] = val_prob.data

        pbar.close()
        print('Finish Validation')
        write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
        write_tensorboard(writer, model, epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
        # eval_callback.on_epoch_end(epoch + 1, model)
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(log_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(log_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(log_dir, "last_epoch_weights.pth"))
        torch.cuda.empty_cache()
    loss_history.writer.close()
