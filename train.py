import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 only
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error tracking

import torch

torch.cuda.empty_cache()  # Clear GPU cache
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import yaml
from pathlib import Path
import pandas as pd
import time

# Import custom utilities
from utils.callbacks import LossHistory
from utils.training_utils import (get_lr_scheduler, set_optimizer_lr, weights_init,
                                  get_lr, show_config, write_event, write_tensorboard)
from utils.training_utils import LossBinary
from utils.datasets import make_loader
from utils.metrics import AllInOneMeter
from models.Unet import Unet
from models.deeplabv3 import DeepLabV3
from models.FCN import FCNs
from models.atten_unet import AttU_Net
from models.segnet import SegNet


def get_split(train_test_split_file):
    """Load and process train-test split file"""
    train_test_id = pd.read_csv(train_test_split_file, index_col=0)
    train_test_id['total'] = train_test_id[['OLP', 'OLK', 'OBU']].sum(axis=1)
    return train_test_id


if __name__ == "__main__":
    # Load configuration file
    with open("./config/train.yml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    optimizer_type = config['training']['optimizer_type']
    batch_size_train = config['training']['batch_size_train']
    batch_size_val = config['training']['batch_size_val']
    model_name = config["model"]["model_name"]
    jaccard_weight = config['model']["jaccard_weight"]
    num_classes = config['model']['num_classes']
    dir_path = config["paths"]["dir_path"]
    data_path = config["paths"]["data_path"].format(dir_path=dir_path)
    train_test_split_file = config["paths"]["train_test_split_file"].format(dir_path=dir_path)
    save_dir = config["paths"]["save_dir"]
    Epoch = config["training"]['epoch']

    # Initialize model based on configuration
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

    # Create log directory with timestamp
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, f"{model_name}_loss" + str(time_str))

    # Training parameters
    input_shape = [640, 448]  # Input image dimensions
    Init_lr = 2e-3  # Initial learning rate
    Min_lr = Init_lr * 0.001  # Minimum learning rate
    momentum = 0.9  # Momentum for optimizer
    weight_decay = 0  # Weight decay

    lr_decay_type = 'cos'  # Learning rate decay type
    save_period = 5  # Model saving interval (epochs)

    # Set up device and model
    device = torch.device('cuda')
    model = model.to(device)
    weights_init(model)  # Initialize model weights

    # Initialize loss history tracker
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # Loss function and data split
    loss_fn = LossBinary(jaccard_weight=jaccard_weight)
    train_test_id = get_split(train_test_split_file)

    # Create data loaders
    train_loader = make_loader(train_test_id, data_path, config, train=True, shuffle=True,
                               train_test_split_file=train_test_split_file)
    valid_loader = make_loader(train_test_id, data_path, config, train=False, shuffle=True,
                               train_test_split_file=train_test_split_file)

    print('Train loader samples: ', len(train_loader))
    print('Validation loader samples: ', len(valid_loader))

    # Count train/val samples
    num_train = (train_test_id['Split'] == 'train').sum()
    num_val = (train_test_id['Split'] == 'val').sum()

    # Display training configuration
    show_config(
        num_classes=num_classes, model_name=model_name, input_shape=input_shape,
        Epoch=Epoch, batch_size_train=batch_size_train, batch_size_val=batch_size_val,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, save_period=save_period,
        save_dir=save_dir, num_train=num_train, num_val=num_val
    )

    # Learning rate adjustment parameters
    nbs = 16  # Nominal batch size
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size_train / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size_val / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # Initialize optimizer
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    # Learning rate scheduler
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    # Calculate steps per epoch
    epoch_step = num_train // batch_size_train + 1
    epoch_step_val = num_val // batch_size_val + 1

    # Initialize data samplers and checkpoints
    train_sampler = None
    val_sampler = None
    shuffle = True

    checkpoint = Path(config['paths']['checkpoint'])
    checkpoint.mkdir(exist_ok=True, parents=True)
    log = checkpoint.joinpath('train.log').open('at', encoding='utf8')

    # Evaluation parameters
    eval_flag = True
    eval_period = 5
    val_lines = train_test_id[train_test_id['Split'] == 'val'].ID.values

    # Initialize TensorBoard writer and training parameters
    writer = SummaryWriter(log_dir=checkpoint)
    w1 = 1.0  # Weight for loss1
    w2 = 0.1  # Weight for loss2
    step = 0  # Global step counter

    # Initialize metrics meters and gradient scaler for mixed precision
    meter_train = AllInOneMeter()
    meter_val = AllInOneMeter()
    scaler = torch.amp.GradScaler()

    # Main training loop
    for epoch in range(Epoch):
        start_time = time.time()

        # Set learning rate for current epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # Initialize epoch metrics
        total_loss = 0
        total_f_score = 0
        val_loss = 0
        val_f_score = 0

        # Training phase
        print('Start Training')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        model.train()
        meter_train.reset()

        for iteration, batch in enumerate(train_loader):
            imgs, pngs, labels = batch

            # Permute dimensions for PyTorch (B,H,W,C) -> (B,C,H,W)
            imgs = imgs.permute(0, 3, 1, 2)
            pngs = pngs.permute(0, 3, 1, 2)

            # Move data to GPU
            with torch.no_grad():
                imgs = imgs.to(device).type(torch.cuda.FloatTensor)
                pngs = pngs.to(device).type(torch.cuda.FloatTensor)
                labels = labels.to(device).type(torch.cuda.FloatTensor)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs, outputs_mask_ind1 = model(imgs)
                train_prob = torch.sigmoid(outputs)
                train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)

                # Calculate losses
                loss1 = loss_fn(outputs, pngs)  # Segmentation loss
                loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, labels)  # Classification loss
                loss = loss1 * w1 + loss2 * w2  # Combined loss

            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Update metrics
            meter_train.add(train_prob, pngs, train_mask_ind_prob1, labels, loss1.item(), loss2.item(), loss.item())
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            step += 1

        # Calculate and store training metrics
        epoch_time = time.time() - start_time
        train_metrics = meter_train.value()
        train_metrics['epoch_time'] = epoch_time
        train_metrics['image'] = imgs.data
        train_metrics['mask'] = pngs.data
        train_metrics['prob'] = train_prob.data
        pbar.close()
        print('Training Complete')

        # Validation phase
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

        model.eval()
        meter_val.reset()

        with torch.no_grad():
            for iteration, batch in enumerate(valid_loader):
                imgs, pngs, labels = batch
                imgs = imgs.permute(0, 3, 1, 2)
                pngs = pngs.permute(0, 3, 1, 2)

                # Move data to GPU
                with torch.no_grad():
                    imgs = imgs.to(device).type(torch.cuda.FloatTensor)
                    pngs = pngs.to(device).type(torch.cuda.FloatTensor)
                    labels = labels.to(device).type(torch.cuda.FloatTensor)

                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    outputs, outputs_mask_ind1 = model(imgs)
                    val_prob = torch.sigmoid(outputs)
                    val_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)

                    # Calculate validation losses
                    loss1 = loss_fn(outputs, pngs)  # Segmentation loss
                    loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, labels)  # Classification loss
                    loss = loss1 * w1 + loss2 * w2  # Combined loss

                val_loss += loss.item()

                # Update validation metrics
                meter_val.add(val_prob, pngs, val_mask_ind_prob1, labels, loss1.item(), loss2.item(), loss.item())
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

        # Calculate and store validation metrics
        valid_metrics = meter_val.value()
        epoch_time = time.time() - start_time
        valid_metrics['epoch_time'] = epoch_time
        valid_metrics['image'] = imgs.data
        valid_metrics['mask'] = pngs.data
        valid_metrics['prob'] = val_prob.data

        pbar.close()
        print('Validation Complete')

        # Log results and save models
        write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
        write_tensorboard(writer, model, epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        # Print epoch summary
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # Save models periodically
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(log_dir,
                                                        'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                                                            (epoch + 1), total_loss / epoch_step,
                                                            val_loss / epoch_step_val)))

        # Save best model
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Saving best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(log_dir, "best_epoch_weights.pth"))

        # Always save last epoch
        torch.save(model.state_dict(), os.path.join(log_dir, "last_epoch_weights.pth"))
        torch.cuda.empty_cache()  # Clear GPU cache

    # Clean up
    loss_history.writer.close()
