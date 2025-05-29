import math
from functools import partial
import torch
import torch.nn as nn
import torchvision
import json

class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def write_tensorboard(writer, model, epoch, train_metrics, valid_metrics):
    if (epoch - 1) % 5 == 0:
        train_image, train_mask, train_prob = train_metrics['image'], train_metrics['mask'], train_metrics['prob']
        valid_image, valid_mask, valid_prob = valid_metrics['image'], valid_metrics['mask'], valid_metrics['prob']
        saved_images = torchvision.utils.make_grid(train_image, nrow=train_image.shape[0], padding=10, pad_value=1)
        writer.add_image('train/Image', saved_images, epoch)

        for n in range(train_mask.shape[1]):
            saved_images = torch.cat((train_mask.narrow(1, n, 1), train_prob.narrow(1, n, 1)), 0)
            saved_images = torchvision.utils.make_grid(saved_images, nrow=train_image.shape[0], padding=10, pad_value=1)
            writer.add_image('train/Mask%s' % n, saved_images, epoch)

        saved_images = torchvision.utils.make_grid(valid_image, nrow=valid_image.shape[0], padding=10, pad_value=1)
        writer.add_image('test/Image', saved_images, epoch)

        for n in range(valid_mask.shape[1]):
            saved_images = torch.cat((valid_mask.narrow(1, n, 1), valid_prob.narrow(1, n, 1)), 0)
            saved_images = torchvision.utils.make_grid(saved_images, nrow=valid_image.shape[0], padding=10, pad_value=1)
            writer.add_image('test/Mask%s' % n, saved_images, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    valid_loss, valid_loss1, valid_loss2 = valid_metrics['loss'], valid_metrics['loss1'], valid_metrics['loss2']
    train_loss, train_loss1, train_loss2 = train_metrics['loss'], train_metrics['loss1'], train_metrics['loss2']
    train_jaccard, valid_jaccard = train_metrics['jaccard'], valid_metrics['jaccard']
    train_miou, train_mPA, train_accuracy = train_metrics['miou'], train_metrics['mPA'], train_metrics['accuracy']
    valid_miou, valid_mPA, valid_accuracy = valid_metrics['miou'], valid_metrics['mPA'], valid_metrics['accuracy']
    train_prob1_acc = train_metrics['prob1_acc']
    valid_prob1_acc = valid_metrics['prob1_acc']

    writer.add_scalars('loss', {'train': train_loss, 'test': valid_loss}, epoch)
    writer.add_scalars('loss1', {'train': train_loss1, 'test': valid_loss1}, epoch)
    writer.add_scalars('loss2', {'train': train_loss2, 'test': valid_loss2}, epoch)
    writer.add_scalars('miou', {'train': train_miou, 'test': valid_miou}, epoch)
    writer.add_scalars('mPA', {'train': train_mPA, 'test': valid_mPA}, epoch)
    writer.add_scalars('accuracy', {'train': train_accuracy, 'test': valid_accuracy}, epoch)
    writer.add_scalars('prob1_acc', {'train': train_prob1_acc, 'test': valid_prob1_acc}, epoch)

    for out in range(1, 2):
        for auc in range(1, 4):
            key = 'out{}auc{}'.format(str(out), str(auc))
            writer.add_scalars(key, {'train': train_metrics[key], 'test': valid_metrics[key]}, epoch)

    writer.add_scalars('jaccard', {'train': train_jaccard, 'test': valid_jaccard}, epoch)


def write_event(log, step, epoch, train_metrics, valid_metrics):
    CMD = 'epoch:{} step:{} time:{:.2f} \n train_loss:{:.3f} {:.3f} {:.3f} \ntrain_auc1:{} {} {}\ntrain_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} train_miou: {:.3f} train_mPA: {:.3f} train_acc: {:.3f} train_prob1_acc {:.3f}\n valid_loss:{:.3f} {:.3f} {:.3f}\nvalid_auc1:{} {} {} \n valid_jaccard:{:.3f} {:.3f} {:.3f} {:.3f}valid_miou: {:.3f} valid_mPA: {:.3f} valid_acc: {:.3f} valid_prob1_acc: {:.3f}'.format(
        epoch, step, train_metrics['epoch_time'],
        train_metrics['loss'], train_metrics['loss1'], train_metrics['loss2'],
        train_metrics['out1auc1'], train_metrics['out1auc2'], train_metrics['out1auc3'],
        train_metrics['jaccard'], train_metrics['jaccard1'], train_metrics['jaccard2'], train_metrics['jaccard3'],
        train_metrics['miou'], train_metrics['mPA'], train_metrics['accuracy'],
        train_metrics['prob1_acc'],
        valid_metrics['loss'], valid_metrics['loss1'], valid_metrics['loss2'],
        valid_metrics['out1auc1'], valid_metrics['out1auc2'], valid_metrics['out1auc3'],
        valid_metrics['jaccard'], valid_metrics['jaccard1'], valid_metrics['jaccard2'], valid_metrics['jaccard3'],
        valid_metrics['miou'], valid_metrics['mPA'], valid_metrics['accuracy'],
        valid_metrics['prob1_acc']
    )
    print(CMD)
    log.write(json.dumps(CMD))
    log.write('\n')
    log.flush()

def preprocess_input(image):
    image /= 255.0
    return image