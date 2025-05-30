import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from utils.training_utils import preprocess_input
import pickle
import h5py
from torchvision.utils import save_image, make_grid


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(torch.device('cuda'))
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, val_lines, train_test_split_file, num_classes, dataset_path, log_dir, device, \
                 miou_out_path="../temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.num_classes = num_classes
        self.image_path = dataset_path
        self.log_dir = log_dir
        self.device = device
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.input_shape = [640, 448]
        self.num_classes = num_classes
        with open(train_test_split_file, 'rb') as f:
            self.mask_ind = pickle.load(f)
        self.image_ids = val_lines
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image, image_id, mask):
        image_data = image
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        images = torch.from_numpy(image_data)
        images = images.to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        pr, pr_ind1, pr_ind2 = self.net(images)
        pr = pr[0]
        val_prob = torch.sigmoid(pr)
        val_prob_p = val_prob.clone()

        val_prob[val_prob > 0.5] = 1
        val_prob[val_prob <= 0.5] = 0
        print(val_prob)
        cat_mask = torch.cat([mask.permute(2, 0, 1).unsqueeze(1), val_prob_p.unsqueeze(1), val_prob.unsqueeze(1)],
                             dim=0)
        saved_masks = make_grid(cat_mask, nrow=3, padding=10, pad_value=1)
        save_image(saved_masks, os.path.join('../miou_fig', image_id + '.png'))
        return val_prob

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")

def load_mask(image_path, img_id):
    mask_file = image_path + "/%s_all.h5" % (img_id)
    f = h5py.File(mask_file, 'r')
    mask_np = f['image'][()]
    mask_np = (mask_np / 255).astype('uint8')
    return mask_np

def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['image'][()]
    img_np = (img_np / 255.0).astype('float32')
    return img_np