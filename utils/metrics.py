from torchnet.meter import AUCMeter
import torch
import numpy as np

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def Accuracy(hist):
    return np.diag(hist).sum() / hist.sum()

class AllInOneMeter(object):
    """
    All in one meter: AUC
    """

    def __init__(self):
        self.out1auc1 = AUCMeter()
        self.out1auc2 = AUCMeter()
        self.out1auc3 = AUCMeter()
        self.loss1 = []
        self.loss2 = []
        self.loss = []
        self.jaccard = []
        self.epsilon = 1e-15
        self.intersection = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.reset()

    def reset(self):
        self.out1auc1.reset()
        self.out1auc2.reset()
        self.out1auc3.reset()
        self.loss1 = []
        self.loss2 = []
        self.loss = []
        self.jaccard = []
        self.intersection = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.hist = np.zeros((2, 2))
        self.classify_prob1 = 0
        self.all = 0

    def add(self, mask_prob, true_mask, mask_ind_prob1, true_mask_ind, loss1, loss2, loss):
        self.out1auc1.add(mask_ind_prob1[:, 0].data, true_mask_ind[:, 0].data)
        self.out1auc2.add(mask_ind_prob1[:, 1].data, true_mask_ind[:, 1].data)
        self.out1auc3.add(mask_ind_prob1[:, 2].data, true_mask_ind[:, 2].data)
        self.loss1.append(loss1)
        self.loss2.append(loss2)
        self.loss.append(loss)
        y_pred = (mask_prob > 0.3).type(true_mask.dtype)
        y_true = true_mask
        self.intersection += (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.union += y_true.sum(dim=-2).sum(dim=-1).sum(dim=0) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.hist += self.compute_miou(mask_prob, true_mask, true_mask_ind)
        self.all += true_mask_ind.shape[0]
        true_class = torch.argmax(true_mask_ind, dim=1)
        prob1_class = torch.argmax(mask_ind_prob1, dim=1)
        self.classify_prob1 += (true_class == prob1_class).sum().item()

    def value(self):
        jaccard_array = (self.intersection / (self.union - self.intersection + self.epsilon))
        jaccard = jaccard_array.mean()
        IoUs = np.mean(per_class_iu(self.hist))
        PA_Recall = np.mean(per_class_PA_Recall(self.hist))
        Precision = np.mean(per_class_Precision(self.hist))
        prob1_acc = round((self.classify_prob1 / self.all), 3) * 100
        metrics = {'out1auc1': self.out1auc1.value()[0], 'out1auc2': self.out1auc2.value()[0],
                   'out1auc3': self.out1auc3.value()[0],
                   'loss1': np.mean(self.loss1), 'loss2': np.mean(self.loss2), 'loss': np.mean(self.loss),
                   'jaccard': jaccard.item(), 'jaccard1': jaccard_array[0].item(), 'jaccard2': jaccard_array[1].item(),
                   'jaccard3': jaccard_array[2].item(), 'miou': IoUs, 'mPA': PA_Recall, 'accuracy': Precision,
                   'prob1_acc': prob1_acc
                   }
        for mkey in metrics:
            metrics[mkey] = round(metrics[mkey], 4)
        return metrics

    def compute_miou(self, mask_prob, true_mask, true_mask_ind):
        num_classes = 2
        hist = np.zeros((num_classes, num_classes))
        for ind in range(true_mask_ind.shape[0]):
            index = torch.argmax(true_mask_ind[ind]).cpu().detach().numpy()
            pred = mask_prob[ind][index].cpu().detach().numpy()
            label = true_mask[ind][index].cpu().detach().numpy()
            label = label.astype(np.uint8)
            pred[pred > 0.3] = 1
            pred[pred <= 0.3] = 0
            pred = pred.astype(np.uint8)
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred} = {:d}'.format(len(label.flatten()), len(pred.flatten())))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        return hist

class TestMetrics(object):
    def __init__(self):
        #super(AllInOneMeter, self).__init__()
        self.out1auc1 = AUCMeter()
        self.out1auc2 = AUCMeter()
        self.out1auc3 = AUCMeter()
        self.jaccard = []
        #self.nbatch = 0
        self.epsilon = 1e-15
        self.intersection = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.reset()

    def reset(self):
        self.out1auc1.reset()
        self.out1auc2.reset()
        self.out1auc3.reset()
        self.jaccard = []
        self.dice = []
        self.intersection = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([3], dtype=torch.float, device='cuda:0')
        self.miou = []
        self.mPA = []
        self.precision = []
        self.recall = []
        self.hist = np.zeros((2, 2))
        self.all = 0
        self.classify_prob1 = 0
        self.classify_prob2 = 0
        self.class_all = np.zeros(3)
        self.class_correct = np.zeros(3)


    def add(self, mask_prob, true_mask, mask_ind_prob1, true_mask_ind):
        self.out1auc1.add(mask_ind_prob1.data, true_mask_ind.data)
        self.out1auc2.add(mask_ind_prob1.data, true_mask_ind.data)
        self.out1auc3.add(mask_ind_prob1.data, true_mask_ind.data)

        y_pred = mask_prob
        y_true = true_mask
        self.intersection += (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.union += y_true.sum(dim=-2).sum(dim=-1).sum(dim=0) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.hist += self.compute_miou(mask_prob, true_mask, true_mask_ind)
        self.all += true_mask_ind.shape[0]
        true_class = torch.argmax(true_mask_ind, dim=1)
        prob1_class = torch.argmax(mask_ind_prob1, dim=0)
        self.classify_prob1 += (true_class==prob1_class).sum().item()
        self.class_all[true_class.cpu()-1] += 1
        if prob1_class==true_class:
            self.class_correct[true_class.cpu()-1] += 1


    def value(self):
        jaccard_array = (self.intersection / (self.union - self.intersection + self.epsilon))
        jaccard = jaccard_array.mean()
        dice_array = (2*self.intersection / (self.union + self.epsilon))
        dice = dice_array.mean()
        IoUs = np.mean(per_class_iu(self.hist))
        PA_Recall = np.mean(per_class_PA_Recall(self.hist))
        Precision = np.mean(per_class_Precision(self.hist))
        Acc = Accuracy(self.hist)
        prob1_acc = round((self.classify_prob1/self.all), 3)
        F1_score = 2*Precision*PA_Recall/(Precision + PA_Recall)
        class_acc = [self.class_correct[i]/self.class_all[i] for i in range(3)]
        metrics = {'out1auc1':self.out1auc1.value()[0], 'out1auc2':self.out1auc2.value()[0],
                   'out1auc3':self.out1auc3.value()[0],
                   'jaccard':jaccard.item(), 'jaccard1':jaccard_array[0].item(),'jaccard2':jaccard_array[1].item(),
                   'jaccard3':jaccard_array[2].item(), 'miou': IoUs, 'Recall': PA_Recall, 'precision': Precision, 'accuracy': Acc,
                   'prob1_acc': prob1_acc, 'dice': dice.item(), 'F1-score': F1_score
                   }
        for mkey in metrics:
            metrics[mkey] = round(metrics[mkey], 4)
        metrics['class_acc'] = class_acc
        return metrics
    def compute_miou(self, mask_prob, true_mask, true_mask_ind):
        num_classes = 2
        hist = np.zeros((num_classes, num_classes))
        for ind in range(true_mask_ind.shape[0]):
            index = torch.argmax(true_mask_ind[ind]).cpu().detach().numpy()
            pred = mask_prob[ind][index].cpu().detach().numpy()
            label = true_mask[ind][index].cpu().detach().numpy()
            label = label.astype(np.uint8)
            pred[pred>0.1] = 1
            pred[pred<=0.1] = 0
            pred = pred.astype(np.uint8)
            if len(label.flatten())!=len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred} = {:d}'.format(len(label.flatten()), len(pred.flatten())))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        return hist