import os
import cv2
import numpy as np
import torch
import torch.nn as nn
    
    
class MultiTaskLoss(nn.Module):
    def __init__(self, epsilon, rate=0.5, gamma=2, ignore_idx=None, weight=[.2, .4, .4]):
        super(MultiTaskLoss, self).__init__()
        self.epsilon = epsilon
        self.rate = rate
        self.gamma = gamma
        self.ignore_idx = ignore_idx
        self.weight = weight
    
    def focal_loss(self, y_pred, y):
        y_pred = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
        y = y.view(y_pred.size())
        if self.weight is not None:
            self.weight = torch.tensor(self.weight).cuda()
            weight = self.weight.unsqueeze(0).unsqueeze(-1).repeat(
                y_pred.shape[0], 1, y_pred.shape[2]
            )
        
        ce = -torch.log(y_pred + self.epsilon) * y
        loss = torch.pow((1 - y_pred), self.gamma) * ce
        if self.weight is not None:
            loss = torch.mul(loss, weight)
            del weight
        loss = torch.sum(loss, dim=1)
        # print(loss.shape)
        # print(loss)

        return torch.mean(loss)
    
    def dice_loss(self, y_pred, y, num_classes):
        ################
        # y_pred_ = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
        # # print(y_pred_.shape)
        # y_ = torch.argmax(y, dim=1).detach().cpu().numpy()
        # print(np.max(y_pred_), np.min(y_pred_))
        # print(len(np.where(y_pred_ > 0)[0]))
        # for i in range(y_pred.shape[0]):
        #     cv2.imwrite(f'./imgs/y_pred_{i}.jpg', y_pred_[i] * 255)
        #     cv2.imwrite(f'./imgs/y_{i}.jpg', y_[i] * 255)
        ################
        # print(y_pred.shape)
        # print(y.shape)
        assert torch.all((y >= 0) & (y < num_classes)), print(torch.max(y), torch.min(y))
        batch_size = y_pred.shape[0]
        
        if self.ignore_idx is not None:
            y_pred = y_pred[:, torch.arange(num_classes) != self.ignore_idx, :, :]
            y = y[:, torch.arange(num_classes) != self.ignore_idx, :, :]
        y_pred = y_pred.contiguous().view(batch_size, -1)
        y = y.contiguous().view(batch_size, -1).float()
        intersection = 2 * torch.sum(y_pred * y, dim=1) + self.epsilon
        union = torch.sum(y_pred * y_pred, dim=1) + torch.sum(y * y, dim=1) + self.epsilon
        loss = 1 - intersection / union
        # print(loss)
    
        return torch.mean(loss)
    
    def combined_loss(self, y_pred, y, num_classes):
        return 0.7 * self.focal_loss(y_pred, y) + \
            0.3 * self.dice_loss(y_pred, y, num_classes)
    
    def cross_entropy_loss(self, y_pred, y):
        num_classes = y_pred.shape[-1]
        y_pred = torch.softmax(y_pred, dim=-1)
        y = nn.functional.one_hot(y, num_classes)
        
        return -torch.sum(y * torch.log(y_pred)) / num_classes
    
    def forward(self, y_mask_pred, y_mask, y_label_pred, y_label):
        # a = self.dice_loss(y_mask_pred, y_mask)
        # b = self.cross_entropy_loss(y_label_pred, y_label)
        # print(
        #     self.dice_loss(y_mask_pred, y_mask).item(),
        #     self.cross_entropy_loss(y_label_pred, y_label).item()
        # )
        # return self.rate * self.dice_loss(y_mask_pred, y_mask) + \
        #     (1 - self.rate) * self.cross_entropy_loss(y_label_pred, y_label)
        # return self.cross_entropy_loss(y_label_pred, y_label)
        num_classes = y_mask_pred.shape[1]
        y_mask = nn.functional.one_hot(y_mask.long(), num_classes).permute(0, 3, 1, 2)
        # a = self.combined_loss(y_mask_pred, y_mask, num_classes)
        # # input('done')
        # return a
        return self.rate * self.combined_loss(y_mask_pred, y_mask, num_classes) + \
            (1 - self.rate) * self.cross_entropy_loss(y_label_pred, y_label)


class EarlyStopping:
    def __init__(self, stop_type, patience=20, delta=0.001):
        assert stop_type in ['dice', 'acc', 'recall', 'loss'], NotImplementedError
        self.stop_type = stop_type
        self.patience = patience
        self.delta = delta
        self.best_metric = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_metric, save_model_path, model_name, net):
        if self.best_metric is None:
            self.best_metric = val_metric
        elif (self.stop_type in ['dice', 'acc', 'recall'] and val_metric < self.best_metric - self.delta) or \
            (self.stop_type in ['loss'] and val_metric > self.best_metric - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.__save_checkpoint(save_model_path, model_name, net)
                self.early_stop = True
        else:
            self.best_metric = val_metric
            self.counter = 0
    
    def __save_checkpoint(self, save_model_path, model_name, net):
        path = os.path.join(save_model_path, model_name)
        torch.save(net.state_dict(), path)
        

class MetricsCalculation:
    def __init__(
        self, 
        mask,
        ground_truth,
        num_classes
    ):
        if len(mask.shape) > 2:
            self.mask = torch.argmax(mask, dim=1)
        else:
            self.mask = torch.argmax(mask, dim=-1)
        self.mask = self.mask.detach().contiguous().cpu().numpy()
        self.ground_truth = ground_truth.detach().contiguous().cpu().numpy()
        # print(self.mask.shape, self.ground_truth.shape)
        # print(self.mask, self.ground_truth)
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.size = mask.shape[0]
        self.__get_confusion_matrix()
    
    def __get_confusion_matrix(self):
        if len(self.mask.shape) > 2:
            for idx in range(self.size):
                for i in range(self.num_classes):
                    for j in range(self.num_classes):
                        self.confusion_matrix[i, j] = len(np.where(
                            (self.mask[idx] == i) & (self.ground_truth[idx] == j)
                        )[0])
        else:
            self.confusion_matrix[0, 0] = len(np.where(
                (self.mask == 1) & (self.ground_truth == 1)
            )[0])
            self.confusion_matrix[0, 1] = len(np.where(
                (self.mask == 0) & (self.ground_truth == 1)
            )[0])
            self.confusion_matrix[1, 0] = len(np.where(
                (self.mask == 1) & (self.ground_truth == 0)
            )[0])
            self.confusion_matrix[1, 1] = len(np.where(
                (self.mask == 0) & (self.ground_truth == 0)
            )[0])
    
    def get_IoU(self):
        IoU = 0
        for i in range(self.num_classes):
            if self.confusion_matrix[i, i] == 0:
                continue
            IoU += self.confusion_matrix[i, i] / (
                np.sum(self.confusion_matrix[i, :]) + np.sum(self.confusion_matrix[:, i]) - self.confusion_matrix[i, i]
            )
        return IoU / self.num_classes
    
    def get_Dice(self):
        Dice = 0
        for i in range(self.num_classes):
            if self.confusion_matrix[i, i] == 0:
                continue
            Dice += 2 * self.confusion_matrix[i, i] / (
                np.sum(self.confusion_matrix[i, :]) + np.sum(self.confusion_matrix[:, i])
            )
        return Dice / self.num_classes
    
    def get_Recall(self):
        Recall = 0
        for i in range(self.num_classes):
            if self.confusion_matrix[i, i] == 0:
                continue
            Recall += self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i, :])
        return Recall / self.num_classes
    
    def get_Precision(self):
        Precision = 0
        for i in range(self.num_classes):
            if self.confusion_matrix[i, i] == 0:
                continue
            Precision += self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[:, i])
        return Precision / self.num_classes
    
    def get_F1_score(self, Recall, Precision):
        if Recall + Precision == 0:
            return 0
        return 2 * (Recall * Precision) / (Recall + Precision)

    def __call__(self):
        IoU = self.get_IoU()
        Dice = self.get_Dice()
        Recall = self.get_Recall()
        Precision = self.get_Precision()
        F1_score = self.get_F1_score(Recall, Precision)

        return round(IoU, 4), round(Dice, 4), round(Recall, 4), round(Precision, 4), round(F1_score, 4)


if __name__ == '__main__':
    a = torch.randn((6, 2))
    b = torch.argmax(torch.softmax(torch.randn((6, 2)), dim=-1), dim=-1)

    cal = MetricsCalculation(a, b, 2)
    print(cal.confusion_matrix)
    print(cal())
