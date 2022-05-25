import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from easydict import EasyDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from audio.dataloader.audio_dataset import AudioDataset
# from model import resnet34
# from audio.utils.utility import print_arguments
# from audio.utils import file_utils
# from audio.models import resnet
class ResNet(nn.Module):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            resnet50 = ResNet(BottleneckBlock, 50)

            resnet18 = ResNet(BasicBlock, 18)

    """

    def __init__(self, block, depth, num_classes=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, padding=1, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Train():
    """Training  Pipeline"""
    def __init__(self):
        self.device = "cuda:0"
        self.num_epoch = 1
        self.net_type = "mbv2"
        self.work_dir='work_space/'
        self.work_dir = 'work_space/resnet'
        self.model_dir = 'work_space/resnet/model'
        self.data_dir = "data/UrbanSound8K"
        self.train_data = 'data/UrbanSound8K/train.txt'
        self.test_data = 'data/UrbanSound8K/test.txt'
        self.class_name = 'data/UrbanSound8K/class_name.txt'
        self.input_shape='(None, 1, 128, 128)'
        self.learning_rate=1e-3
        # file_utils.create_dir(self.model_dir)

        self.train_loader, self.test_loader = self.build_dataset()
        # 获取模型
        self.model = self.build_model()
        # 获取优化方法
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=5e-4)
        # 获取学习率衰减函数
        self.scheduler = MultiStepLR(self.optimizer, milestones=[50, 80], gamma=0.1)
        # 获取损失函数
        self.losses = torch.nn.CrossEntropyLoss()

    def build_dataset(self):
        """构建训练数据和测试数据"""
        input_shape = eval(self.input_shape)
        # 加载训练数据
        train_dataset = AudioDataset(self.train_data,
                                     self.class_name,
                                     self.data_dir,
                                     mode='train',
                                     spec_len=input_shape[3])
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True,
                                  num_workers=8)
        self.class_name = train_dataset.class_name
        self.class_dict = train_dataset.class_dict
        self.num_classes = len(self.class_name)
        # 加载测试数据
        test_dataset = AudioDataset(self.test_data,
                                    class_name=self.class_name,
                                    data_dir=self.data_dir,
                                    mode='test',
                                    spec_len=input_shape[3])
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False,
                                 num_workers=8)
        print("train nums:{}".format(len(train_dataset)))
        print("test  nums:{}".format(len(test_dataset)))
        return train_loader, test_loader

    def build_model(self):
        """构建模型"""
        # model = resnet.resnet34(num_classes=10)
        model=ResNet(BasicBlock, 34, num_classes=10)
        # model=torchvision.models.resnet34(num_classes=10,pretrained=True)
        model.to(self.device)
        return model

    def epoch_test(self, epoch):
        """模型测试"""
        loss_sum = []
        accuracies = []
        self.model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(self.test_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                output = self.model(inputs)
                # 计算损失值
                loss = self.losses(output, labels)
                # 计算准确率
                output = torch.nn.functional.softmax(output, dim=1)
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                labels = labels.data.cpu().numpy()
                acc = np.mean((output == labels).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss)
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Test epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss

    def epoch_train(self, epoch):
        """模型训练"""
        loss_sum = []
        accuracies = []
        self.model.train()
        for step, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()
            output = self.model(inputs)
            # 计算损失值
            loss = self.losses(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            labels = labels.data.cpu().numpy()
            acc = np.mean((output == labels).astype(int))
            accuracies.append(acc)
            loss_sum.append(loss)
            # if step % 50 == 0:
            #     lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            #     print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f，lr:%f' % (
            #         datetime.now(), epoch, step, len(self.train_loader), sum(loss_sum) / len(loss_sum),
            #         sum(accuracies) / len(accuracies), lr))
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Train epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss
    def save_model(self, epoch, acc):
        """保存模型"""
        model_path = os.path.join(self.model_dir, 'model_{:0=3d}_{:.3f}.pth'.format(epoch, acc))
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.jit.save(torch.jit.script(self.model), model_path)

if __name__ == '__main__':
    t = Train()
    for epoch in range(t.num_epoch):
        train_acc, train_loss = t.epoch_train(epoch)
        test_acc, test_loss = t.epoch_test(epoch)
        t.scheduler.step()
    t.save_model(epoch, test_acc)
