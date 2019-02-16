from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # return F.log_softmax(out, dim=1)
        return out


class LeNet(nn.Module):
    def __init__(self,):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self, data_set='MNIST'):
        super(Net, self).__init__()
        self.data_set = data_set

        if 'MNIST' == data_set:
            self.classifier = LeNet()
            self.classifier_model_name = 'lenet_mnist_model.pth'
            self.localization_in_channels = 1
            self.after_localization_size = 3
        else:
            self.classifier = self.ResNet18()
            self.classifier_model_name = 'ResNet18.pth'
            self.localization_in_channels = 3
            self.after_localization_size = 4

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.localization_in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * self.after_localization_size * self.after_localization_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.last_classifier_layer = 7

    def ResNet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.after_localization_size * self.after_localization_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        grid = torch.clamp(grid, min=-1, max=1)
        x = F.grid_sample(x, grid)

        return x, theta

    # LeNet classification network forward function
    # def classifier(self, x):
    #         x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #         x = x.view(-1, 320)
    #         x = F.relu(self.fc1(x))
    #         x = F.dropout(x, training=self.training)
    #         x = self.fc2(x)
    #         return F.log_softmax(x, dim=1)

    # Full forward function
    def forward(self, x):
        # transform the input
        x, theta = self.stn(x)

        # Perform classification
        log_softmax = self.classifier(x)

        return x, log_softmax, theta

    def load_pretrained_classifier(self):
        import os

        curr_path = os.path.dirname(os.path.realpath(__file__))
        print(curr_path)

        # Load the pre-trained model
        pretrained_model = os.path.join(curr_path, self.classifier_model_name)
        pretrain_parameters = torch.load(pretrained_model, map_location='cpu')

        # layer = 0
        own_state = self.classifier.state_dict()
        # print(own_state)
        for param, key in zip(self.classifier.parameters(), pretrain_parameters.keys()):
            param.requires_grad = False
            param = pretrain_parameters[key].data
            if 'module.' in key:
                key = key[7:]
            own_state[key].copy_(param)

            # if self.last_classifier_layer == layer:
            # if self.last_classifier_layer == layer:
            #     break
            # layer += 1

        own_state = self.classifier.state_dict()
        # print(own_state)


