import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.vgg = VGG()
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()

    def forward(self, input):
        input = self.vgg(input)
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        w = self.branch3(input)

        input = w * branch2 + (1 - w) * branch1

        return input


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512])
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        state_dict.pop('classifier.0.weight')
        state_dict.pop('classifier.0.bias')
        state_dict.pop('classifier.3.weight')
        state_dict.pop('classifier.3.bias')
        state_dict.pop('classifier.6.weight')
        state_dict.pop('classifier.6.bias')
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        return x

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.dc = BaseDeconv(512, 512, activation=nn.ReLU(), use_bn=True)
        self.conv1 = BaseConv(512, 512, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(512, 256, 9, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(256, 128, 7, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 64, 7, activation=nn.ReLU(), use_bn=True)
        self.conv5 = BaseConv(64, 1, 3, activation=None, use_bn=False)
        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, input):
        input = self.dc(input)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.conv5(input)
        input = self.mp(input)
        return input


class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv1 = BaseConv(512, 256, 3, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 128, 3, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(128, 64, 3, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(64, 1, 3, activation=None, use_bn=False)

    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        return input


class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.fc1 = BaseLinear(512, 32, activation=nn.ReLU(), use_drop=True)
        self.fc2 = BaseLinear(32, 1, activation=nn.Sigmoid(), use_drop=False)

    def forward(self, input):
        input = torch.mean(input, (2, 3))
        input = self.fc1(input)
        input = self.fc2(input)

        # divide w into 100 bins
        input = input * 100
        input = torch.round(input)
        input = input / 100
        input = input.view(-1, 1, 1, 1)

        return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class BaseDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, use_bn=False):
        super(BaseDeconv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.deconv.weight.data.normal_(0, 0.01)
        self.deconv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.deconv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class BaseLinear(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, use_drop=False):
        super(BaseLinear, self).__init__()
        self.use_drop = use_drop
        self.activation = activation
        self.fc = nn.Linear(in_channels, out_channels)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        self.drop = nn.Dropout()

    def forward(self, input):
        input = self.fc(input)
        if self.activation:
            input = self.activation(input)
        if self.use_drop:
            input = self.drop(input)

        return input


if __name__ == '__main__':
    model = Model().cuda()
    input = torch.randn(16, 3, 400, 400).cuda()
    output = model(input)
    print(input.size())
    print(output.size())
