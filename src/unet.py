import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = self.Conv_block(6, 16)
        self.c2 = self.Conv_block(16, 32)
        self.c3 = self.Conv_block(32, 64)
        self.c4 = self.Conv_block(64, 128)
        self.c5 = self.Conv_block(128, 256)
        #self.c6 = self.Conv_block(256, 512)
        self.p = self.Max_pool()

        self.b1 = self.Conv_block(256, 512)

        #self.u1 = self.Upscale(512, 512)
        #self.d1 = self.Conv_block(1024, 512)
        self.u1 = self.Upscale(512, 256)
        self.d1 = self.Conv_block(512, 256)
        self.u2 = self.Upscale(256, 128)
        self.d2 = self.Conv_block(256, 128)
        self.u3 = self.Upscale(128, 64)
        self.d3 = self.Conv_block(128, 64)
        self.u4 = self.Upscale(64, 32)
        self.d4 = self.Conv_block(64, 32)
        self.u5 = self.Upscale(32, 16)
        self.d5 = self.Conv_block(32, 16)

        self.op = nn.Conv2d(16, 1,  kernel_size = 1)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)

    def Conv_block(self, in_channels, out_filters):
        return nn.Sequential(
        nn.Conv2d(in_channels, out_filters, kernel_size = 3, padding=1),
        nn.BatchNorm2d(out_filters),
        nn.PReLU(out_filters, init = 0.1),
        nn.Conv2d(out_filters, out_filters, kernel_size = 3, padding=1),
        nn.BatchNorm2d(out_filters),
        nn.PReLU(out_filters, init = 0.1),
        nn.Dropout(0.2))


    def Max_pool(self):
        return nn.Sequential(
        nn.MaxPool2d(kernel_size = 2),
        nn.Dropout(0.2))

    def Upscale(self, in_channels, out_filters):
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_filters, kernel_size = 2, stride = 2),
        nn.BatchNorm2d(out_filters),
        nn.PReLU(out_filters, init = 0.1),
        nn.Dropout(0.2))

    def forward(self, input):
        x1 = self.c1(input)
        y1 = self.p(x1)
        #print('Conv1')
        #print(x1.shape)
        #print(y1.shape)

        x2 = self.c2(y1)
        y2 = self.p(x2)
        #print('Conv2')
        #print(x2.shape)
        #print(y2.shape)

        x3 = self.c3(y2)
        y3 = self.p(x3)
        #print('Conv3')
        #print(x3.shape)
        #print(y3.shape)

        x4 = self.c4(y3)
        y4 = self.p(x4)
        #print('Conv4')
        #print(x4.shape)
        #print(y4.shape)

        x5 = self.c5(y4)
        y5 = self.p(x5)
        #print('Conv5')
        #print(x5.shape)
        #print(y5.shape)

        x6 = self.b1(y5) #Bridge
        #print('Bridge')
        #print(x6.shape)

        y6 = self.u1(x6)
        y6 = torch.cat((x5, y6), dim = 1)
        x7 = self.d1(y6)
        #print('UpConv5')
        #print(y8.shape)
        #print(x9.shape)

        y9 = self.u2(x7)
        y9 = torch.cat((x4, y9), dim = 1)
        x10 = self.d2(y9)
        #print('UpConv4')
        #print(y9.shape)
        #print(x10.shape)

        y10 = self.u3(x10)
        y10 = torch.cat((x3, y10), dim = 1)
        x11 = self.d3(y10)
        #print('UpConv3')
        #print(y10.shape)
        #print(x11.shape)

        y11 = self.u4(x11)
        y11 = torch.cat((x2, y11), dim = 1)
        x12 = self.d4(y11)
        #print('UpConv2')
        #print(y11.shape)
        #print(x12.shape)

        y12 = self.u5(x12)
        y12 = torch.cat((x1, y12), dim = 1)
        x13 = self.d5(y12)
        #print('UpConv1')
        #print(y12.shape)
        #print(x13.shape)
        #print('Final: ',self.op(x13).shape)

        return self.op(x13)