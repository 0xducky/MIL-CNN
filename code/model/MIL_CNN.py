import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init


class h_sigmoid(torch.nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(torch.nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class CoordAtt(torch.nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = torch.nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = torch.nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(mip)
        self.conv2 = torch.nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 =torch.nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.tanh(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.tanh(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
    
class MyModel(torch.nn.Module):
    
    def __init__(self,poolsize):
        super(MyModel,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,kernel_size = 3,padding=1)
        self.separable_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels = 32,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32,out_channels = 64,kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Tanh(),
            CoordAtt(64,64)
        )
        self.separable_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Tanh(),
            CoordAtt(64,64)
        )
        self.separable_conv3 = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Tanh(),
            CoordAtt(64,64)
        )
        self.separable_conv4 = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64,out_channels = 64,kernel_size=1),
            torch.nn.BatchNorm2d(64),
        )
        self.ca_1 = ChannelAttention(64)
        self.ca_2 = ChannelAttention(64)
        self.ca_3 = ChannelAttention(64)
        self.bn32 = torch.nn.BatchNorm2d(32)
        self.bn64_1 = torch.nn.BatchNorm2d(64)
        self.bn64_2 = torch.nn.BatchNorm2d(64)
        self.bn64_3 = torch.nn.BatchNorm2d(64)
        self.linear4 = torch.nn.Linear(1344,11)

        
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.maxPooling1 = torch.nn.MaxPool2d(2)
        self.maxPooling2 = torch.nn.MaxPool2d(4)
        self.softmax = torch.nn.Softmax()
        self.poolsize = poolsize
        self.dorpout = torch.nn.Dropout()



        
    def spp_layer(self,batch_data):
        res = []
        #print(batch_data.size())
        batch_size = batch_data.size()[0]
        for pool_size in self.poolsize:
            avgpool = torch.nn.AdaptiveAvgPool2d(output_size=pool_size)
            res.append(avgpool(batch_data).view(batch_size,-1))
        return torch.cat(res, dim=1).view(batch_size,-1)

    def forward(self,batch_data):
        
        batch_size = batch_data.size()[0]
        batch_data = self.tanh(self.conv1(batch_data))
        batch_data = self.maxPooling1(batch_data)
        batch_data = self.bn32(batch_data)
        
        batch_data = self.separable_conv1(batch_data)
        batch_data = self.maxPooling2(batch_data)
        
        
        batch_data = self.tanh(self.separable_conv2(batch_data))
        channel_weight = self.ca_2(batch_data)
        batch_data = channel_weight * batch_data
        batch_data = self.bn64_2(batch_data)
        
        batch_data = self.tanh(self.separable_conv3(batch_data))
        channel_weight = self.ca_3(batch_data)
        batch_data = channel_weight * batch_data
        batch_data = self.bn64_3(batch_data)
        
        batch_data = self.separable_conv4(batch_data)
        batch_data = self.spp_layer(batch_data)

        batch_data = self.softmax(self.linear4(batch_data))

        return batch_data