import torch.nn as nn

# Generator class 
class Generator(nn.Module):
    def __init__(self, input_size, hsize1, hsize2, outsize):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(input_size, hsize1)
        self.Relu = nn.ReLU()
        self.l2 = nn.Linear(hsize1, hsize2)
        self.l3 = nn.Linear(hsize2, outsize)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.Relu(out)
        out = self.l2(out)
        out = self.Relu(out)
        out = self.l3(out)
        out = self.tanh(out)
        return out

#Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_size, h1size, h2size):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(input_size, h1size)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.l2 = nn.Linear(h1size, h2size)
        self.l3 = nn.Linear(h2size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.Lrelu(out)
        out = self.l2(out)
        out = self.Lrelu(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        return out

