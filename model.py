import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_features, in_features, 3),
                                   nn.InstanceNorm2d(in_features),
                                    nn.ReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_features, in_features, 3),
                                    nn.InstanceNorm2d(in_features))

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels = 3, n_residual_blocks = 3, n_conv_blocks = 3):
        super(Generator, self).__init__()

        out_features = 256 // (2  ** (n_conv_blocks-1)) # 64

        model = [nn.ReflectionPad2d(in_channels),
                 nn.Conv2d(in_channels, out_features, 7),
                 nn.InstanceNorm2d(out_features),
                 nn.ReLU(inplace=True)]

        n_conv_blocks -= 1
        in_features = out_features

        for _ in range(n_conv_blocks):
            out_features = 2 * in_features
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features = in_features // 2
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(out_features, in_channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
  def conv_block_lrelu(self, in_filters, out_filters, normalize=True):
    layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize: layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

  def __init__(self, input_nc = 3, conv_layers = 5):
    super(Discriminator, self).__init__()

    out_filters = 64

    model = [nn.Conv2d(input_nc, out_filters, 4, stride=2, padding=1),
             nn.LeakyReLU(0.2, inplace=True)]

    for i in range(conv_layers - 3):
      in_filters = out_filters
      out_filters = in_filters * 2
      model += [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)]

    model += [  nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True) ]

    model += [nn.Conv2d(512, 1, 4, padding=1)]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    x = self.model(x)
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.flatten(start_dim=1)
    return x

from torch.autograd import Variable

class ReplayBuffer():
  def __init__(self, max_size = 50):
    assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
    self.max_size = max_size
    self.data = []

  def push_and_pop(self, data):
    to_return = []
    for element in data.data:
      element = torch.unsqueeze(element, 0)
      if len(self.data) < self.max_size:
        self.data.append(element)
        to_return.append(element)
      else:
        if random.uniform(0,1) > 0.5:
          i = random.randint(0, self.max_size-1)
          to_return.append(self.data[i].clone())
          self.data[i] = element
        else:
          to_return.append(element)
    return Variable(torch.cat(to_return))
