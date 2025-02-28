import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        A basic residual block with two 3x3 convolutions.
        If stride != 1 or in_planes != planes, a projection (1x1 convolution)
        is applied to the input for the skip connection.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2, 2, 2, 2], output_dim=512):
        """
        A ResNet that takes 64x64x3 images as input and outputs a flattened
        vector of dimension `output_dim`.

        Args:
            block: The block class to use (e.g. BasicBlock).
            num_blocks: A list of the number of blocks in each layer.
            output_dim: The dimension of the final flattened vector.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution: output size remains 64x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Build layers. Note: using stride 2 to downsample.
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)  # 64x64
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 32x32
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 16x16
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 8x8
        
        # Global average pooling to reduce spatial dimensions to 1x1.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to produce the final flattened output.
        self.fc = nn.Linear(512 * block.expansion, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a sequential layer consisting of multiple residual blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x: [batch, 3, 64, 64]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)           # shape: [batch, 512, 1, 1]
        out = out.view(out.size(0), -1)     # flatten to [batch, 512]
        out = self.fc(out)                # final flattened output vector
        return out

class WeightFeatureExtractorNet(nn.Module):

    def __init__(self, weight_vec_size):
        super(WeightFeatureExtractorNet, self).__init__()

        self.weight_vec_size = weight_vec_size

        self.image_extractor = ResNet()
        self.weight_extractor = nn.Sequential(
            nn.Linear(weight_vec_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128)
        )
    
    def forward(self, image, weight):

        image = image.permute([0, 3, 1, 2])
        image_feat = self.image_extractor(image)
        weight_feat = self.weight_extractor(weight)
        return torch.hstack([image_feat, weight_feat])