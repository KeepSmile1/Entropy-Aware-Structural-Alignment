import torch
import torch.nn as nn

"""
注意力层（可选）
"""
class DSAL(nn.Module):
    def __init__(self, channels):
        super(DSAL, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        foreground = self.maxpool(x)
        foreground = self.conv(foreground)
        foreground = self.sigmoid(foreground)
        foreground = foreground * x 
        
        background = self.avgpool(foreground)
        background = self.conv(background)
        background = self.sigmoid(background)
        background = background * x
        
        return background

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True), 
            nn.Linear(channels // reduction, channels, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avgpool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)
    

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        # self.dsal = DSAL(planes)  # DSAL 注意力模块
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.dsal(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)


        return out


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 512, layers[1])
        self.layer2_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(512)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 512, 1024, layers[2])
        self.layer3_conv = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(1024)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        return x


class GateFusion(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        """
        :param input_size: 每个特征的维度，例如 512
        :param output_size: 融合后输出的维度，例如 1024
        :param num_features: 特征数量，例如 4
        """
        super(GateFusion, self).__init__()
        self.gate_fc = nn.Linear(num_features * input_size, num_features)
        self.proj_fc = nn.Linear(input_size, output_size)

    def forward(self, features):
        """
        :param features: 列表，每个元素形状为 [batch, input_size] 或单个样本时 [input_size]
        :return: 融合后的特征，形状为 [batch, output_size] 或 [output_size]
        """
        concatenated = torch.cat(features, dim=-1)  # shape: [batch, num_features * input_size]
        gates = torch.sigmoid(self.gate_fc(concatenated))  # shape: [batch, num_features]
        # 对每个特征按对应门控权重融合
        fused = sum(g.unsqueeze(-1) * feat for g, feat in zip(gates.unbind(dim=-1), features))
        output = self.proj_fc(fused)
        return output



"""
example
"""
class ResNetWithGateFusion(nn.Module):
    def __init__(self, gate_fusion_input_size=512, gate_fusion_output_size=1024, num_features=4):
        """
        :param gate_fusion_input_size: GateFusion 输入特征的维度，例如 512
        :param gate_fusion_output_size: GateFusion 输出特征的维度，例如 1024
        :param num_features: GateFusion 特征数量，例如 4
        """
        super(ResNetWithGateFusion, self).__init__()
        self.resnet = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # GateFusion
        self.gate_fusion = GateFusion(
            input_size=gate_fusion_input_size,
            output_size=gate_fusion_output_size,
            num_features=num_features
        ).cuda()

    def forward(self, image, f_code, f_parent, f_child, f_depth):
        """
        :param image: 输入图像，形状为 [batch_size, 3, H, W]
        :param f_code: 3755 个字符的 f_code 特征，形状为 [3755, 512]
        :param f_parent: 3755 个字符的 f_parent 特征，形状为 [3755, 512]
        :param f_child: 3755 个字符的 f_child 特征，形状为 [3755, 512]
        :param f_depth: 3755 个字符的 f_depth 特征，形状为 [3755, 512]
        :return: 图像特征与融合后的 3755 个字符特征之间的余弦相似度，形状为 [batch_size, 3755]
        """
        image_features = self.resnet(image)
        image_features = self.global_avg_pool(image_features).squeeze(-1).squeeze(-1) 
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # fused_char_features = torch.cat([f_code, f_depth, f_parent, f_child], dim=-1)
        fused_char_features = self.gate_fusion([f_code, f_parent, f_child, f_depth]) 
        similarity_matrix = image_features @ fused_char_features.t() 

        return similarity_matrix