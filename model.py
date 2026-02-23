import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# --- 1. 注意力模块 ---
class CAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pool_out = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(pool_out))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.cam = CAM(channels, reduction)
        self.sam = SAM(kernel_size)

    def forward(self, x):
        return self.sam(self.cam(x))


# --- 2. 残差模块 ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# --- 3. Stem 模块 ---
class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        return self.conv_out(out)


# --- 4. SPPM 模块 ---
class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_sizes = [1, 2, 4]
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for size in self.pool_sizes
        ])
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        out = self.shortcut(x)
        for stage in self.stages:
            pooled = stage(x)
            out = out + F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
        return out


# --- 5. FFM 模块 ---
class FFM(nn.Module):
    def __init__(self, high_res_channels, low_res_channels, out_channels):
        super().__init__()
        self.rb = ResidualBlock(high_res_channels + low_res_channels, out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, high_res_feat, low_res_feat):
        h, w = high_res_feat.shape[2:]
        low_res_feat_up = F.interpolate(low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
        out = torch.cat([high_res_feat, low_res_feat_up], dim=1)
        out = self.rb(out)
        return self.cbam(out)


# --- 6. 细节分支 ---
class DetailBranch(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_after_concat = nn.Sequential(
            nn.Conv2d(in_channels + out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        return self.conv_after_concat(out)


# --- 7. BiCrack 主模型 ---
class BiCrack(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        resnet = models.resnet18(weights=None)

        self.stem = StemBlock(in_channels=3, out_channels=64)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.sppm = SPPM(256, 128)
        self.ffm1 = FFM(high_res_channels=128, low_res_channels=128, out_channels=128)
        self.ffm2 = FFM(high_res_channels=64, low_res_channels=128, out_channels=64)

        self.detail_branch = DetailBranch(in_channels=3, out_channels=64)
        self.final_rb = ResidualBlock(64 + 64, 64)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        stem_out = self.stem(x)
        feat4 = self.layer1(stem_out)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)

        sppm_out = self.sppm(feat16)
        fuse8 = self.ffm1(feat8, sppm_out)
        fuse4 = self.ffm2(feat4, fuse8)

        detail_out = self.detail_branch(x)
        fuse4_up = F.interpolate(fuse4, size=detail_out.shape[2:], mode='bilinear', align_corners=False)
        concat_feat = torch.cat([detail_out, fuse4_up], dim=1)

        refined_feat = self.final_rb(concat_feat)
        out = self.upsample(refined_feat)
        out = self.classifier(out)
        return out