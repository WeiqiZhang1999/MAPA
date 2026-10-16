from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallConvNet3D(nn.Module):
    """Compact 3D ConvNet for MedMNIST3D baselines and later DD evaluation."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock3D(in_channels, width, stride=1),
            nn.MaxPool3d(2),
            ConvBlock3D(width, width * 2, stride=1),
            nn.MaxPool3d(2),
            ConvBlock3D(width * 2, width * 4, stride=1),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        width: int = 32,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
    ) -> None:
        super().__init__()
        self.in_planes = width
        self.norm_layer = norm_layer
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(self.in_planes, planes, stride, norm_layer=self.norm_layer)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


class InvertedResidual3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv3d(in_ch, hidden_ch, kernel_size=1, bias=False),
                    nn.BatchNorm3d(hidden_ch),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv3d(
                    hidden_ch,
                    hidden_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_ch,
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_3D(nn.Module):
    """MobileNetV2-style 3D classifier for lightweight cross-architecture tests."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32) -> None:
        super().__init__()
        c1 = max(8, width)
        c2 = max(16, width * 2)
        c3 = max(24, width * 3)
        c4 = max(32, width * 4)
        c5 = max(64, width * 6)
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU6(inplace=True),
            InvertedResidual3D(c1, c1, stride=1, expand_ratio=1),
            InvertedResidual3D(c1, c2, stride=2, expand_ratio=6),
            InvertedResidual3D(c2, c2, stride=1, expand_ratio=6),
            InvertedResidual3D(c2, c3, stride=2, expand_ratio=6),
            InvertedResidual3D(c3, c3, stride=1, expand_ratio=6),
            InvertedResidual3D(c3, c4, stride=2, expand_ratio=6),
            InvertedResidual3D(c4, c4, stride=1, expand_ratio=6),
            nn.Conv3d(c4, c5, kernel_size=1, bias=False),
            nn.BatchNorm3d(c5),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def build_model(name: str, in_channels: int, num_classes: int, width: int = 32) -> nn.Module:
    key = name.lower()
    if key == "convnet3d":
        return SmallConvNet3D(in_channels=in_channels, num_classes=num_classes, width=width)
    if key in {"mobilenetv2_3d", "mobilenet_3d", "mobilenet3d"}:
        return MobileNetV2_3D(in_channels=in_channels, num_classes=num_classes, width=width)
    if key in {"torchvision_r3d18", "r3d_18_torchvision"}:
        try:
            from torchvision.models.video import r3d_18
        except ImportError as exc:
            raise ImportError("torchvision is required for eval_model=torchvision_r3d18") from exc
        model = r3d_18(weights=None, num_classes=num_classes)
        if in_channels != 3:
            old_conv = model.stem[0]
            model.stem[0] = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        return model
    if key == "resnet18_3d":
        return ResNet3D(in_channels=in_channels, num_classes=num_classes, layers=(2, 2, 2, 2), width=width)
    if key == "resnet10_3d_gn":
        return ResNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            layers=(1, 1, 1, 1),
            width=width,
            norm_layer=_group_norm,
        )
    if key == "resnet18_3d_gn":
        return ResNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            layers=(2, 2, 2, 2),
            width=width,
            norm_layer=_group_norm,
        )
    raise ValueError(f"Unknown model: {name}")
