import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):

    def __init__(self, inp, oup, k=3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp,oup,kernel_size=k,stride=1,padding=k // 2,bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class K210_YOLO(nn.Module):
    """
    K210轻量YOLO模型
    输入:
        [B, 3, 224, 224]
    训练输出:
        [B, 7, 7, 5, 8]
    部署 raw 输出:
        [B, 40, 7, 7]
    """
    def __init__(self, num_classes=3, num_anchors=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)
        # Backbone
        # 目标：1MB 左右 kmodel
        # 全普通 Conv
        # 无 stride=2 Conv
        # 全 MaxPool 下采样

        self.features = nn.Sequential(

            # 224 -> 112
            ConvBNReLU(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 112 -> 56
            ConvBNReLU(16, 24),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 56 -> 28
            ConvBNReLU(24, 48),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 28 -> 14
            ConvBNReLU(48, 80),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 14 -> 7
            ConvBNReLU(80, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7 x 7
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
        )

        # Detection Head

        self.head = nn.Sequential(
            ConvBNReLU(128, 64),
            nn.Conv2d(64,out_channels,kernel_size=1,stride=1,padding=0,bias=True))

    def forward_raw(self, x):
        #部署导出用:
        #输出 [B, 40, 7, 7]
        x = self.features(x)
        out = self.head(x)
        return out

    def forward(self, x):
        """
        训练用:
        输出 [B, 7, 7, 5, 8]
        """
        out = self.forward_raw(x)
        B, _, H, W = out.shape
        out = out.reshape(B,self.num_anchors,5 + self.num_classes,H,W)
        out = out.permute(0,3,4,1,2).contiguous()
        return out


if __name__ == "__main__":

    model = K210_YOLO(
        num_classes=3,
        num_anchors=5
    )

    x = torch.randn(1, 3, 224, 224)

    y_train = model(x)
    y_raw = model.forward_raw(x)

    print("train output:", y_train.shape)
    print("raw output:", y_raw.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("params:", total_params)
    print("params MB int8 approx:", total_params / 1024 / 1024)