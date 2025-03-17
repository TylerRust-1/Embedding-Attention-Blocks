import torch
from torch import nn
import torch.nn.functional as F
import clip


class EAB(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mid_dim = in_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
            nn.Softmax(-1)
        )

    def forward(self, image_features, embeddings):
        attention = self.attention(embeddings)
        return image_features * attention.unsqueeze(-1).unsqueeze(-1)


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class MergeBlock(nn.Module):
    def __init__(self, n_channels_low, n_channels_high):
        super().__init__()
        self.low_res_conv = nn.Sequential(
            nn.Conv2d(n_channels_low, n_channels_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_channels_high)
        )

    def forward(self, x):
        x_low_res = x.pop()
        x_high_res = x[-1]
        x_low_res = self.low_res_conv(x_low_res)
        x_low_res = resize(x_low_res, size=x_high_res.shape[2:])
        x[-1] = torch.relu_(x_low_res + x_high_res)


class SingleScaleModel(nn.Module):
    def __init__(self, n_channels, clip_model, image_encode_dim, text_encode_dim, dropout):
        super().__init__()

        self.clip_model = clip_model

        encode_dim = image_encode_dim + text_encode_dim * 2
        self.eab = EAB(encode_dim, n_channels)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(n_channels, 2, kernel_size=1)

    def forward(self, x, t):
        input_size = x.shape[2:]

        x = self.clip_model.encode_image(x)
        t = self.clip_model.encode_text(t)

        x = self.eab(x, t)

        x = self.dropout(x)
        x = self.classifier(x)
        x = resize(x, input_size)
        return x


class MultiScaleModel(nn.Module):
    def __init__(self, n_channels, clip_model, image_encode_dim, text_encode_dim, dropout):
        super().__init__()

        self.clip_model = clip_model

        encode_dim = image_encode_dim + text_encode_dim * 2
        self.eab = nn.ModuleList([
            EAB(encode_dim, n_channels[i]) for i in range(len(n_channels))
        ])

        self.merge_blocks = nn.ModuleList([
            MergeBlock(n_channels[i], n_channels[i-1]) for i in range(1, len(n_channels))
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(n_channels[0], 2, kernel_size=1)

    def forward(self, x, t):
        input_size = x.shape[2:]

        x = self.clip_model.encode_image(x)
        t = self.clip_model.encode_text(t)

        for i in range(len(x)):
            x[i] = self.eab[i](x[i], t)

        for i in range(len(x)-2, -1, -1):
            self.merge_blocks[i](x)
        x = x[0]

        x = self.dropout(x)
        x = self.classifier(x)
        x = resize(x, input_size)
        return x
