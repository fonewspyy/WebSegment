import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.UNet import DownBlock, DoubleConv, UpBlock


class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.down_conv1 = DownBlock(in_channels, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)

    def forward(self, x):
        x1, skip1 = self.down_conv1(x)
        x2, skip2 = self.down_conv2(x1)
        x3, skip3 = self.down_conv3(x2)
        x4, skip4 = self.down_conv4(x3)
        return [skip1, skip2, skip3, skip4, x4]

class UNet_Partial(nn.Module):
    def __init__(self, in_channels, out_channels, text_embeddings, up_sample_mode='conv_transpose'):
        super(UNet_Partial, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_sample_mode = up_sample_mode
        self.text_embeddings = text_embeddings

        self.encoder = UNetEncoder(in_channels=self.in_channels)
        self.gap = nn.Sequential(
            # 기존 코드에서는 group norm
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        )
        # Bottleneck(Downsampling path - Upsampling path 연결하는 부분)
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # FC
        # self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

        self.weight_nums = [8 * 8, 8 * 8, 8 * 1]
        self.bias_nums = [8, 8, 1]

        self.controllers = nn.ModuleList()
        for icls in range(self.out_channels):
            # class별 경량화 헤드
            self.controllers.append(
                nn.Sequential(
                    nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, sum(self.weight_nums+self.bias_nums), kernel_size=1, stride=1, padding=0),
                )
            )

        self.precls_conv = nn.Sequential(
            # 기존 코드에서는 group norm
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, kernel_size=1)
        )

        # BatchNorm layers 추가
        # self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(8*out_channels) for _ in range(len(self.weight_nums) - 1)])

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        # params.shape = [class_num, param_size]
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        # 153
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        # class 수
        num_insts = params.size(0)
        # layer 개수
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        # 2d에서는 features의 차원은 4인지 확인할 것
        assert features.dim() == 4
        n_layers = len(weights)
        x = features

        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            # group convolution을 수행함
            if torch.isnan(w).any() or torch.isinf(w).any():
                print(f'nan or inf detected in weights at layer {i}')
            if torch.isnan(b).any() or torch.isinf(b).any():
                print(f'nan or inf detected in bias at layer {i}')

            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            # NaN 값 체크
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i}")
            if i < n_layers - 1:
                x = torch.clamp(x, min=-10, max=10)
                # x = self.batch_norm_layers[i](x)
                x = F.relu(x)
        return x

    def forward(self, x):
        [skip1, skip2, skip3, skip4, x4] = self.encoder(x)
        # x_feat : 256-dim feature
        x = self.double_conv(x4)
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        out = self.up_conv1(x, skip1)

        # [B, C, 1, 1]
        x_feat = self.gap(x4)
        b = x_feat.shape[0]
        logits_array = []

        # class별 파라미터 생성
        class_params = list()
        for icls in range(self.out_channels):
            # CLIP Embedding
            # self.text_embeddings[icls].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(b, -1, 1, 1) => [B, D, 1, 1]
            # x_cond => [B, C+D, 1, 1]

            x_cond = torch.cat([x_feat, self.text_embeddings[icls].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(b, -1, 1, 1)], dim=1)
            # x_cond = x_feat
            params = self.controllers[icls](x_cond)
            class_params.append(params.squeeze(-1).squeeze(-1))  # 마지막 두 차원을 제거하여 파라미터 텐서를 (batch_size, param_size) 형태로 변환
        class_params = torch.stack(class_params, dim=1)  # (batch_size, num_classes, param_size)

        # batch 내 샘플별로 클래스별 예측 생성
        for i in range(b):
            params = class_params[i]

            # batch sample별 out 값
            head_inputs = self.precls_conv(out[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.out_channels, 1, 1, 1)
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)

            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        logits_array = torch.cat(logits_array, dim=0)
        logits_array = torch.clamp(logits_array, min=-10, max=10)


        return logits_array