# -*- coding: utf-8 -*-
# @Time    : 2024/6/20 15:37 
# @Author  : Js Chen
# @File    : UDS_Net.py
# @Software: PyCharm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params
from .HAD_Decoder import HADD
from .MFAM import MFAM
# from HAD_Decoder import HADD
# from MFAM import MFAM

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):

        conv = nn.Conv2d(in_channels,out_channels,kernel_size, stride=stride,  padding=padding, bias=not (use_batchnorm) )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None,):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels = in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x



class UDSnetDecode(nn.Module):
    def __init__(
        self,
        encoder_channels,  
        decoder_channels,  
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        

        encoder_channels = encoder_channels[1:]  # [64,48,80,224,640]
        encoder_channels = encoder_channels[::-1]  #  [640,224,80,48,64]
        

        # self.x_0_0 = DecoderBlock(640, 224, 256, use_batchnorm, attention_type)
        # self.x_0_0 = Attention_Decoder(640, 224,256, size=32, num_heads=16,ratio=10)
        # self.x_0_0 = Block_decoder(640, 224, 256,size=16*2)
        self.x_0_0 = HADD(640, 224, 256, size=16*2)
        
        # Layer 1 
        self.x_1_0 = DecoderBlock(224, 80, 224, use_batchnorm, attention_type)
        # self.x_0_1 = DecoderBlock(256, 80+224, 128, use_batchnorm, attention_type)  # skip_channels
        # self.x_0_1 = MCAFTM_Decode(in_channels=256, skip_channels = 80+224, out_channels=128, num_heads=4)
        # self.x_0_1 = Attention_Decoder(256, 80+224,128, size=64, num_heads=8,ratio=10)
        # self.x_0_1 = Block_decoder(256, 80+224, 128, size=32*2)
        self.x_0_1 = HADD(256, 80+224, 128, size=32*2)
        
        # Layer 2
        self.x_2_0 = DecoderBlock(80, 48, 80, use_batchnorm, attention_type)
        self.x_1_1 = DecoderBlock(224, 48+80, 224, use_batchnorm, attention_type)
        # self.x_0_2 = DecoderBlock(128, 48+80+224, 64, use_batchnorm, attention_type)
        # self.x_0_2 = MCAFTM_Decode(in_channels=128, skip_channels = 48+80+224, out_channels=64, num_heads=4)
        # self.x_0_2 = Attention_Decoder(128, 48+80+224, 64, size=128, num_heads=4,ratio=10)
        # self.x_0_2 = Block_decoder(128, 48+80+224, 64, size=64*2)
        self.x_0_2 = HADD(128, 48+80+224, 64, size=64*2)
        
        # Layer 3
        self.x_3_0 = DecoderBlock(48, 64, 48, use_batchnorm, attention_type)
        self.x_2_1 = DecoderBlock(80, 64+48, 80, use_batchnorm, attention_type)
        self.x_1_2 = DecoderBlock(224, 64+48+80, 224, use_batchnorm, attention_type)
        # self.x_0_3 = DecoderBlock(64, 64+48+80+224, 32, use_batchnorm, attention_type)
        # self.x_0_3 = MCAFTM_Decode(in_channels=64, skip_channels = 64+48+80+224, out_channels=32, num_heads=8)
        # self.x_0_3 = Attention_Decoder(64, 64+48+80+224, 32, size=256, num_heads=2,ratio=10)
        # self.x_0_3 = Block_decoder(64, 64+48+80+224, 32, size=128*2)
        self.x_0_3 = HADD(64, 64+48+80+224, 32, size=128*2)
        
        # Layer 4
        self.x_0_4 = DecoderBlock(32, 0, 16, use_batchnorm, attention_type)
        # self.x_0_4 = DecoderBlockLast(32, 0, 16, size=256*2)

    def forward(self, *features):

        encoder_features = features[1:]  # [e0, e1, e2, e3, e4]
        e4, e3, e2, e1, e0 = encoder_features[::-1]  #  [e4(640), e3(224), e2(80), e1(48), e0(64)]
        
  
        # self._validate_feature_shapes(e4, e3, e2, e1, e0)
        
  
        # Stage 0 (16x16 → 32x32)
        x0 = self.x_0_0(e4, e3)  # 640+224 →256
        
        # Stage 1 (32x32 → 64x64)
        s1_0 = self.x_1_0(e3, e2)  # 224+80 →224
        x1 = self.x_0_1(x0, torch.cat([e2, s1_0], dim=1))  # 256 + (80+224) →128
        
        # Stage 2 (64x64 → 128x128)
        s2_0 = self.x_2_0(e2, e1)  # 80+48 →80
        s1_1 = self.x_1_1(s1_0, torch.cat([e1, s2_0], dim=1))  # 224 + (48+80) →224
        x2 = self.x_0_2(x1, torch.cat([e1, s2_0, s1_1], dim=1))  # 128 + (48+80+224) →64
        
        # Stage 3 (128x128 → 256x256)
        s3_0 = self.x_3_0(e1, e0)  # 48+64 →48
        s2_1 = self.x_2_1(s2_0, torch.cat([e0, s3_0], dim=1))  # 80 + (64+48) →80
        s1_2 = self.x_1_2(s1_1, torch.cat([e0, s3_0, s2_1], dim=1))  # 224 + (64+48+80) →224
        x3 = self.x_0_3(x2, torch.cat([e0, s3_0, s2_1, s1_2], dim=1))  # 64 + (64+48+80+224) →32
        
        # Final stage (256x256 → 512x512)
        output = self.x_0_4(x3)

        return output
    

def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc
        #  EfficientLocalizationAttention
        # self.ela_layers = nn.ModuleList([
        #     EfficientLocalizationAttention(out_channels[i]) if i > 0 else nn.Identity()
        #     for i in range(len(out_channels))
        # ])
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0] : self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1] : self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2] :],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)

            features.append(x)
            # x = self.ela_layers[i](x)  # 

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    encoders = {
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38, 55),
            "model_name": "efficientnet-b7",
        },
    },
}
    Encoder = encoders[name]["encoder"]

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)

    return encoder

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class UDS_Net(nn.Module):
    def __init__(self, encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=3, classes=1):
        super(UDS_Net, self).__init__()

        self.encoder_depth=5
        self.decoder_use_batchnorm= True,
        self.decoder_channels= (256, 128, 64, 32, 16)
        self.decoder_attention_type= None
        self.classes = classes

        self.encoder = get_encoder( encoder_name, in_channels=in_channels, depth=self.encoder_depth, weights=encoder_weights)

        self.decoder = UDSnetDecode(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            use_batchnorm=self.decoder_use_batchnorm,
            attention_type=self.decoder_attention_type,
        )
        
        self.decoder1 = UDSnetDecode(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            use_batchnorm=self.decoder_use_batchnorm,
            attention_type=self.decoder_attention_type,
        )

        self.segmentation_head1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,#upsampling=2
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,#upsampling=2
        )
        self.MLFM = MFAM()

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        # Encoder
        features = self.encoder(x)

        # 
        feat1, feat2, feat3, feat4, feat5, feat6 = features
        # print(feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape, feat6.shape)
        features6 = self.MLFM([feat2, feat3, feat4, feat5, feat6])
        features1 = feat1, feat2, feat3, feat4, feat5, features6
        
        
        # Decoder
        decoder_output = self.decoder(*features1)
        # print(decoder_output.shape)
        results  = self.segmentation_head1(decoder_output)
        results = nn.Softplus()(results)

        decoder_output_1 = self.decoder1(*features1)
        results_1 = self.segmentation_head_1(decoder_output_1)
        ### center crop
        std = crop(results_1, img_H, img_W, 0 , 0)
        std = nn.Softplus()(std)

        return results, std
    

# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w , crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data




