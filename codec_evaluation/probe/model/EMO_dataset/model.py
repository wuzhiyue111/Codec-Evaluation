import torch.nn as nn
import math
import torch.nn.functional as F
from einops import reduce

class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        """
        input: [B, T, D]
        B: batch size
        T: 时间维度
        D: 特征维度（即音频的通道数）
        """
        b, d, _ = x.shape  
        y = self.avg_pool(x)    
    
        y = y.view(b, d)  
        y = self.fc(y)    
        
        y = y.view(b, d, 1)  
        return x * y.expand_as(x)  
    

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(DSConv, self).__init__()
        # deep convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,  
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch, 
            bias=False
        )

    def forward(self, x):
        x = self.depthwise_conv(x) 
        
        return x
    
class EMOProber(nn.Module):
    def __init__(self, 
                 codec_dim, 
                 token_rate,
                 target_sec,
                 n_segments,
                 num_outputs,
                 channel_reduction = 16,
                 padding = 1,
                 kernel_size = 3,
                 stride = 2,
                 ):
        super(EMOProber, self).__init__()
        self.num_outputs = num_outputs 
        self.n_segments = n_segments
        self.channel_attention = nn.Sequential(
            SEBlock(channel = codec_dim, reduction = channel_reduction),
            SEBlock(channel = codec_dim, reduction = channel_reduction),
            SEBlock(channel = codec_dim, reduction = channel_reduction)
            )
        
        self.dsconv = DSConv(in_ch=codec_dim,                
                            out_ch=codec_dim , 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding)
        
        self.init_linear(token_rate = token_rate, 
                         target_sec = target_sec, 
                         n_segments = n_segments, 
                         padding = padding, 
                         kernel_size = kernel_size, 
                         stride = stride, 
                         codec_dim = codec_dim)

    def init_linear(self, 
                    token_rate,
                    target_sec,
                    n_segments,
                    padding,
                    kernel_size,
                    stride,
                    codec_dim,
                    ):
        self.linear = nn.Linear(codec_dim, codec_dim // 16)

        self.target_T = token_rate * target_sec // n_segments
        T1 = math.floor((self.target_T + 2 * padding - kernel_size) / stride)  + 1 
        input_dim = (math.floor((T1 + 2 * padding - kernel_size) / stride) + 1) * codec_dim // 16

        self.output = nn.Linear(input_dim, self.num_outputs)

    def forward(self, x, y):
        x = x.float()  #[B*n_segments, D, T] 

        x_channel = self.channel_attention(x)
        x_conv = self.dsconv(x_channel)  
        x_channel = self.channel_attention(x_conv)
        x_conv = self.dsconv(x_channel)    #[B*n_segments, D, T']

        x_conv = x_conv.permute(0, 2, 1)  #[B*n_segments, T', D]
        x_conv = self.linear(x_conv)      #[B*n_segments, T', D//16]

        x_flattened = x_conv.flatten(start_dim=1, end_dim=-1)    #[B*n_segments, input_dim=T' * D//16]

        output = self.output(x_flattened)  #[B*n_segments, 2]
        if output.shape[0] != y.shape[0]:
            output = reduce(output, '(b g) n -> b n', reduction = 'mean', g = self.n_segments) 

        loss = F.mse_loss(output, y)

        return loss, output