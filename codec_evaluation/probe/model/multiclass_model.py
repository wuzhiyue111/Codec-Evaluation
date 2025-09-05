import torch
import torch.nn as nn


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

    def forward(self, feature):
        """
        feature: [B, T, D]
        B: batch size
        T: 时间维度
        D: 特征维度（即音频的通道数）
        """
        B, T, _ = feature.shape  
        feature_avg = self.avg_pool(feature)    
    
        feature_avg = feature_avg.view(B, T)  
        feature_fc = self.fc(feature_avg)    
        
        feature_fc = feature_fc.view(B, T, 1)  
        return feature * feature_fc.expand_as(feature)  
    

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        """
        Args:
            in_ch: Number of input feature channels 
            out_ch: Number of output feature channels
        """

        super(DSConv, self).__init__()
        # deep convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=in_ch,  
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch, 
            bias=False
        )
        # 逐点卷积
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,  
            kernel_size=1,  
            stride=1,
            padding=0,
            bias=False
        )
    def forward(self, x):
        x = self.depthwise_conv(x) 
        x = self.pointwise_conv(x)
        
        return x
    
class MulticlassProber(nn.Module):
    def __init__(self, 
                 codec_dim, 
                 target_T,
                 num_outputs,
                 drop_out = 0.1,
                 channel_reduction = 16,
                 padding = 1,
                 kernel_size = 3,
                 stride = 1,
                 ):
        super(MulticlassProber, self).__init__()
        self.num_outputs = num_outputs
        self.target_T = target_T

        current_T = target_T
        for ratio in [1, 2]:
            setattr(self, f'channel_attention{ratio}', nn.Sequential(
                SEBlock(channel=current_T, reduction=channel_reduction),
                SEBlock(channel=current_T, reduction=channel_reduction),
                SEBlock(channel=current_T, reduction=channel_reduction)
            ))
            setattr(self, f'dsconv{ratio}', DSConv(
                in_ch=current_T,
                out_ch=current_T // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            current_T //= 2

        self.drop_out = nn.Dropout(p=drop_out)
        self.init_linear(codec_dim = codec_dim)

    def init_linear(self, codec_dim):
        self.linear = nn.Linear(codec_dim, codec_dim // 16) 
        input_dim = (self.target_T // 4) * (codec_dim // 16)

        self.output = nn.Linear(input_dim, self.num_outputs)

    def group_mean(self, data, n_segment_list):
        if sum(n_segment_list) != len(data):
            raise ValueError("length error!")

        groups = torch.split(data, n_segment_list)

        result = []
        for group in groups:
            
            group_means = group.float().mean(dim=0)  # 按列求均值
            result.append(group_means)
        output_tensor = torch.stack(result, dim=0) 
        return output_tensor

    def forward(self, x, truth_label, n_segment_list):    
        """
        x:[B*n_segments, D, T] 
        n_segments: Number of audio segments
        """ 
        x = x.permute(0, 2, 1)   #[B*n_segments, T, D] 
        x_channel = self.channel_attention1(x)
        x_conv = self.dsconv1(x_channel)  
        x_channel = self.channel_attention2(x_conv)
        x_conv = self.dsconv2(x_channel)    #[B*n_segments, T//4, D]

        x_conv = self.linear(x_conv)      #[B*n_segments, T', D//16] 

        x_flattened = x_conv.flatten(start_dim=1, end_dim=-1)    #[B*n_segments, input_dim=T' * D//16]

        output = self.output(x_flattened)  #[B*n_segments, 24]

        if output.shape[0] != truth_label.shape[0]:
            output = self.group_mean(data=output, n_segment_list=n_segment_list) 
        output = self.drop_out(output)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(output, truth_label)

        return loss, output