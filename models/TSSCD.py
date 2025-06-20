import os
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F

from utils import *
from config import Configs
from torch import nn
from torchsummary import summary

class TSSCD_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(TSSCD_Unet, self).__init__()
        self.out_channels = out_channels
        self.config = config
        # 128, 256, 512, 1024, 4096
        c1, c2, c3, c4, c5 = config
        self.embedding = nn.Conv1d(in_channels, c1, 1)
        # 第一层卷积 output length: 60
        self.layer1 = nn.Sequential(
            nn.Conv1d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 第二层卷积 output length: 30
        self.layer2 = nn.Sequential(
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # Downsampling 1/2, Temporal Length = 30
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第三层卷积 output length: 15
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # Downsampling 1/2, Temporal Length = 30
            
            nn.Conv1d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 第四层卷积 output length: 8
        self.layer4 = nn.Sequential(
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # Downsampling 1/2, Temporal Length = 30
            
            nn.Conv1d(c3, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 第五层卷积 + 反卷积 8 → 8
        self.layer5 = nn.Sequential(
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # Downsampling 1/2, Temporal Length = 30
            
            nn.Conv1d(c4, c5, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c5, c5, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c5, c4, 4, 2, 1, bias=False)
        )

        # 反卷积层 ALL 2×
        self.upconv_layer4 = nn.Sequential( 
            nn.Conv1d(c4 + c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c4, c3, 4, 2, 1, bias=False),
        )
        self.upconv_layer3 = nn.Sequential(
            nn.Conv1d(c3 + c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c3, c2, 4, 2, 1, bias=False),
        )
        self.upconv_layer2 = nn.Sequential(
            nn.Conv1d(c2 + c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c2, c1, 4, 2, 1, bias=False),
        )
        self.upconv_layer1 = nn.Sequential(
            nn.Conv1d(c1 + c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, out_channels, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # h = self.layer1(x) # length: 60
        h = self.layer1(self.embedding(x))
        
        self.s1 = self.layer2(h) # length: 30
        self.s2 = self.layer3(self.s1)  # length: 15
        self.s3 = self.layer4(self.s2)  # length: 8
        self.s4 = self.layer5(self.s3)  # length: 8
        
        s4 = torch.cat([self.s3, self.s4], dim=1)
        s4 = self.upconv_layer4(s4)  # length: 16
        
        s3 = torch.cat([self.s2, s4[:, :, :15]], dim=1)
        s3 = self.upconv_layer3(s3)  # length: 30

        s2 = torch.cat([self.s1, s3], dim=1)
        s2 = self.upconv_layer2(s2)  # length: 60  

        s2 = torch.cat([h, s2], dim=1)
        s1 = self.upconv_layer1(s2)  # length: 60
    
        return s1
    
class TSSCD_FCN(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(TSSCD_FCN, self).__init__()
        self.out_channels = out_channels
        self.config = config
        # 128, 256, 512, 1024, 4096
        c1, c2, c3, c4, c5 = config
        
        # 第一层卷积 60 → 30
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/2, Temporal Length = 30
        )
        
        # 第二层卷积 30 → 15
        self.layer2 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/4, Temporal Length = 15
        )

        # 第三层卷积 15 → 8
        self.layer3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            # modified by FCN-8s
            # ==================================
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            # ==================================
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/8, Temporal Length = 8
        )

        # 第四层卷积 8 → 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(c3, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            # modified by FCN-8s
            # ==================================
            nn.Conv1d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            # ==================================
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/16, Temporal Length = 4
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(c4, c5, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c5, c5, 1),
            nn.ReLU(inplace=True)
        )
        
        # 第六层使用卷积层取代FC层
        self.score_1 = nn.Conv1d(c5, out_channels, 1)
        self.score_2 = nn.Conv1d(c3, out_channels, 1)
        self.score_3 = nn.Conv1d(c2, out_channels, 1)

        # 第七层反卷积 L_out = (L_in - 1) × stride - 2 × padding + dilation × (kernel_size - 1) + output_padding + 1
        # scale 2x
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)    #  8 → 16
        # cropped NOW
        # scale 4x
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 6, 4, 1, bias=False)    # 15 → 60
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x) # length: 30
        self.s1 = self.layer2(h) # s1: length: 15
        self.s2 = self.layer3(self.s1)  # s2: length: 8
        self.s3 = self.layer4(self.s2)  # s3: length: 4
        self.s4 = self.layer5(self.s3)  # s4: length: 4
        
        s4 = self.score_1(self.s4)
        s4 = self.upsampling_2x(s4) # s3: length: 4 → 8
        s2 = self.score_2(self.s2)
        
        s2 += s4 # length: 8
        s2 = self.upsampling_4x(s2) # s2: length: 8 → 16
        s2 = s2[:, :, :15]  # s2: length: 15 cropped
        
        s1 = self.score_3(self.s1) # s1: length: 15
        
        score = s1 + s2
        score = self.upsampling_8x(score) # s1: length: 15 → 60
        return score

class TSSCD_TransEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_model):
        super(TSSCD_TransEncoder, self).__init__()
        self.embedding = nn.Linear(in_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=0.1, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, out_channels)
        )

    def forward(self, x):
        # Batch × Channel × Length
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        output = self.decoder(x)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(1000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        pe = self.pe[:, :x.size(1), :]
        x = x + pe
        return x

if __name__ == '__main__':
    batch_size, seq_len = 32, 60
    device = device_on()
    configs = Configs()
    models = {
        'TSSCD_Unet': TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_FCN': TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_TransEncoder': TSSCD_TransEncoder(configs.input_channels, configs.classes, 64)
    }
    model_accs = {key: {} for key in models.keys()}
    for model_name, model in models.items():
        # load weights
        model = model.to(device)
        model_state_dict = torch.load(os.path.join(f'models\\model_data\\{model_name}', '88.pth'), map_location='cuda', weights_only=True)
        model.load_state_dict(model_state_dict)
        model.eval()
        # metrics
        log_files = [os.path.join(f'models\\model_data\\log\\{model_name}', i) \
                    for i in os.listdir(f'models\\model_data\\log\\{model_name}') \
                    ]
        
        infos = [extract_accuracy_from_log(log_file) for log_file in log_files]
        model_saved_accuracy = [info[0][info[1]] for info in infos if info is not None]

        main_metric = 'spatial_LccAccuracy'
        # main_metric = 'mIoU'
        print(f'\n====={model_name} {main_metric} rank:=====')
        for acc_dict in sorted(model_saved_accuracy, 
                               key=lambda x: x[main_metric], 
                               reverse=True)[:6]:
            print(f'{acc_dict['pth']}: {acc_dict[main_metric]}')
        model_data_idx = max(enumerate(model_saved_accuracy), key=lambda x: x[1][main_metric])[0]
        print(f'{model_name}: {model_saved_accuracy[model_data_idx]['pth']}.pth reaches the best {main_metric}')
        
        metric_values = dict()
        for acc_dict in model_saved_accuracy:
            for metric, value in acc_dict.items():
                if metric == 'pth': continue
                if metric not in metric_values:
                    metric_values[metric] = list()
                metric_values[metric].append(value)
        
        for metric, values in metric_values.items():
            model_accs[model_name].update({metric: f'{np.mean(values):.4f} ± {np.std(values):.4f}'})
    for model_name, acc_dict in model_accs.items(): 
        print(f'================{model_name}================')
        for metric, value in acc_dict.items(): print(f'{metric}:\t{value}')
        
    # Batch × Channel × Length
    # input_tensor = torch.randn(batch_size, 12, seq_len).to(device)
    # output = model(input_tensor)
    # print(output.shape)
    
    # summary(model, input_size=(60, 12), batch_size=-1)