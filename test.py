import os
import torch
from torch import nn

from config import Configs
from utils import *
from datetime import datetime
from data_loader import load_data
from models.TSSCD import TSSCD_FCN, TSSCD_TransEncoder, TSSCD_Unet
from metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore

class Diceloss(nn.Module):
    def __init__(self, smooth=1.):
        super(Diceloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=0).sum(dim=0)
        loss = (1 - ((2. * intersection + self.smooth) / (
                pred.sum(dim=0).sum(dim=0) + target.sum(dim=0).sum(dim=0) + self.smooth)))
        return loss.mean()

def validModel(test_dl, model):
    evaluator = Evaluator(configs.classes)
    loss_fn = nn.CrossEntropyLoss()
    loss_ch_noch = Diceloss()
    FP_list = list()
    
    with torch.no_grad():
        valid_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        valid_tqdm.set_description_str('Valid : ')
        valid_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        evaluator.reset()
        spatialscore = SpatialChangeDetectScore()
        temporalscore = TemporalChangeDetectScore(series_length=60, error_rate=1)
        
        for valid_data, valid_labels in valid_tqdm:
            valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)
            valid_data = valid_data[:, :-1, :]
            location_data = valid_data[:, -1, 7].cpu().numpy()
            print(location_data)
            if isinstance(model, TSSCD_TransEncoder):
                valid_pred = model(valid_data.permute(0, 2, 1).float())
                valid_pred = valid_pred.permute(0, 2, 1)
            else:
                valid_pred = model(valid_data.float())
                
            pre_label = torch.argmax(input=valid_pred, dim=1)
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = valid_labels.max(dim=1).values == valid_labels.min(dim=1).values
            
            loss1 = loss_fn(valid_pred, valid_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            valid_loss = loss1 + loss2

            evaluator.add_batch(valid_labels.cpu().numpy(), torch.argmax(input=valid_pred, dim=1).cpu().numpy())

            valid_loss_sum = torch.cat([valid_loss_sum, torch.unsqueeze(input=valid_loss, dim=-1)], dim=-1)
            valid_tqdm.set_postfix({'valid loss': valid_loss_sum.mean().item()})

            predList = torch.argmax(input=valid_pred, dim=1).cpu().numpy()
            labelList = valid_labels.cpu().numpy()

            for pre, label, location in zip(predList, labelList, location_data):
                pre, label = pre[None, :], label[None, :]
                _, prechangepoints, pretypes = FilteringSeries(pre, method='Majority', window_size=3)
                _, labchangepoints, labtypes = FilteringSeries(label, method='NoFilter')
                spatialscore.addValue(labchangepoints[0], prechangepoints[0])
                spatialscore.addLccValue(pretypes[0], labtypes[0])
                temporalscore.addValue(labchangepoints[0], prechangepoints[0])
                if np.array_equal(pretypes[0], labtypes[0]):
                    FP_list.append((location, pre[0], label[0]))
                
        valid_tqdm.close()

        # Evaluation Accuracy
        Acc = evaluator.Pixel_Accuracy()
        Acc_class, Acc_mean = evaluator.Class_Accuracy()
        print('OA:', round(Acc, 4))
        print('AA:', round(Acc_mean, 4), '; Acc_class:', [round(i, 4) for i in Acc_class])
        F1 = evaluator.F1()
        print('F1:', round(F1, 4))
        Kappa = evaluator.Kappa()
        print('Kappa:', round(Kappa, 4))
        mIoU = evaluator.Mean_Intersection_over_Union()
        print(f'MIoU:', f'{round(mIoU, 4)}')
        # Spaital metrics
        spatialscore.getScore()
        spatial_f1 = spatialscore.spatial_f1
        print('spatial_LccAccuracy: ', f'{round(spatialscore.getLccScore(), 4)}')
        print('spatial_PA: ', round(spatialscore.spatial_pa, 4))
        print('spatial_UA: ', round(spatialscore.spatial_ua, 4))
        print('spatial_f1: ', round(spatial_f1, 4))
        # Temporal metrics
        temporalscore.getScore()
        print('temporal_CdAccuracy: ', f'{round(temporalscore.getCDScore(), 4)}')
        print('temporal_PA: ', round(temporalscore.temporal_pa, 4))
        print('temporal_UA: ', round(temporalscore.temporal_ua, 4))
        print('temporal_f1: ', round(temporalscore.temporal_f1, 4))
        return FP_list
if __name__ == '__main__':
    configs = Configs()
    device = device_on()
    
    train_dl, test_dl = load_data(batch_size=64, test_mode=True)
    
    model_instances = {
        'TSSCD_FCN': TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_Unet': TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_TransEncoder': TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_dmodel)
    }
    model_name = 'TSSCD_TransEncoder'
    model = model_instances['TSSCD_TransEncoder']
    model = model.to(device=device)
    
    model_state_dict = torch.load(os.path.join(f'model_data\\{model_name}', '2.pth'), map_location='cuda', weights_only=True)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    FP_list = validModel(
        test_dl=test_dl,
        model=model
    )
    print(FP_list[7])