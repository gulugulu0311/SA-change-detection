import os
import torch
import logging

from datetime import datetime
from torch import nn
from torch import optim
from config import Configs
from models.TSSCD import TSSCD_FCN, TSSCD_TransEncoder, TSSCD_Unet
from utils import *
from data_loader import *
from metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def trainModel(model, configs, train_dl, test_dl, model_name='TSSCD_Unet', iter_num=200, fold=1):
    # log setting
    log_filename = f'models\\model_data\\log\\{model_name}\\{fold}.log'
    logger = logging.getLogger(f'logger_{fold}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # load data
    # train_dl, test_dl = load_data(batch_size=64)
    loss_fn = nn.CrossEntropyLoss()  # classification loss function
    loss_ch_noch = Diceloss()  # changed loss function
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)

    # Start training
    early_stopping = EarlyStopping(patience=16)
    best_acc, best_spatialscore, best_temporalscore = 0, 0, 0
    
    model_saved_times, last_saved_epoch = 0, 0
    model_metrics = dict()
    for epoch in range(iter_num):
        train_tqdm = tqdm(iterable=train_dl, total=len(train_dl))
        train_tqdm.set_description_str(f'Train epoch: {epoch}')
        
        train_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss1_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss2_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        
        for train_data, train_labels in train_tqdm:
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            if isinstance(model, TSSCD_TransEncoder):
                pred = model(train_data.permute(0, 2, 1).float())
                pred = pred.permute(0, 2, 1)
            else:
                pred = model(train_data.float())
            pre_label = torch.argmax(input=pred, dim=1)
            # time series has changed or not
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = train_labels.max(dim=1).values == train_labels.min(dim=1).values

            loss1 = loss_fn(pred, train_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss1_sum = torch.cat([train_loss1_sum, torch.unsqueeze(input=loss1, dim=-1)], dim=-1)
                train_loss2_sum = torch.cat([train_loss2_sum, torch.unsqueeze(input=loss2, dim=-1)], dim=-1)
                train_loss_sum = torch.cat([train_loss_sum, torch.unsqueeze(input=loss, dim=-1)], dim=-1)
                train_tqdm.set_postfix(
                    {'train loss': train_loss_sum.mean().item(), 'train loss1': train_loss1_sum.mean().item(),
                     'train loss2': train_loss2_sum.mean().item()})
        logger.info(f'Epoch {epoch}, Train loss: {train_loss_sum.mean().item()}。')
        train_tqdm.close()
        
        _, best_acc, best_spatialscore, best_temporalscore, \
        model_saved_times, last_saved_epoch,\
        model_metrics = validModel( test_dl, model, device, configs, logger,
                                    True, best_acc,
                                    best_spatialscore,
                                    best_temporalscore,
                                    epoch, last_saved_epoch,
                                    model_saved_times,
                                    model_name,
                                    fold=fold)
        
        logger.info(f'model saved {model_saved_times} times, last saved epoch: {last_saved_epoch}.\n')
        print(f'model saved {model_saved_times} times, last saved epoch: {last_saved_epoch}.\n')
        lr_scheduler.step()
        # early_stopping(valid_loss_sum)
        # if early_stopping.early_stop:
        #     break
    
    
    logger.removeHandler(file_handler)
    file_handler.close()
    return model_metrics

def validModel(test_dl, model, device, configs, logger, saveModel=True,
               best_acc=0, best_spatialscore=0, best_temporalscore=0,
               epoch=0, last_saved_epoch=0, model_saved_times=0, model_name='TSSCD_FCN', fold=1):
    evaluator = Evaluator(configs.classes)
    loss_fn = nn.CrossEntropyLoss()
    loss_ch_noch = Diceloss()
    with torch.no_grad():
        valid_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        valid_tqdm.set_description_str('Valid : ')
        valid_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        evaluator.reset()
        spatialscore = SpatialChangeDetectScore()
        temporalscore = TemporalChangeDetectScore(series_length=60, error_rate=1)
        for valid_data, valid_labels in valid_tqdm:
            valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)
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

            for pre, label in zip(predList, labelList):
                pre, label = pre[None, :], label[None, :]
                _, prechangepoints, pretypes = FilteringSeries(pre, method='Majority', window_size=3)
                _, labchangepoints, labtypes = FilteringSeries(label, method='NoFilter')
                spatialscore.addValue(labchangepoints[0], prechangepoints[0])
                spatialscore.addLccValue(pretypes[0], labtypes[0])
                temporalscore.addValue(labchangepoints[0], prechangepoints[0])
                
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
        print(f'MIoU:', f'{round(mIoU, 4)} ({round(best_acc, 4)})')
        # Spaital metrics
        spatialscore.getScore()
        spatial_f1 = spatialscore.spatial_f1
        print('spatial_LccAccuracy: ', f'{round(spatialscore.getLccScore(), 4)} ({round(best_spatialscore, 4)})')
        print('spatial_PA: ', round(spatialscore.spatial_pa, 4))
        print('spatial_UA: ', round(spatialscore.spatial_ua, 4))
        print('spatial_f1: ', round(spatial_f1, 4))
        # Temporal metrics
        temporalscore.getScore()
        temporal_f1 = temporalscore.temporal_f1
        print('temporal_CdAccuracy: ', f'{round(temporalscore.getCDScore(), 4)} ({round(best_temporalscore, 4)})')
        print('temporal_PA: ', round(temporalscore.temporal_pa, 4))
        print('temporal_UA: ', round(temporalscore.temporal_ua, 4))
        print('temporal_f1: ', round(temporalscore.temporal_f1, 4))
        
        logger.info(f'mIoU: {round(mIoU, 4)}; OA: {round(Acc, 4)}; AA: {round(Acc_mean, 4)}, {[round(i, 4) for i in Acc_class]}; F1: {round(F1, 4)}; Kappa: {round(Kappa, 4)};')
        logger.info(f'spatial_LccAccuracy: {round(spatialscore.getLccScore(), 4)}; spatial_PA: {round(spatialscore.spatial_pa, 4)}; spatial_UA: {round(spatialscore.spatial_ua, 4)}; spatial_F1: {round(spatial_f1, 4)}')
        logger.info(f'temporal_CdAccuracy: {round(temporalscore.getCDScore(), 4)}; temporal_PA: {round(temporalscore.temporal_pa, 4)}; temporal_UA: {round(temporalscore.temporal_ua, 4)}; temporal_F1: {round(temporalscore.temporal_f1, 4)}')
        if saveModel:
            if not os.path.exists(os.path.join('models\\model_data')):
                os.mkdir(os.path.join('models\\model_data'))
            if mIoU >= best_acc and spatialscore.getLccScore() >= best_spatialscore:
                torch.save(model.state_dict(), os.path.join(f'models\\model_data\\{model_name}', f'{fold}.pth'))
                logger.info(f'Epoch {epoch} saved.')
                best_acc = mIoU
                best_spatialscore = spatialscore.getLccScore()
                best_temporalscore = temporalscore.getCDScore()
                
                model_saved_times += 1
                last_saved_epoch = epoch
            return valid_loss_sum.mean().item(), best_acc, best_spatialscore, best_temporalscore,\
                   model_saved_times, last_saved_epoch,\
                   { # Finally saved model's metrics
                       'mIoU': round(mIoU, 4),
                       'spatial_LccAccuracy': round(spatialscore.getLccScore(), 4),
                       'temporal_CdAccuracy': round(temporalscore.getCDScore(), 4),
                       
                       'OA': round(Acc, 4),
                       'AA': round(Acc_mean, 4),
                       'Acc_class': [round(i, 4) for i in Acc_class],
                       'F1': round(F1, 4),
                       'Kappa': round(Kappa, 4),
                       'spatial_PA': round(spatialscore.spatial_pa, 4),
                       'spatial_UA': round(spatialscore.spatial_ua, 4),
                       'spatial_F1': round(spatial_f1, 4),
                       'temporal_PA': round(temporalscore.temporal_pa, 4),
                       'temporal_UA': round(temporalscore.temporal_ua, 4),
                       'temporal_F1': round(temporalscore.temporal_f1, 4)
                   }
        else:
            return
def generate_model_instances(configs):
    return zip(
        ['TSSCD_Unet', 'TSSCD_TransEncoder', 'TSSCD_FCN'],
        [
            TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
            TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_dmodel),
            TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden)
        ]
    )

if __name__ == '__main__':
    iter_num, batch_size = 200, 64
    configs = Configs()
    
    # # 验证集 —— 随机排列交叉验证 n_split 次
    tralid, test = load_data(batch_size=batch_size, split_rate=0.8)
    # for fold, (train_dl, valid_dl) in enumerate(random_permutation(tralid, n_split=5, split_rate=0.75, batch_size=64)):
    #     for model_name, model in generate_model_instances(configs):
    #         if model_name == 'TSSCD_TransEncoder':
    #             continue
    #         model = model.to(device=device)
    #         model_metrics = trainModel(
    #             model=model, 
    #             configs=configs, # just for class_nums
    #             train_dl=train_dl, test_dl=valid_dl,
    #             model_name=model_name, 
    #             iter_num=iter_num, 
    #             fold=fold+21
    #         )
    # 测试集
    for model_name, model in generate_model_instances(configs):
        # if model_name == 'TSSCD_TransEncoder':
        #     continue
        model = model.to(device=device)
        model_metrics = trainModel(
            model=model,
            configs=configs, # just for class_nums
            train_dl=make_dataloader(tralid, type='train', is_shuffle=True, batch_size=batch_size), 
            test_dl=make_dataloader(test, type='test', is_shuffle=None, batch_size=batch_size),
            model_name=model_name,
            iter_num=iter_num,
            fold=1001
        )
        
    # Additional model (use all samples)
    # for j in range(110, 121):
    #     tralid, test = load_data(batch_size=batch_size, split_rate=0.8)
    #     for model_name, model in generate_model_instances(configs):
    #         # if model_name != 'TSSCD_Unet':
    #         #     continue
    #         model = model.to(device=device)
    #         model_metrics = trainModel(
    #             model=model,
    #             configs=configs, # just for class_nums
    #             train_dl=make_dataloader(tralid, type='train', is_shuffle=True, batch_size=batch_size), 
    #             test_dl=make_dataloader(test, type='test', is_shuffle=None, batch_size=batch_size),
    #             model_name=model_name,
    #             iter_num=iter_num,
    #             fold=j 
    #         )

