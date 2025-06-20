import numpy as np
import random

from torch.utils import data
from utils import *
from tqdm import tqdm
from collections import Counter

class MaskDataset(data.Dataset):
    def __init__(self, paths, type):
        super(MaskDataset, self).__init__()
        self.image_paths = paths
        self.type = type

    def __getitem__(self, index):
        data, label = self.image_paths[index, :-1], self.image_paths[index, -1]
        if self.type == 'train':
            """if train dataset, then apply data enhancement"""
            if random.random() < 0.5:
                data = np.flip(data, axis=1).copy()
                label = label[::-1].copy()
        else:
            pass
        return data, label

    def __len__(self):
        return self.image_paths.shape[0]
    
def load_data(batch_size=64, split_rate=0.8, test_mode=False):
    print(f'Batch size: {batch_size}, split rate: {split_rate}')
    cols2keep=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'VV', 'VH', 'label']
    if test_mode:
        cols2keep.insert(-1, 'lat_lon')
    # samples for train & samples for valid
    samples4train, samples4valid = get_all_files_in_samples(".\\samples", split_rate=split_rate)
    print('Processing samples files...')
    samples4train, samples4valid = [proc_bands_value(file, cols2keep=cols2keep) for file in tqdm(samples4train)], \
                                   [proc_bands_value(file, cols2keep=cols2keep) for file in tqdm(samples4valid)]
    array_list_train, array_list_valid = [ts.values.transpose(1, 0) for ts in samples4train if ts is not None and ts.shape == (60, len(cols2keep))], \
                                         [ts.values.transpose(1, 0) for ts in samples4valid if ts is not None and ts.shape == (60, len(cols2keep))]
    train, test = np.stack(array_list_train, axis=0), \
                  np.stack(array_list_valid, axis=0)
                  
    # samples distribution adjustment
    mask_train, mask_valid = (train[:, -1, :] != 4).all(axis=1), (test[:, -1, :] != 4).all(axis=1)
    train, test = train[mask_train],  test[mask_valid] # remove background type
    train[:, -1, :][train[:, -1, :] == 5], test[:, -1, :][test[:, -1, :] == 5] = 4, 4   # herbicide -> index 4
    train[:, -1, :][train[:, -1, :] == 6], test[:, -1, :][test[:, -1, :] == 6] = 5, 5   # other vegetation -> index 5
    train[:, -1, :], test[:, -1, :] = train[:, -1, :] - 1, test[:, -1, :] - 1 # index minus 1
    
    # Load data
    print(f'train shape: {train.shape}, test shape: {test.shape}')
    train[:, :-1, :], test[:, :-1, :] = standardization(train[:, :-1, :]), \
                                        standardization(test[:, :-1, :])
    # static lcc
    if __name__ == '__main__':
        lc = {
            0: 'SA',
            1: 'TF',
            2: 'OW',
            3: 'HL',
        }
        all = np.concatenate([train, test], axis=0)
        _, _ , changetypes = FilteringSeries(all[:, -1, :].reshape(all.shape[0], -1), method='Majority', window_size=5)
        lcc_counter = Counter([tuple(i.astype(int)) for i in changetypes])
        total = sum(lcc_counter.values())
        for key, value in lcc_counter.most_common():
            key_str = '→'.join([lc[i] for i in key])
            percent = (value / total) * 100
            print(f'{key_str} : {value} ({percent:.2f}%)')
            
    # train_ds, test_ds = MaskDataset(paths=train, type='train'), \
    #                     MaskDataset(paths=test, type='test')
    # train_dl, test_dl = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True), \
    #                     data.DataLoader(dataset=test_ds, batch_size=batch_size)
    # return train_dl, test_dl
    return train, test

def make_dataloader(dataset, type, is_shuffle, batch_size=64):
    ds = MaskDataset(paths=dataset, type=type)
    return data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=is_shuffle)
def random_permutation(tralid_ds, n_split=5, split_rate=0.8, batch_size=64):
    # 每次迭代返回 (train_indices, test_indices) 元组
    n_samples, test_size = tralid_ds.shape[0], 1 - split_rate
    n_test = int(n_samples * test_size)
    print(n_samples, n_test)
    for _ in range(n_split):
        # permutation indices
        indices = np.random.permutation(n_samples)
        train_indices, valid_indices = indices[n_test:], indices[:n_test]
        train_dl, valid_dl = make_dataloader(tralid_ds[train_indices], type='train', is_shuffle=True, batch_size=batch_size), \
                             make_dataloader(tralid_ds[valid_indices], type='test', is_shuffle=False, batch_size=batch_size)   
        yield train_dl, valid_dl
        
if __name__ == '__main__':
    # Batch × Channel × Length
    # tralid_dl, test_dl = load_data(batch_size=64, split_rate=0.8)
    tralid, test = load_data(batch_size=64, split_rate=0.8)
    print(f'train shape: {tralid.shape}, test shape: {test.shape}')
    # random permutation cross validation
    for train_dl, valid_dl in random_permutation(tralid, n_split=5, split_rate=0.8, batch_size=64):
        pass
    
    