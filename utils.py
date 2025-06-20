import os
import re
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from osgeo import gdal, ogr, osr

from scipy.ndimage import generic_filter
from scipy.stats import mode
class EarlyStopping:
    '''
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    '''

    def __init__(self, patience=8, min_delta=0):
        '''
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def device_on():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Uisng {device} device')
    return device
def vec2mask(ref_image, vector_path, output_mask_path):
    # open reference image
    geotrans, proj = ref_image.GetGeoTransform(), ref_image.GetProjection()
    cols, row = ref_image.RasterXSize, ref_image.RasterYSize
    # create mask image
    driver = gdal.GetDriverByName('GTiff')
    mask_ds = driver.Create(output_mask_path, cols, row, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    mask_ds.SetGeoTransform(geotrans), mask_ds.SetProjection(proj)
    
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(0)
    # open vec file
    vec_src = ogr.Open(vector_path)
    layer = vec_src.GetLayer()
    
    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
    mask_ds, vec_src, layer = None, None, None
    print(f'Mask image saved to: {output_mask_path}')
def MajorityFilter(preds, kernel_size=3):
    '''
    Must be 2D array: (samples, time_steps)
    '''
    assert kernel_size % 2 == 1, 'Kernel size must be odd'
    half_kernel = kernel_size // 2
    
    # fill boudary with symmetric padding
    padded = np.pad(preds, ((0, 0), (half_kernel, half_kernel)), mode='symmetric')
    strided = np.lib.stride_tricks.sliding_window_view(padded, (1, kernel_size))
    strided = strided.squeeze(axis=2)
    
    # vectorized majority filter
    def vectorized_mode(window):
        values, counts = np.unique(window, return_counts=True)
        return values[np.argmax(counts)]
    # apply majority filter
    filtered = np.apply_along_axis(vectorized_mode, 2, strided)
    return filtered

def proc_bands_value(csv_file, bands_col='bands_value', 
                     cols2keep=['B2', 'B3', 'B4','B5', 'B6', 'B7','B8', 'B8A', 'B11', 'B12', 'VV', 'VH', 
                                'label', 'lat_lon', 'label']):
    df = pd.read_csv(csv_file)
    try:
        # deal with bands values {'B2=114, B3=513, ... '}
        df[bands_col] = df[bands_col].str.strip('{}')
        df_dicts = df[bands_col].str.split(', ').apply(
            lambda x: {k: float(v) if v.strip() != 'null' else np.nan for k, v in (item.split('=') for item in x)}
        )
        df_expanded = pd.DataFrame(df_dicts.tolist())
        df_expanded = df_expanded.ffill().bfill()  # 填充缺失值
        # Date and location
        df['date'] = pd.to_datetime(df['system:time_start']).dt.to_period('M')
        df['lat_lon'] = df.apply(lambda row: [row['latitude'], row['longitude']], axis=1)
        df_final = pd.concat([df.drop(columns=[bands_col]), df_expanded], axis=1)
    except Exception as e:
        print(f'Error processing DataFrame: {e}, ', csv_file)
        # os.remove(csv_file)
        return None
    return df_final[cols2keep]

def get_all_files_in_samples(dir, split_rate=0.8, show_tonum=False):
    '''
        Get all samples file path in root_dir.\n
        Return:
            samples4train: Train samples file path.
            samples4valid: Valid samples file path.
    '''
    samples4train, samples4valid = list(), list()
    total_samples_num = 0

    for dirpath, _, filenames in os.walk(dir):
        csv_files = [f for f in filenames if f.endswith('.csv')]
        total_samples_num += len(csv_files)

        if dirpath != dir:
            dir_name = os.path.basename(dirpath)
            print(f'{dir_name}: {len(csv_files)}')

        random.shuffle(csv_files)
        split_index = int(len(csv_files) * split_rate)
        
        samples4train.extend(
            os.path.join(dirpath, filename) 
            for filename in csv_files[:split_index]
        )
        
        if split_rate != 1:
            samples4valid.extend(
                os.path.join(dirpath, filename)
                for filename in csv_files[split_index:]
            )
    if show_tonum: print('Total Samples:', total_samples_num)
    return samples4train, samples4valid
# def minMax_standardization(data):
#     # data : (batch_size, channels, seq_length)
#     lower_bound = np.percentile(data, 2, axis=(0, 2), keepdims=True)
#     upper_bound = np.percentile(data, 98, axis=(0, 2), keepdims=True)
#     return (data - lower_bound) / (upper_bound - lower_bound + 1e-8) * 2 - 1
def standardization(data):
    # data : (batch_size, channels, seq_length)
    return data
    if type == 'z-score':
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
        return (data - mean) / (std + 1e-8)
    else:
        return data

def extract_time_series_data(lat_lon, img_path):
    lat, lon = lat_lon[1], lat_lon[0]
    data = gdal.Open(img_path)
    # get geo transform
    transform = data.GetGeoTransform()
    x, y = int((lon - transform[0]) / transform[1]), int((lat - transform[3]) / transform[5])
    pixel_values = data.ReadAsArray(x, y, 1, 1)
    del data
    return pixel_values.flatten()

def extract_change_event_from_pixel(lcc, cd):
    def dtc_flooding(lcc):
        if len(lcc) >= 3:
            for i in range(len(lcc) - 2):
                if np.array_equal(lcc[i:i+3], np.array([0, 1, 2])):
                    return True, i + 1
        return False, -1

    def dtc_recurring(lcc):
        if len(lcc) >= 3:
            for i in range(len(lcc) - 2):
                if np.array_equal(lcc[i:i+3], np.array([0, 1, 0])):
                    return True, i + 1
            return False, -1
        else:
            return False, -1
    def dtc_herbicide(lcc):
        idx = np.argmax(lcc == 3)
        if lcc[idx] == 3:
            return idx
        else:
            return -1
    def dtc_mowing_1st(lcc):
        if len(lcc) < 2:
            return 99
        consecutive_01 = (lcc[:-1] == 0) & (lcc[1:] == 1)
        idx = np.argmax(consecutive_01)
        return idx if consecutive_01[idx] else 99
    def dtc_flooding_fast(lcc):
        for i in range(len(lcc) - 1):
            if np.array_equal(lcc[i:i+2], np.array([0, 2])):
                return True, i
        return False, -1
    
    mowing_1st, mowing_2nd, recurring, no_change, flooding_fast = 99, 99, 99, 99, 99
    # invasion
    invasion = cd[0] if lcc[0] == 1 and lcc[1] == 0 else 99
    # flooding
    is_flooding, flooding_cd = dtc_flooding(lcc)
    flooding = cd[flooding_cd] if is_flooding else 99
    # recurring
    is_recurring, recurring_cd = dtc_recurring(lcc)
    recurring = cd[recurring_cd] if is_recurring else 99
    # herbicide
    is_herbicide, herbicide = dtc_herbicide(lcc), 99
    if is_herbicide != -1 and is_herbicide != 0:
        herbicide = cd[is_herbicide - 1]
    # flooding_fast
    is_flooding_fast, flooding_fast_cd = dtc_flooding_fast(lcc)
    flooding_fast = cd[flooding_fast_cd] if is_flooding_fast else 99
    
    # mowing_1st
    idx_mowing_1st = dtc_mowing_1st(lcc)
    if idx_mowing_1st != 99:
        # mowing_2nd
        mowing_1st = cd[idx_mowing_1st]
        if len(lcc) >= 4:
            mowing_2nd = dtc_mowing_1st(lcc[idx_mowing_1st + 1:])
            if mowing_2nd != 99:
                mowing_2nd += idx_mowing_1st + 1
                mowing_2nd = cd[mowing_2nd]
            
    return [invasion, mowing_1st, mowing_2nd, flooding, herbicide, recurring, flooding_fast, no_change] # 《no change》 must be last

def generate_event_map(model_preds, valid_area, events, max_lc_change=5, is_static=False):
    lccmap = list()
    static_info = None
    height, width = model_preds.shape[0], model_preds.shape[1]
    for ts in model_preds[valid_area]:
        change_points = np.where(ts[:-1]!= ts[1:])[0] + 1
        # no change
        if change_points.shape[0] == 0:
            lccmap.append(np.concatenate([np.repeat(99, len(events) - 1), np.array([ts[0]])]))  # No change type is last one 
        # too many lc changes
        elif len(change_points) > (max_lc_change - 1):
            cd = change_points[: max_lc_change - 1] # as [2, 12, 19, 30]
            change_points = np.concatenate([[0], change_points])
            lc = ts[change_points][: max_lc_change] # as [1, 0, 1, 0, 1]
            event = extract_change_event_from_pixel(lc, cd)
            lccmap.append(np.array(event))
        # lc changes < max_lc_change
        else:
            lc = ts[np.concatenate([[0], change_points])]
            event = extract_change_event_from_pixel(lc, change_points)
            lccmap.append(np.array(event))
    lccmap = np.stack(lccmap)
    lcc = np.full((height, width, len(events)), 99)
    lcc[valid_area] = lccmap
    return lcc, static_info
def extract_accuracy_from_log(file_path):
    pth = file_path.split('\\')[-1].split('.')[0]
    try:
        with open(file_path, 'r') as file:
            log_content = file.read()

        epoch_info = {}
        epoch_pattern = re.compile(r'Epoch (\d+), Train loss: ([\d.]+)')
        metric_pattern = re.compile(r'(\w+): ([\d.]+)')
        last_saved_pattern = re.compile(r'last saved epoch: (\d+)')

        for epoch_match in epoch_pattern.finditer(log_content):
            epoch_num = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2))
            metrics = {'pth': pth,
                       'train_loss': train_loss}

            start_index = epoch_match.end()
            next_epoch_start = log_content.find('Epoch', start_index)
            if next_epoch_start == -1:
                next_epoch_start = len(log_content)

            metric_text = log_content[start_index:next_epoch_start]
            for metric_match in metric_pattern.finditer(metric_text):
                metric_name = metric_match.group(1)
                metric_value = float(metric_match.group(2))
                metrics[metric_name] = metric_value
                
            last_saved_match = last_saved_pattern.search(metric_text)
            if last_saved_match:
                last_saved_epoch = int(last_saved_match.group(1))
            epoch_info[epoch_num] = metrics
        return epoch_info, last_saved_epoch, int(pth)

    except FileNotFoundError:
        print('file not found')
    except Exception as e:
        print(f'Error: {e}')

# =========================================================
# =========================================================


def CreateGeoTiff(outRaster, image, geo_transform, projection):
    '''Write GeoTiff'''
    no_bands = 0
    rows = 0
    cols = 0
    driver = gdal.GetDriverByName('GTiff')
    if len(image.shape) == 2:
        no_bands = 1
        rows, cols = image.shape
    elif len(image.shape) == 3:
        no_bands, rows, cols = image.shape

    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Float32)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)
    if no_bands == 1:
        DataSet.GetRasterBand(1).WriteArray(image)
    else:
        for i in range(no_bands):
            DataSet.GetRasterBand(i + 1).WriteArray(image[i])
    del DataSet


def Median_filtering(image, window_size=3):
    high, wide = image.shape
    img = image.copy()
    mid = (window_size - 1) // 2
    med_arry = []
    for i in range(high - window_size):
        for j in range(wide - window_size):
            for m1 in range(window_size):
                for m2 in range(window_size):
                    med_arry.append(int(image[i + m1, j + m2]))
            med_arry.sort()  # Sort window pixels
            # Assign the median value of the filter window to the pixel in the middle of the filter window
            img[i + mid, j + mid] = med_arry[(len(med_arry) + 1) // 2]
            del med_arry[:]
    return img


def block_fn(x, center_val):
    unique_elements, counts_elements = np.unique(x.ravel(), return_counts=True)

    if np.isnan(center_val):
        return np.nan
    elif center_val == 1:
        return 1.0
    else:
        return unique_elements[np.argmax(counts_elements)]


def DetectChangepoints(data):
    changepoints, changetypes = [], []
    for series in data:
        id = np.where((series[1:] - series[:-1]) != 0)[0]
        changepoints.append(id)
        changetypes.append(np.append(series[id], series[-1]))
    return changepoints, changetypes


def FilteringSeries(data, method='NoFilter', window_size=3):
    '''Temporal consistency modification'''
    if method == 'NoFilter':
        changepoints, changetypes = DetectChangepoints(data)
        return data, changepoints, changetypes
    elif method == 'Majority':
        res = MajorityFilter(data, kernel_size=window_size)
        changepoints, changetypes = DetectChangepoints(res)
        return res, changepoints, changetypes


def mat2rgb(mat):
    '''Grayscale Matrix Visualization'''
    rgblist = [[88, 184, 255], [25, 70, 31], [138, 208, 27], [222, 168, 128], [212, 67, 56], [255, 214, 156],
               [255, 222, 173], [255, 255, 255], [255, 255, 255]]
    mat = mat.astype('int8')
    return np.array(rgblist)[mat]
