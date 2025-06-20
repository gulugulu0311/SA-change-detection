import os
import sys
import math
import torch
import pandas as pd


from tqdm import tqdm, trange
from osgeo import gdal
from collections import Counter

from utils import *
from config import Configs
from models.TSSCD import TSSCD_FCN, TSSCD_Unet, TSSCD_TransEncoder

def scaleMinMax(x):
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))

def generate_date_labels():
    dates = [f"{year}-{month:02d}" 
            for year in range(2020, 2025) 
            for month in range(1, 13)]
    return dates[:60]

events = ['Invasion', 'Mowing 1st', 'Mowing 2nd', 'Waterlogging', 'Herbicide control', 'Recurring', 'WL_fast', 'No change']
vec_path = '.\\TimeSeriesImages\\SA_region\\SA_extend_0619_SD_JS.shp'

def main(province, model,
         mode=False):
    folder = f'.\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{province}'
    # sort imgs by date
    sorted_date = sorted(os.listdir(folder), \
                        key=lambda x: pd.to_datetime(x.split('.')[0], format='%Y_%m'))
    
    sorted_date = sorted_date[5:]
    print(f'date range: {sorted_date[0]} ~ {sorted_date[-1]}')
    imgs_list = [os.path.join(folder, date) for date in sorted_date]

    # check imgs consistence...
    check_img_size = list()
    for img in imgs_list:
        img_data = gdal.Open(img)
        raster_x_size, raster_y_size = img_data.RasterXSize, img_data.RasterYSize
        band_count = img_data.RasterCount
        check_img_size.append((raster_y_size, raster_x_size, band_count))
        img_data = None
    print('\nimgs shape (y, x):', Counter(check_img_size))

    # create mask image
    ref_img = gdal.Open(imgs_list[25]) # randomly selected...
    raster_x_size, raster_y_size = ref_img.RasterXSize, ref_img.RasterYSize
    band_count = ref_img.RasterCount
    geo_info = {
        "geo_transform": ref_img.GetGeoTransform(),
        "projection": ref_img.GetProjection()
    }
    mask_img_path = f'.\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{province}_mask\\mask.tif'
    
    if not os.path.exists(mask_img_path):
        print('\nStart to create mask image...')
        vec2mask(ref_img, vec_path, mask_img_path)
        return
    else:
        mask_data = gdal.Open(mask_img_path)
        
        mask_data = mask_data.ReadAsArray() # mask_data
        print(f'\nSuccessfully load mask image. \npath: {mask_img_path}')
        
    print(f'\nref_img: Y size: {raster_y_size}, X size: {raster_x_size},  bands: {band_count}')
    print(f'mask_img shape (y, x): {mask_data.shape}')
    ref_img = None

    batch_size, valid_blocks = 256, list()   # classification block size

    # define classification area
    total_iter = (raster_y_size // batch_size) * (raster_x_size // batch_size)
    with trange(total_iter) as loop:
        for k in loop:
            i, j = (k // (raster_x_size // batch_size)) * batch_size, \
                (k % (raster_x_size // batch_size)) * batch_size
            current_batch_height, current_batch_width = min(batch_size, raster_y_size - i), \
                                                        min(batch_size, raster_x_size - j)
            block_data = mask_data[i:i + current_batch_height, j:j + current_batch_width]
            valid_area = block_data == 1
            if valid_area.sum() == 0:
                continue
            else:
                valid_blocks.append((k, valid_area))
                continue

    print(f'valid block: {len(valid_blocks)}')
    del mask_data

    # preprocess for efficient classification
    data4cls = f'.\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{province}_data4cls'
    if os.path.exists(data4cls):
        block_idx = [int(file.split('.')[0].split('_')[-1]) for file in os.listdir(data4cls)]
        if (sorted(block_idx) == sorted([i[0] for i in valid_blocks])):
            print(f'\nSuccessfully load npz files.')
        else:
            print(f'\nNPZ files are not consistent with valid block.')
            with tqdm(valid_blocks) as loop:
                for k, valid_area in loop:
                    i, j = (k // (raster_x_size // batch_size)) * batch_size, \
                        (k % (raster_x_size // batch_size)) * batch_size
                    current_batch_height, current_batch_width = min(batch_size, raster_y_size - i), \
                                                                min(batch_size, raster_x_size - j)
                    input_ts_data = np.zeros((current_batch_height, current_batch_width, 12, 60), dtype=np.float32)
                    datasets = list()
                    for date_idx, file_path in enumerate(imgs_list):
                        dataset = gdal.Open(file_path)
                        if dataset is None:
                            raise Exception(f"Failed to open {file_path}")
                        datasets.append(dataset)
                        try:
                            input_ts_data[:, :, :, date_idx] = dataset.ReadAsArray(j, i, current_batch_width, current_batch_height).transpose(1, 2, 0)
                        except Exception as e:
                            raise Exception(f"Error reading {file_path}: {e}")
                    for dataset in datasets:
                        dataset = None
                    np.savez_compressed(os.path.join(data4cls, f'block_{k}.npz'), data=input_ts_data)
            return
        
    # classification for specific area
    if mode:
        date_labels = generate_date_labels()
        regional_output_folder = f'.\\display\\{mode["region"]}'
        if not os.path.exists(regional_output_folder):
            print(f'Start to create folder: {regional_output_folder}')
            os.makedirs(regional_output_folder)
            
        centroid, radius = mode['centroid'], mode['radius']
        print(f'\nStart to classify the area around {centroid} with radius {radius/1000} (km)...')

        delta_lat = radius / 111319.9
        delta_lon = radius / (111319.9 * math.cos(math.radians(centroid[1])))
        
        min_lon, max_lon = centroid[0] - delta_lon, centroid[0] + delta_lon
        min_lat, max_lat = centroid[1] - delta_lat, centroid[1] + delta_lat
        
        min_j, max_j = float('inf'), -float('inf')
        min_i, max_i = float('inf'), -float('inf')
        
        block_data_dict = dict()
        for k, valid_area in valid_blocks:
            i, j = (k // (raster_x_size // batch_size)) * batch_size, \
                   (k % (raster_x_size // batch_size)) * batch_size
            current_batch_height, current_batch_width = min(batch_size, raster_y_size - i), \
                                                        min(batch_size, raster_x_size - j)
            # is overlap with the specific region?
            block_ul_lon, block_ul_lat = geo_info['geo_transform'][0] + j * geo_info['geo_transform'][1], geo_info['geo_transform'][3] + i * geo_info['geo_transform'][5]
            block_lr_lon = block_ul_lon + current_batch_width * geo_info['geo_transform'][1]
            block_lr_lat = block_ul_lat + current_batch_height * geo_info['geo_transform'][5]
            
            lon_overlap = (block_ul_lon < max_lon) and (block_lr_lon > min_lon)
            lat_overlap = (block_lr_lat < max_lat) and (block_ul_lat > min_lat)
            
            # specific region contained blocks
            if lon_overlap and lat_overlap:     
                min_j, max_j = min(min_j, j), max(max_j, j + current_batch_width)
                min_i, max_i = min(min_i, i), max(max_i, i + current_batch_height)
                
                npz_file = np.load(os.path.join(data4cls, f'block_{k}.npz'))['data']
                model_input_data = standardization(npz_file[valid_area])    # standardization (Batch × Channel × Length)
                model_input_data = torch.Tensor(model_input_data).to(device)
                
                batch_sub_size = int(batch_size / 4) # split batch for inference...

                all_preds, nrgb_img = list(), npz_file[:, :, list([6, 2, 1, 0]), :] / 10000
                for idx in range(0, model_input_data.shape[0], batch_sub_size):  
                    batch_data = model_input_data[idx:idx + batch_sub_size]
                    with torch.no_grad():
                        if isinstance(model, TSSCD_TransEncoder):
                            batch_preds = model(batch_data.permute(0, 2, 1)).permute(0, 2, 1)
                        else:
                            batch_preds = model(batch_data)
                    all_preds.append(batch_preds)
                    
                preds = torch.argmax(input=torch.cat(all_preds, dim=0), dim=1).cpu().numpy()
                preds = MajorityFilter(preds, kernel_size=5)  # temporal filter
                
                output_data = np.full((current_batch_height, current_batch_width, 60), 99, dtype=np.int8)
                output_data[valid_area] = preds
                # Extract Events from Landcover changes
                if mode['output_cls']:
                    block_data_dict[(i, j)] = (output_data, nrgb_img)
                else:
                    lcc, _ = generate_event_map(output_data, valid_area, events, max_lc_change=5)
                    block_data_dict[(i, j)] = (lcc, nrgb_img)
                print(f'block {k} finished.')
                
        merged_height, merged_width = int(max_i - min_i), int(max_j - min_j)
        # Merged output
        merged_output = np.full((merged_height, merged_width, 60 if mode['output_cls'] else len(events)), 99, dtype=np.int32)
        base_map = np.full((merged_height, merged_width, 4, 60), np.nan, dtype=np.float64)
        
        for (i, j), (event, nrgb) in block_data_dict.items():
            y_start = i - min_i
            y_end = y_start + event.shape[0]
            x_start = j - min_j
            x_end = x_start + event.shape[1]
            
            merged_output[y_start:y_end, x_start:x_end, :] = event
            base_map[y_start:y_end, x_start:x_end, :, :] = nrgb
        
        new_geo_transform = (
            geo_info['geo_transform'][0] + min_j * geo_info['geo_transform'][1],
            geo_info['geo_transform'][1], geo_info['geo_transform'][2],
            geo_info['geo_transform'][3] + min_i * geo_info['geo_transform'][5], 
            geo_info['geo_transform'][4], geo_info['geo_transform'][5]
        )
        # ===================== Output cls ======================
        if mode['output_cls']:
            print('output: classification map.')
            for date_idx in range(60):
                if date_labels[date_idx] not in mode['dates']: continue
                
                output_path = regional_output_folder + f'\\{date_labels[date_idx]}_cls.tif'
                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(output_path, merged_width, merged_height, 1, gdal.GDT_Int32, options=['COMPRESS=LZW'])
                ds.SetGeoTransform(new_geo_transform)
                ds.SetProjection(geo_info['projection'])
                ds.GetRasterBand(1).WriteArray(merged_output[:, :, date_idx])
                ds.GetRasterBand(1).SetNoDataValue(99)
                ds.FlushCache()
                ds = None
                
                # nrgb_map
                output_path = regional_output_folder + f'\\{date_labels[date_idx]}_nrgb.tif'
                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(output_path, merged_width, merged_height, 4, gdal.GDT_Float64, options=['COMPRESS=LZW'])
                ds.SetGeoTransform(new_geo_transform)
                ds.SetProjection(geo_info['projection'])
                
                for nrgb_idx in range(4):
                    ds.GetRasterBand(nrgb_idx + 1).WriteArray(base_map[:, :, nrgb_idx, date_idx])
                ds.FlushCache()
                ds = None
                
        else:
            print('output: event map.')
            for idx, event in enumerate(events):
                if mode['event2plot'] is not None and event != mode['event2plot']:
                    continue
                output_path = regional_output_folder + f'\\{event}.tif'
                event_map = merged_output[:, :, idx]
                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(output_path, merged_width, merged_height, 1, gdal.GDT_Int32, options=['COMPRESS=LZW'])
                ds.SetGeoTransform(new_geo_transform)
                ds.SetProjection(geo_info['projection'])
                ds.GetRasterBand(1).WriteArray(event_map)
                ds.GetRasterBand(1).SetNoDataValue(99)
                ds.FlushCache()
                ds = None
        return
# ====== Plot =======
        # x = np.linspace(geo_info['geo_transform'][0],
        #                 geo_info['geo_transform'][0] + geo_info['geo_transform'][1] * merged_width,
        #                 merged_width)
        # y = np.linspace(geo_info['geo_transform'][3],
        #                 geo_info['geo_transform'][3] + geo_info['geo_transform'][5] * merged_height,
        #                 merged_height)

        # from matplotlib import pyplot as plt
        # from matplotlib.ticker import FuncFormatter, MultipleLocator
        
        # plt.rcParams.update({
        #     'font.family': 'Arial',
        #     'font.size': 6
        # })
        
        # event2plot = mode['event2plot']
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        # im = None
        # event_idx = events.index(event2plot)
        # event_map = merged_output[:, :, event_idx]
        # event_map = np.where(event_map == 99, np.nan, event_map)
        
        # ax.imshow(base_map, cmap='gray',
        #                     vmin=0.0, vmax=0.4,
        #                     extent=[x.min(), x.max(), y.max(), y.min()],)
        # im = ax.imshow(event_map, cmap='viridis', vmin=0, vmax=59,
        #                         interpolation='none',
        #                         extent=[x.min(), x.max(), y.max(), y.min()])
        
        # ax.tick_params(axis='y', labelrotation=90)
        # for label in ax.get_yticklabels():
        #     label.set_verticalalignment('center')
        #     label.set_horizontalalignment('center')
            
        # ax.set_title(event2plot) # Set title
        # ax.set_xlabel(''), ax.set_ylabel('')
        
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: deg2dms(x, pos, mode['p2keep'], axis='lon')))
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: deg2dms(x, pos, mode['p2keep'], axis='lat')))
        
        # interval_degree = mode['interval_degree']
        # ax.xaxis.set_major_locator(MultipleLocator(interval_degree))
        # ax.yaxis.set_major_locator(MultipleLocator(interval_degree))
        
        # ax.tick_params(axis='both', labelsize=4, pad=3, width=0.5, length=1.5)
        
        # if mode['is_show_cbar']:
        #     print('is_show_cbar:', mode['is_show_cbar'])

        #     fig.subplots_adjust(bottom=0.3) # cbar and ax location adjust
        #     cax = fig.add_axes([0.15, 0.15, 0.7, 0.03])
            
        #     cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        #     date_labels = generate_date_labels()
        #     tick_positions = np.arange(5, 60, 12)
        #     cbar.set_ticks(tick_positions + 0.5)
        #     cbar.set_ticklabels([date_labels[i] for i in tick_positions])
        # plt.show()
        # return

    # classification for all
    with tqdm(valid_blocks) as loop:
        for k, valid_area in loop:
            i, j = (k // (raster_x_size // batch_size)) * batch_size, \
                (k % (raster_x_size // batch_size)) * batch_size
            current_batch_height, current_batch_width = min(batch_size, raster_y_size - i), \
                                                        min(batch_size, raster_x_size - j)

            input_ts_data = np.load(os.path.join(data4cls, f'block_{k}.npz'))['data']   # (H, W, Channel, Length)
            model_input_data = input_ts_data[valid_area]    # Batch × Channel × Length
            
            model_input_data = standardization(model_input_data)    # standardization
            model_input_data = torch.Tensor(model_input_data).to(device)
            
            batch_sub_size = int(batch_size / 4) # split batch for inference...
            all_preds = list()
            for b in range(0, model_input_data.shape[0], batch_sub_size):  
                batch_data = model_input_data[b:b + batch_sub_size]
                with torch.no_grad():
                    if isinstance(model, TSSCD_TransEncoder):
                        batch_preds = model(batch_data.permute(0, 2, 1)).permute(0, 2, 1)
                    else:
                        batch_preds = model(batch_data)
                all_preds.append(batch_preds)
                
            preds = torch.cat(all_preds, dim=0)
            preds = torch.argmax(input=preds, dim=1).cpu().numpy()
            preds = MajorityFilter(preds, kernel_size=3)  # temporal filter
            
            output_data = np.full((current_batch_height, current_batch_width, 60), 99, dtype=np.int8)
            output_data[valid_area] = preds
            # spatial filter
            # output_data = spatial_mode_filter(output_data, size=3)
            
            lcc, _ = generate_event_map(output_data, valid_area, events, max_lc_change=5)  # lcc: (current_batch_height, current_batch_width, len(events))
            
            # save output - land cover change result
            driver = gdal.GetDriverByName("GTiff")
            output_dataset = driver.Create(f'.\\TimeSeriesImages\\SA_blocks_clipped&mosaic\\{province}_cls\\block_{k}.tif', \
                                    current_batch_width, current_batch_height, len(events), gdal.GDT_Int32, options=["COMPRESS=LZW"])
            block_geo_transform = (
                geo_info['geo_transform'][0] + j * geo_info['geo_transform'][1],
                geo_info['geo_transform'][1],  
                geo_info['geo_transform'][2],
                geo_info['geo_transform'][3] + i * geo_info['geo_transform'][5],
                geo_info['geo_transform'][4], 
                geo_info['geo_transform'][5]
            )
            
            output_dataset.SetGeoTransform(block_geo_transform)
            output_dataset.SetProjection(geo_info['projection'])
            for band_idx in range(len(events)):
                band = output_dataset.GetRasterBand(band_idx + 1)
                band.WriteArray(lcc[:, :, band_idx])
                band.SetNoDataValue(99)
                
            output_dataset.FlushCache()
            output_dataset = None
    
if __name__ == '__main__':
    # load model
    device = device_on()
    configs = Configs()
    model_instances = {
        'TSSCD_FCN': TSSCD_FCN(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_Unet': TSSCD_Unet(configs.input_channels, configs.classes, configs.model_hidden),
        'TSSCD_TransEncoder': TSSCD_TransEncoder(configs.input_channels, configs.classes, configs.Transformer_dmodel)
    }
    model_name = 'TSSCD_Unet'
    model = model_instances[model_name]
    model = model.to(device)
    model_state_dict = torch.load(os.path.join(f'models\\model_data\\{model_name}', '1001.pth'), map_location='cuda', weights_only=True)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # mode = {
    #     # 'centroid': [121.25212, 30.37126],    # ZJ
    #     # 'centroid': [121.9785, 31.196],   # SH 9duansha
    #     'centroid': [120.482807, 33.836398],   # JS sheyangRiver
    #     # 'centroid': [119.204, 37.7554],    # SD YRD
    #     # 'centroid': [120.6456, 33.5228],    # JS YC
    #     'radius': 1000,
    #     # 'event2plot': 'Mowing 1st',
    #     'region': 'JS_SheyangRiver',
    #     'output_cls': True,
    #     'dates': ['2021-09', '2022-10', '2023-10', '2024-07', '2024-10']
    # }
    # main('JS', model, mode)
    
    # province_list = [ 'JS', 'SD', 'FJ', 'ZJ', 'SH', 'GDGX']
    province_list = ['JS', 'SD']
    for province in province_list:
        main(province, model, mode=False)