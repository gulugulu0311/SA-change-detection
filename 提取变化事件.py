import os
import numpy as np
import pandas as pd

from utils import *
from osgeo import gdal
from tqdm import tqdm

# min_lon, min_lat, max_lon, max_lat
# extent, region = (119.0077, 37.8884, 119.34, 37.6268), 'YRD'
# extent, region = (121.6365, 31.707, 121.8514, 31.5948), 'ChongmingIsland'
extent, region = (121.172, 30.4043, 121.3086, 30.3608), 'HZB'
cls_img = '.\\TimeSeriesImages\\result\\classification.tif'


ds = gdal.Open(cls_img)
transform, projection = ds.GetGeoTransform(), ds.GetProjection()
# 左 上
x1, y1 = int((extent[0] - transform[0]) / transform[1]), int((extent[1] - transform[3]) / transform[5])
# 右 下
x2, y2 = int((extent[2] - transform[0]) / transform[1]), int((extent[3] - transform[3]) / transform[5])

width, height = x2 - x1, y2 - y1
region_cls = ds.ReadAsArray(x1, y1, width, height)
del ds

max_lc_change = 7 # change points number = max_lc_change - 1
data = region_cls.reshape(60, -1).transpose(1, 0)   # (-1, 60)
valid_indices = np.where(~np.all(data == 99, axis=1))[0] # valid mask

dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
dates = dates.strftime('%Y%m').to_numpy()

events = ['invasion', 'mowing_1st', 'mowing_2nd', 'flooding', 'herbicide', 'recurring', 'nochange']
lccmap = []
for ts in tqdm(data[valid_indices]):
    change_points = np.where(ts[:-1] != ts[1:])[0] + 1
    # no change
    if change_points.shape[0] == 0:
        lccmap.append(np.concatenate([np.repeat(99, len(events) - 1), np.array([ts[0]])]))
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
lcc = np.full((data.shape[0], len(events)), 99)
lcc[valid_indices] = lccmap
lcc = lcc.reshape(height, width, -1)
print(lcc.shape)
for idx, event in enumerate(events):
    output_path = f'.\\display\\{region}\\{event}.tif'
    event_map = lcc[:, :, idx]
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, width, height, 1, gdal.GDT_Int32, options=['COMPRESS=LZW'])
    new_transform = (
        transform[0] + x1 * transform[1],
        transform[1],
        transform[2],
        transform[3] + y1 * transform[5],
        transform[4],
        transform[5] 
    )
    ds.SetGeoTransform(new_transform)
    ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(event_map)
    ds.GetRasterBand(1).SetNoDataValue(99)
    ds.FlushCache()
    ds = None

