# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019

@author: Yue
"""
#%%
import os
try:
    from osgeo import gdal
except:
    gdal = None
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pyarrow.feather as feather
from shapely import wkt

# lgend = ['Restrict', 'R1', 'R2', 'RC', 'C', 'CBD', 'IH',
#          'I1', 'I2', 'A', 'E', 'G', 'U']
# color = ['black', 'deeppink', 'lightpink', 'darkviolet',
#          'royalblue', 'gold', 'lightseagreen', 'orange',
#          'orangered', 'yellowgreen', 'forestgreen',
#          'springgreen', 'lightgray', 'white']

# def read_tif(filename, resample=1):
#     ds = gdal.Open(filename)
#     para = {'proj': ds.GetProjection(), 'tran': ds.GetGeoTransform(),
#             'x': ds.RasterXSize, 'y': ds.RasterYSize}
#     if resample is None:
#         img = ds.ReadAsArray(0, 0, para['x'], para['y'])
#     else:
#         try:
#             x, y = resample
#             rx, ry = para['x'] * 1. / x, para['y'] * 1. / y
#         except:
#             rx, ry = resample, resample
#             x, y = int(para['x'] / rx), int(para['y'] / ry)
#         t = list(para['tran'])
#         t[1], t[5] = t[1] * rx, t[5] * ry
#         para['tran'] = t
#         para['x'], para['y'] = x, y
#         img = ds.ReadAsArray(0, 0, buf_xsize=para['x'], buf_ysize=para['y'])
#     para['img'] = img
#     return para

# def write_tif(filename, img, para, dtype=gdal.GDT_Byte):
#     ds = gdal.GetDriverByName('GTiff').Create(filename, para['x'], para['y'], 1, dtype)
#     ds.SetProjection(para['proj'])
#     ds.SetGeoTransform(para['tran'])
#     ds.GetRasterBand(1).WriteArray(img)
#     ds = None
#     print(f'Saved: {filename}')

def read_feather(file_path):
    # 读取feather文件
    df = pd.read_feather(file_path)

    # 将WKT格式的几何列(geometry)转换回地理信息
    df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x))

    # 将pandas的DataFrame对象转换为geopandas的GeoDataFrame对象
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # 设置坐标参考系统（CRS）
    # 请根据实际情况替换为您的坐标参考系统
    gdf.set_crs("EPSG:28413", inplace=True)

    return gdf

# 将新方法添加到geopandas模块
gpd.read_feather = read_feather

def read_raster(filename, driver='TIF', resample=1, mask=None):
    if driver == 'TIF':
        raster = read_tif(filename, resample)
    elif driver == 'NPY':
        raster = {'img': np.load(filename)}
        # TODO: resample
    elif driver == 'MAT':
        # TODO: add mat loader
        print(f'MAT will be supported in the future.')
        return
    else:
        print(f'No driver {driver} is founded.')
        return
    raster['mask'] = mask
    return raster

def write_raster(filename, img, para, driver='TIF'):
    if driver == 'TIF':
        write_tif(filename, img, para)

class LUloader:
    '''
    path
    z_LU: 0 must be Restrict or just skip and begin from 1
    isProb
    VorR
    raw
    var
    '''
    def __init__(self, LUs, asProb, LUType, noR=False, path='.', cmap=None, **kwargs):
        self.path = path
        self.z_LU = len(LUs)
        if not noR:
           self.z_LU -= 1
        self.legend = LUs
        self.cmap = cmap
        self.isProb = asProb
        self.VorR = LUType
        self.noR = noR
        self.load(**kwargs)
        # try:
        #     self.load(**kwargs)
        # except:
        #     print('Load Error.')
        #     pass
    
    def load(self, LUname, driver, **kwargs):
        if self.VorR == 'R':
            self.load_raster(LUname, driver=driver, **kwargs)
        elif self.VorR == 'V':
            self.load_vector(LUname, driver=driver, **kwargs)


    def load_vector(self, LUname, LUcol, LID=None, driver='shapefile', encoding='utf-8', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading vector landuse: {LUname}, LUCol: {LUcol}, Driver: {driver}/{encoding}')
        LUname = os.path.join(self.path, LUname)
        # read the raw data
        if driver == 'feather':
            vector = gpd.read_feather(f'{LUname}.feather')
        else:
            vector = gpd.read_file(LUname, driver=driver, encoding=encoding)
        if LID is not None:
            vector = vector.set_index(LID)
        # make the variable data
        _LU = np.array(vector[LUcol], dtype=np.int8)
        if self.noR:
            _LU += 1
        _R = _LU == 0
        if self.isProb:
            _LU = pd.DataFrame(self.prob_LU(_LU), index=vector.index, columns=pd.MultiIndex.from_product([["LU"], [i + 1 for i in range(self.z_LU)]]))
            _LU['R'] = _R
            gdf = gpd.GeoDataFrame(_LU, geometry=vector.geometry, crs=vector.crs)
        else:
            gdf = gpd.GeoDataFrame({'LU': _LU, 'R': _R}, geometry=vector.geometry, crs=vector.crs)
        self.raw = vector
        self.var = gdf
        print(f"Loaded vector: {LUname}[{driver}/{encoding}: {len(self.raw)}]")

    def load_raster(self, LUname, mask=None, resample=1, driver='TIF', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading raster landuse: {LUname}, MaskDN: {mask}, Driver: {driver}/{resample}x')
        LUname = os.path.join(self.path, LUname)
        raster = read_raster(LUname, driver=driver, resample=resample, mask=mask)
        _LU = np.array(raster['img'], dtype=np.int8)
        _R = _LU == 0
        _M = _LU == mask
        if mask is not None:
            _LU[_M] = 0
        if self.isProb:
            _LU = self.prob_LU(_LU)
        self.raw = raster
        self.var = {'LU': _LU, 'R': _R, 'M': _M}
        print(f"Loaded raster: {LUname}[{driver}/{resample}x: {self.var['LU'].shape}]")

    def save_vector(self, name, optLU=None, driver='shapefile', encoding='utf-8'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU)
        name = os.path.join(self.path, name)
        self.var['opt'] = optLU
        if driver == 'feather':
            self.var.to_feather(f'name.feather')
        self.var.to_file(name, driver=driver, encoding=encoding)

    def save_raster(self, name, optLU=None, driver='TIF'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU, self.raw['mask'])
        name = os.path.join(self.path, name)
        write_raster(name, optLU, self.raw, driver=driver)

    def save(self, name, optLU=None, **kwargs):
        if self.VorR == 'V':
            self.save_vector(name, optLU, **kwargs)
        if self.VorR == 'R':
            self.save_raster(name, optLU, **kwargs)

    def prob_LU(self, LU=None):
        if LU is None:
            LU = self.var['LU']
            if self.isProb == True:
                return np.array(LU, dtype=np.float32)
        LU = np.array(LU, dtype=np.int8)
        LU = np.array([LU == (c + 1) for c in range(self.z_LU)], dtype=np.float32)
        return np.moveaxis(LU, 0, -1)
    
    def cate_LU(self, pLU=None):
        if pLU is None:
            pLU = self.var['LU']
            if self.isProb == False:
                return np.array(pLU, dtype=np.int8)
        LU = np.argmax(np.array(pLU, dtype=np.float32), axis=-1) + 1
        return np.array(LU, dtype=np.int8)

    def to_LU(self, LU=None, mask=None):
        if LU is None:
            LU = self.cate_LU(LU)
        _R = np.array(self.var['R'], dtype=bool)
        LU[_R] = 0
        if self.VorR == 'R':
            LU[self.var['M']] = self.z_LU + 1 if mask is None else mask
        return LU

    def plot_pLU(self, pLU=None, show=True, save_name=None, bg=None, bgarg={}, legendOrient='horizontal', path='plot'):
        LU = self.cate_LU(pLU)
        self.plot_LU(LU=LU, show=show, save_name=save_name, bg=bg, bgarg=bgarg, legendOrient=legendOrient, path=path)

    def plot_LU(self, LU=None, show=True, save_name=None, bg=None, bgarg={}, legendOrient='horizontal', path='plot'):
        LU = self.to_LU(LU)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.axis('off')
        if bg is not None:
            bg.plot(ax=ax, **bgarg)
        M = self.z_LU + 1
        try:
            cmap = ListedColormap(self.cmap)
        except:
            # print('Use default colormap.')
            cmap = None
        # if legend is None:
        #     print('Use default legend.')
        #     legend = lgend
        if self.VorR == 'V':
            try:
                self.var.plot(LU, ax=ax, cmap=cmap, vmin=0, vmax=M, legend=False) # TODO: legend
            except:
                self.raw.plot(LU, ax=ax, cmap=cmap, vmin=0, vmax=M, legend=False)
        elif self.VorR == 'R':
            ax.imshow(LU, cmap=cmap, vmin=0, vmax=M, interpolation='none')
            # TODO:legend
            plt.colorbar(ax=ax, ticks=[0,0.5, 1], orientation=legendOrient).set_ticklabels(legend)
        fig.tight_layout()
        if save_name is not None:
            path = os.path.join(self.path, path)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, save_name))
        if show:
            fig.show()

# %%
