# coding: utf-8
'''
投影类
@author: Administrator
'''
# from mpl_toolkits.basemap.pyproj import Proj  # , transform
from pyproj import Proj
import numpy as np
from dp_2d import fill_2d

class prj_core():

    def __init__(self, projstr, res, row, col):
        self.proj4str = projstr
        self.RES = res
        self.ROW = row
        self.COL = col

    def lonslats2ij(self, lons, lats):  # 投影 lons,lats -> i,j
        '''
        经纬度转行列号
        参数是2维(或1维)数组
        返回值是1维数组
        '''
        # 转成1维，因为proj只接收1维参数
        pfunc = Proj(self.proj4str)
        lons = np.array(lons).reshape((-1))
        lats = np.array(lats).reshape((-1))
        # 通过平面坐标系计算投影后的行和列
        x, y = pfunc(lons, lats)  # proj只接收1维参数!!
        i = self.__y2i(y)
        j = self.__x2j(x)
        return i, j

    def __y2i(self, y):  # y 转行号
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        return np.floor((self.ROW / 2 * self.RES - y) / self.RES).astype(int)

    def __x2j(self, x):  # x 转列号
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        return np.floor((self.COL / 2 * self.RES + x) / self.RES).astype(int)

    def ij2lonlat(self):
        '''
        生成投影各格点的经纬度
        '''
        pfunc = Proj(self.proj4str)

        # 制作一个二维的矩阵
        i, j = np.mgrid[0:self.ROW:1, 0:self.COL:1]
        y = self.__i2y(i)
        x = self.__j2x(j)

        # 把二维的x,y 降维
        x = np.array(x).reshape((-1))  # 转成1维，因为proj只接收1维参数
        y = np.array(y).reshape((-1))
        lon, lat = pfunc(x, y, inverse=True)

        return lon, lat

    def __i2y(self, i):
        if isinstance(i, (list, tuple)):
            i = np.array(i)

        y = self.ROW / 2 * self.RES - self.RES * i
        return y

    def __j2x(self, j):
        if isinstance(j, (list, tuple)):
            j = np.array(j)
        x = self.RES * j - self.COL / 2 * self.RES
        return  x

class prj_gll():
    '''
    等经纬度区域类
    '''
    def __init__(self, nlat=90., slat=-90., wlon=-180., elon=180., resLat=None, resLon=None, rowMax=None, colMax=None):
        '''
        nlat, slat, wlon, elon: 北纬, 南纬, 西经, 东经
        resLat: 纬度分辨率（度）
        resLon: 经度分辨率（度）
        '''
        self.nlat = float(nlat)  # 北纬
        self.slat = float(slat)  # 南纬
        self.wlon = float(wlon)  # 西经
        self.elon = float(elon)  # 东经

        if resLat is None and rowMax is None:
            raise ValueError("resLat and rowMax must set one")

        if resLon is None and colMax is None:
            raise ValueError("resLon and colMax must set one")

        if resLat is None:
            self.rowMax = int(rowMax)
            self.resLat = (self.nlat - self.slat) / self.rowMax
        else:
            self.resLat = float(resLat)
            self.rowMax = int(round((self.nlat - self.slat) / self.resLat))  # 最大行数

        if resLon is None:
            self.colMax = int(colMax)
            self.resLon = (self.elon - self.wlon) / self.colMax
        else:
            self.resLon = float(resLon)
            self.colMax = int(round((self.elon - self.wlon) / self.resLon))  # 最大列数

    def generateLatsLons(self):
        lats, lons = np.mgrid[
            self.nlat - self.resLat / 2. : self.slat + self.resLat * 0.1 :-self.resLat,
            self.wlon + self.resLon / 2. : self.elon - self.resLon * 0.1 : self.resLon]
        return lats, lons

    def lonslats2ij(self, lons, lats):
        j = self.lons2j(lons)
        i = self.lats2i(lats)
        return i, j

    def lons2j(self, lons):
        '''
        lons: 输入经度
        ret: 返回 输入经度在等经纬度网格上的列号，以左上角为起点0,0
        '''
        if isinstance(lons, (list, tuple)):
            lons = np.array(lons)
        if isinstance(lons, np.ndarray):
            idx = np.isclose(lons, 180.)
            lons[idx] = -180.
        return np.floor((lons - self.wlon) / self.resLon).astype(int)  # 列号

    def lats2i(self, lats):
        '''
        lats: 输入纬度
        ret: 返回 输入纬度在等经纬度网格上的行号，以左上角为起点0,0
        '''
        if isinstance(lats, (list, tuple)):
            lats = np.array(lats)
        return np.floor((self.nlat - lats) / self.resLat).astype(int)  # 行号

def fill_points_2d(array2d, invalidValue=0):
    '''
    2维矩阵无效值补点
    array2d  2维矩阵
    invalidValue  无效值
    '''
    # 用右方的有效点补点
    mask = np.isclose(array2d, invalidValue)
    fill_2d(array2d, mask, 'r')

    # 用左方的有效点补点
    mask = np.isclose(array2d, invalidValue)
    fill_2d(array2d, mask, 'l')

    # 用上方的有效点补点
    mask = np.isclose(array2d, invalidValue)
    fill_2d(array2d, mask, 'u')
        
    # 用下方的有效点补点
    mask = np.isclose(array2d, invalidValue)
    fill_2d(array2d, mask, 'd')


