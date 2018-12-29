# coding: utf-8
'''
投影公共模块
@author: zhangtao
'''

from math import ceil

from pyproj import Proj, transform

from PB.pb_space import deg2meter
from dp_2d import fill_2d
import numpy as np


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
            self.rowMax = int(
                round((self.nlat - self.slat) / self.resLat))  # 最大行数

        if resLon is None:
            self.colMax = int(colMax)
            self.resLon = (self.elon - self.wlon) / self.colMax
        else:
            self.resLon = float(resLon)
            self.colMax = int(
                round((self.elon - self.wlon) / self.resLon))  # 最大列数

    def generateLatsLons(self):
        lats, lons = np.mgrid[
            self.nlat - self.resLat / 2.: self.slat + self.resLat * 0.1:-self.resLat,
            self.wlon + self.resLon / 2.: self.elon - self.resLon * 0.1: self.resLon]
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


class prj_core(object):
    '''
    投影公共类
    '''

    def __init__(self, projstr, res, unit="m",
                 row=None, col=None, pt_tl=None, pt_br=None,
                 **kwargs):
        '''
        [args]:
        projstr proj4投影参数字符串 
        res     分辨率
        unit    分辨率单位，支持 m km deg，默认 m
        row     行数
        col     列数
        pt_tl   左上角经纬度元组， 形式如 (lon, lat)
        pt_br   右下角经纬度元组， 形式如 (lon, lat)

        row、 col 和  pt_tl、 pt_br 两对里必须传一对，用以确定网格大小， 不能都是None

        projstr 样例：
        1. 等经纬
           "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +x_0=-half_res +y_0=half_res"
        2. 极射赤面
           "+proj=stere +ellps=clrk66 +lat_0=90 +lat_ts=70 +lon_0=0 +k_0=0.969858730377 +a=6371000 +units=m"
        3. 兰勃特等面积
           "+proj=laea +lat_0=-74.180000 +lon_0=-146.620000 +x_0=0 +y_0=0 +ellps=WGS84"
        4. 阿伯斯  (常用于中国区域)
           "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +ellps=krass +a=6378245.0 +b=6356863.0"
        5. 待补充

        '''
        self.proj4str = projstr
        self.pfunc = Proj(self.proj4str)  # 转换函数
        # self.res 统一转换成单位米
        if unit == "m":
            self.res = res
        elif unit == "km":
            self.res = res * 1000
        elif unit == "deg":
            self.res = deg2meter(res)

        if row is not None and col is not None:
            self.row = row
            self.col = col
            self.x_tl = -self.col / 2. * self.res
            self.y_tl = self.row / 2. * self.res

        elif pt_tl is not None and pt_br is not None:
            self.x_tl, self.y_tl = self.pfunc(*pt_tl)
            self.x_br, self.y_br = self.pfunc(*pt_br)

            self.row = int(ceil((self.y_tl - self.y_br) / self.res)) + 1
            self.col = int(ceil((self.x_br - self.x_tl) / self.res)) + 1
        else:
            raise ValueError(
                "row、 col 和  pt_tl、 pt_br 两对里必须传一对，用以确定网格大小， 不能都是None")

        self.grid_lonslats()

    def lonslats2ij(self, lons, lats):
        '''
        '经纬度转行列号 lons,lats -> i,j
        '参数是n维数组 经纬度
        '返回值是n维数组 行列号
        '''
        if isinstance(lons, (list, tuple)):
            lons = np.array(lons)
        if isinstance(lats, (list, tuple)):
            lats = np.array(lats)

        if isinstance(lons, np.ndarray):
            assert lons.shape == lats.shape, \
                "lons and lats must have same shape."

            args_shape = lons.shape
            # 转成1维，因为proj只接收1维参数
            lons = lons.reshape((-1))
            lats = lats.reshape((-1))
            # 通过平面坐标系计算投影后的行和列
            x, y = self.pfunc(lons, lats)

            i = self.__y2i(y)
            j = self.__x2j(x)
            return i.reshape(args_shape), j.reshape(args_shape)
        else:
            x, y = self.pfunc(lons, lats)
            i = self.__y2i(y)
            j = self.__x2j(x)
            return i, j

    def __y2i(self, y):
        '''
        y 转 行号
        '''
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        return np.rint((self.y_tl - y) / self.res).astype(int)

    def __x2j(self, x):
        '''
        x 转 列号
        '''
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        return np.rint((x - self.x_tl) / self.res).astype(int)

    def grid_lonslats(self):
        '''
        '生成投影后网格 各格点的经纬度
        '''
        # 制作一个2维的矩阵
        i, j = np.mgrid[0:self.row:1, 0:self.col:1]
        y = self.__i2y(i)
        x = self.__j2x(j)

        # 把二维的x,y 转成1维，因为proj只接收1维参数
        x = x.reshape((-1))
        y = y.reshape((-1))

        lons, lats = self.pfunc(x, y, inverse=True)
        # 转回2维
        self.lons = lons.reshape((self.row, self.col))
        self.lats = lats.reshape((self.row, self.col))

    def __i2y(self, i):
        '''
        '行号 转 y
        '''
        if isinstance(i, (list, tuple)):
            i = np.array(i)

        y = self.y_tl - i * self.res
        return y

    def __j2x(self, j):
        '''
        '列号 转 x
        '''
        if isinstance(j, (list, tuple)):
            j = np.array(j)
        x = j * self.res + self.x_tl
        return x

    def create_lut(self, Lons, Lats):
        '''
        '创建投影查找表, 
        '即 源数据经纬度位置与投影后位置的对应关系
        '''
        if isinstance(Lons, (list, tuple)):
            Lons = np.array(Lons)
        if isinstance(Lats, (list, tuple)):
            Lats = np.array(Lats)
        assert Lons.shape == Lats.shape, \
            "Lons and Lats must have same shape."

        # 投影后的行列 proj1_i,proj1_j
        proj1_i, proj1_j = self.lonslats2ij(Lons, Lats)

        # 根据投影前数据别分获取源数据维度，制作一个和数据维度一致的数组，分别存放行号和列号
        # 原始数据的行列, data1_i, data1_j
        data1_row, data1_col = Lons.shape
        data1_i, data1_j = np.mgrid[0:data1_row:1, 0:data1_col:1]

        # 投影方格以外的数据过滤掉

        condition = np.logical_and.reduce((proj1_i >= 0, proj1_i < self.row,
                                           proj1_j >= 0, proj1_j < self.col))
        p1_i = proj1_i[condition]
        p1_j = proj1_j[condition]
        d1_i = data1_i[condition]
        d1_j = data1_j[condition]

        fillValue = -999
        ii = np.full((self.row, self.col), fillValue, dtype='i4')
        jj = np.full((self.row, self.col), fillValue, dtype='i4')
        # 开始根据查找表对第一个文件的投影结果进行赋值
        ii[p1_i, p1_j] = d1_i
        jj[p1_i, p1_j] = d1_j
        self.lut_i = ii
        self.lut_j = jj

    def transform2ij(self, proj_str1, x1, y1):
        '''
        '不同投影方式之间转换
        '返回值是整数
        '''
        args_shape = x1.shape
        x1 = np.array(x1).reshape((-1))  # 转成1维
        y1 = np.array(y1).reshape((-1))
        p1 = Proj(proj_str1)
        x2, y2 = transform(p1, self.pfunc, x1, y1)
        i = self.__y2i(y2)
        j = self.__x2j(x2)
        return i.reshape(args_shape), j.reshape(args_shape)


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


if __name__ == '__main__':
    # 等经纬测试-----------------------------------------------------------------
    # 等经纬的 projstr 需要加上  +x_0=-half_res +y_0=half_res 才能保证经纬度在中心 ！！！
    # test1 test2 两种方式由于浮点数原因，有微妙的差别
    res = 0.17  # deg
    half_res = deg2meter(res) / 2.

    projstr = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=-%f +y_0=%f +datum=WGS84' % (
        half_res, half_res)

    print "test1"
    res = 5565
    p = prj_core(projstr, res, pt_tl=(-179.5, 89.5),
                 pt_br=(179.5, -89.5))  # 角点也要放在格点中心位置
    print p.y_tl, p.x_tl
    print '1', p.lonslats2ij(np.nan, np.nan)
    print p.lons.shape
    print p.lons[0, 0],  p.lats[0, 0]
    print p.lons[-1, -1],  p.lats[-1, -1]
#     print p.lonslats2ij(-180, 90), p.lonslats2ij(-180, 90), p.lonslats2ij(180, 90), p.lonslats2ij(-180, -90)
#     print p.lonslats2ij(10.6, 39)
#     print p.lonslats2ij(-179, 89), p.lonslats2ij(-179.001, 89.001), p.lonslats2ij(-178.999, 88.999)
#     print p.lons[0]

    print "\ntest2"
    res = 0.05
    p = prj_core(projstr, res, unit="deg", row=3600, col=7200)
#     p = prj_core(projstr, 17000, row=1179, col=2350)
    print p.y_tl, p.x_tl
    print 'mmmmm'
    print p.lons[0, 0],  p.lats[0, 0]
    print p.lons[-1, -1],  p.lats[-1, -1]
#     print p.lonslats2ij(-180, 90)
    print '11111111', p.lonslats2ij(np.nan, np.nan)
#     print p.lonslats2ij(-180, 90), p.lonslats2ij(-180, 90), p.lonslats2ij(180, 90), p.lonslats2ij(-180, -90)
#     print p.lonslats2ij(10.6, 39)
#     print p.lonslats2ij(-179, 89), p.lonslats2ij(-179.001, 89.001), p.lonslats2ij(-178.999, 88.999)
#     print p.lons[0]

    print "\nold"
    from dp_prj import prj_gll
    p = prj_gll(resLat=1, resLon=1)
    print p.lonslats2ij(-180, 90)
#     print p.lonslats2ij(-180, 90), p.lonslats2ij(-180, 90), p.lonslats2ij(180, 90), p.lonslats2ij(-180, -90)
#     print p.lonslats2ij(10.6, 39)
# print p.lonslats2ij(-179, 89), p.lonslats2ij(-179.001, 89.001),
# p.lonslats2ij(-178.999, 88.999)
