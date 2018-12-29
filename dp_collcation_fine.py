# coding: utf-8
'''
匹配类
@author: wangpeng
'''
import os
import sys

import h5py
import yaml

from DV import dv_plt, dv_img
from PB import pb_space
from PB.pb_sat import sun_glint_cal
from dp_prj import fill_points_2d
import numpy as np


def get_area_mean_std(data, P, window):

    ary_data = []
    ary_mean = []
    ary_std = []

    for i in xrange(len(P.Lut_row_if)):
        data_area = np.full((window[0], window[1]), -999.)
        # 目前只要完成的区域块，不足的不使用
        if P.Lut_row_e[i] - P.Lut_row_b[i] == window[0] and \
           P.Lut_col_e[i] - P.Lut_col_b[i] == window[1]:
            iy = P.Lut_row_b[i] + P.Lut_row_if[i]
            ix = P.Lut_col_b[i] + P.Lut_col_if[i]
            data_area[P.Lut_row_if[i], P.Lut_col_if[i]] = data[iy, ix]

            # nan值改为-999.
            idx = np.where(np.isnan(data_area))
            data_area[idx] = -999.

            # 全部是nan 就改成-999.
            if (np.isnan(data[iy, ix])).all():
                mean = -999.
                std = -999.
            else:
                mean = np.nanmean(data[iy, ix])
                std = np.nanstd(data[iy, ix])
        else:
            mean = -999.
            std = -999.
        ary_data.append(data_area)
        ary_mean.append(mean)
        ary_std.append(std)

    return np.array(ary_data), np.array(ary_mean), np.array(ary_std)


class ReadModeYaml():
    """
        读取yaml格式配置文件,解析匹配的传感器对的默认配置参数
    """

    def __init__(self, inFile):

        if not os.path.isfile(inFile):
            print 'Not Found %s' % inFile
            sys.exit(-1)

        with open(inFile, 'r') as stream:
            cfg = yaml.load(stream)
        self.sensor1 = cfg['sensor1']
        self.sensor2 = cfg['sensor2']
        self.chan1 = cfg['chan1']
        self.chan2 = cfg['chan2']
        self.rewrite = cfg['rewrite']
        self.drawmap = cfg['drawmap']
        self.FovWind1 = cfg['FovWind1']
        self.EnvWind1 = cfg['EnvWind1']
        self.FovWind2 = cfg['FovWind2']
        self.EnvWind2 = cfg['EnvWind2']

        self.S1_satHeight = cfg['S1_satHeight']
        self.S1_resolution = cfg['S1_resolution']
        self.S2_Fov_fov = cfg['S2_Fov_fov']
        self.S2_Env_fov = cfg['S2_Env_fov']
        self.row = cfg['S2_row']
        self.col = cfg['S2_col']

        self.solglint_min = cfg['solglint_min']
        self.solzenith_max = cfg['solzenith_max']
        self.satzenith_max = cfg['satzenith_max']
        self.timedif_max = cfg['timedif_max']
        self.angledif_max = cfg['angledif_max']
        self.distdif_max = cfg['distdif_max']
        self.AutoRange = cfg['AutoRange']
        self.clear_band_ir = cfg['clear_band_ir']
        self.clear_min_ir = cfg['clear_min_ir']
        self.clear_band_vis = cfg['clear_band_vis']
        self.clear_max_vis = cfg['clear_max_vis']

        if 'write_spec' in cfg.keys():
            self.write_spec = cfg['write_spec']
        else:
            self.write_spec = None

        if 'axis_ref' in cfg.keys():
            self.axis_ref = cfg['axis_ref']
        if 'axis_rad' in cfg.keys():
            self.axis_rad = cfg['axis_rad']
        if 'axis_tbb' in cfg.keys():
            self.axis_tbb = cfg['axis_tbb']

        # 将通道阈值放入字典
        self.CH_threshold = {}
        for ch in self.chan1:
            if ch not in self.CH_threshold.keys():
                self.CH_threshold[ch] = {}
            for threshold in cfg[ch]:
                self.CH_threshold[ch][threshold] = cfg[ch][threshold]


def _get_band_dataset(set_name, hdf5_file, band=None):
    """
    获取某个通道对应名字的数据集，如果没有，赋值为 None
    :param band: 通道名
    :param set_name: 数据集名
    :param hdf5: hdf5 文件
    :return:
    """
    if band is None:
        keys = hdf5_file.keys()
        if set_name in keys:
            return hdf5_file.get(set_name)[:]
        else:
            return None
    else:
        keys = hdf5_file.get(band).keys()
        if set_name in keys:
            return hdf5_file.get(band)[set_name][:]
        else:
            return None


def regression(x, y, value_min, value_max, flag, ICFG, MCFG, Band):

    # FY4分布
    MainPath, MainFile = os.path.split(ICFG.ofile)
    if not os.path.isdir(MainPath):
        os.makedirs(MainPath)

    meanbais = (np.mean(x - y) / np.mean(y)) * 100.

    p = dv_plt.dv_scatter(figsize=(7, 5))
    p.easyplot(x, y, None, None, marker='o', markersize=5)

    p.xlim_min = p.ylim_min = value_min
    p.xlim_max = p.ylim_max = value_max

    p.title = u'%s' % (ICFG.ymd)
    p.xlabel = u'%s %s %s' % (ICFG.sat1, ICFG.sensor1, flag)
    p.ylabel = u'%s %s %s' % (ICFG.sat2, ICFG.sensor2, flag)
    # 计算AB
    ab = np.polyfit(x, y, 1)
    p.regression(ab[0], ab[1], 'b')

    # 计算相关性
    p.show_leg = True
    r = np.corrcoef(x, y)
    rr = r[0][1] * r[0][1]
    nums = len(x)
    # 绘制散点
    strlist = [[r'$%0.4fx%+0.4f (R=%0.4f) $' % (ab[0], ab[1], rr),
                r'count:%d' % nums, r'%sMeanBias: %0.4f' % (flag, meanbais)]]
    p.annotate(strlist, 'left', 'r')
    ofile = os.path.join(MainPath, '%s+%s_%s+%s_%s_%s_%s.png' %
                         (ICFG.sat1, ICFG.sensor1, ICFG.sat2, ICFG.sensor2, ICFG.ymd, Band, flag))
    p.savefig(ofile, dpi=300)


class COLLOC_COMM(object):
    """
    交叉匹配的公共类，首先初始化所有参数信息
    """

    def __init__(self, row, col, BandLst):

        # 默认填充值 和 数据类型
        self.row = row
        self.col = col
        self.FillValue = -999.
        self.dtype = 'f4'
        self.BandLst = BandLst

        # 粗匹配掩码记录表
        self.MaskRough = np.full((row, col), 0, 'i1')
        self.PubIdx = np.full((row, col), 0, 'i1')
        # 精匹配掩码记录表，按照通道存放
        self.MaskFine = {}

        # 初始化字典内的存放每个通道的数据空间
        for band in BandLst:
            self.MaskFine[band] = np.full((row, col), 0, 'i1')

        # 投影后的全局变量信息
        self.S1_Time = np.full((row, col), self.FillValue, self.dtype)
        self.S1_Lon = np.full((row, col), self.FillValue, self.dtype)
        self.S1_Lat = np.full((row, col), self.FillValue, self.dtype)
        self.S1_SatA = np.full((row, col), self.FillValue, self.dtype)
        self.S1_SatZ = np.full((row, col), self.FillValue, self.dtype)
        self.S1_SoA = np.full((row, col), self.FillValue, self.dtype)
        self.S1_SoZ = np.full((row, col), self.FillValue, self.dtype)

        self.S2_Time = np.full((row, col), self.FillValue, self.dtype)
        self.S2_Lon = np.full((row, col), self.FillValue, self.dtype)
        self.S2_Lat = np.full((row, col), self.FillValue, self.dtype)
        self.S2_SatA = np.full((row, col), self.FillValue, self.dtype)
        self.S2_SatZ = np.full((row, col), self.FillValue, self.dtype)
        self.S2_SoA = np.full((row, col), self.FillValue, self.dtype)
        self.S2_SoZ = np.full((row, col), self.FillValue, self.dtype)

    def correct_target_ref_data(self):
        '''
        订正第二颗传感器可见光通道的ref值 
        '''
        coeff = np.cos(np.deg2rad(self.S1_SoZ)) / \
            np.cos(np.deg2rad(self.S2_SoZ))
        if hasattr(self, 'S1_FovRefMean'):
            for band in sorted(self.S1_FovRefMean.keys()):
                if band in self.S2_FovRefMean.keys():
                    print '订正 sensor2 ref band %s' % band
                    idx = np.where(self.S2_FovRefMean[band] > 0)
                    self.S2_FovRefMean[band][idx] = self.S2_FovRefMean[
                        band][idx] * coeff[idx]
                    idx = np.where(self.S2_EnvRefMean[band] > 0)
                    self.S2_EnvRefMean[band][idx] = self.S2_EnvRefMean[
                        band][idx] * coeff[idx]

    def save_rough_data(self, P1, P2, D1, D2, modeCfg):
        """
        第一轮匹配，根据查找表进行数据的mean和std计算，并且对全局物理量复制（角度，经纬度，时间等）
        """
        print u'对公共区域位置进行数据赋值......'
        condition = np.logical_and(P1.Lut_row >= 0, P1.Lut_col >= 0)
        idx = np.where(condition)
        print u'FY LEO 公共区域匹配点个数 %d' % len(idx[0])
        # 粗匹配点没有则返回
        if len(idx[0]) == 0:
            return
        # 记录公共匹配点
        self.PubIdx[idx] = 1

        # 根据朝找表 计算原数据行列信息
        i1 = P1.Lut_row[idx]
        j1 = P1.Lut_col[idx]
        i2 = idx[0]
        j2 = idx[1]

        # 11111111 保存传感器1,2 的投影公共数据信息
        self.S1_Time[idx] = D1.Time[i1, j1]
        self.S1_Lon[idx] = D1.Lons[i1, j1]
        self.S1_Lat[idx] = D1.Lats[i1, j1]
        self.S1_SatA[idx] = D1.satAzimuth[i1, j1]
        self.S1_SatZ[idx] = D1.satZenith[i1, j1]
        self.S1_SoA[idx] = D1.sunAzimuth[i1, j1]
        self.S1_SoZ[idx] = D1.sunZenith[i1, j1]

        self.S2_Time[idx] = D2.Time[i2, j2]
        self.S2_Lon[idx] = D2.Lons[i2, j2]
        self.S2_Lat[idx] = D2.Lats[i2, j2]
        self.S2_SatA[idx] = D2.satAzimuth[i2, j2]
        self.S2_SatZ[idx] = D2.satZenith[i2, j2]
        self.S2_SoA[idx] = D2.sunAzimuth[i2, j2]
        self.S2_SoZ[idx] = D2.sunZenith[i2, j2]

        # 如果需要保存光谱信息 则进行光谱数据处理并保存，只保存公共区域的信息
        if modeCfg.write_spec:
            if hasattr(D2, 'radiance') and not hasattr(self, 'S2_Spectral'):
                self.S2_Spectral = []
            if hasattr(self, 'S2_Spectral'):
                self.S2_Spectral = D2.radiance

        # 记录探元号
        if hasattr(D2, 'pixel_num') and not hasattr(self, 'S2_PixelNum'):
            self.S2_PixelNum = None
        if hasattr(self, 'S2_PixelNum'):
            self.S2_PixelNum = D2.pixel_num

        # 记录轨道号信息
        if hasattr(D1, 'orbit_direction') and not hasattr(self, 'S1_orbit_direction'):
            self.S1_orbit_direction = []
            self.S1_orbit_num = []
        if hasattr(self, 'S1_orbit_direction'):
            LstIdx1 = int(len(D1.orbit_direction) / 2)
            self.S1_orbit_direction = D1.orbit_direction[LstIdx1]
            self.S1_orbit_num = int(D1.orbit_num[LstIdx1])

        if hasattr(D2, 'orbit_direction') and not hasattr(self, 'S2_orbit_direction'):
            self.S2_orbit_direction = []
            self.S2_orbit_num = []
        if hasattr(self, 'S2_orbit_direction'):
            LstIdx1 = int(len(D2.orbit_direction) / 2)
            self.S2_orbit_direction = D2.orbit_direction[LstIdx1]
            self.S2_orbit_num = int(D2.orbit_num[LstIdx1])

        # 保存传感器1的定标系数保存
        if hasattr(D1, 'cal_coeff1') and not hasattr(self, 'S1_cal_coeff'):
            # 存放规则k0,k1,k2 ,红外通道则是k0,k2,k3,  k1放到数据集中了
            self.S1_cal_coeff = {}
        if hasattr(self, 'S1_cal_coeff'):
            for band1 in modeCfg.chan1:
                self.S1_cal_coeff[band1] = D1.cal_coeff1[band1]

        # 保存辅助信息，经纬度区域信息 （精匹配增加）
        self.__init_class_array_LonLat(
            D1, 'Lons', 'S1_LonArea', modeCfg, P1, idx)
        self.__init_class_array_LonLat(
            D1, 'Lats', 'S1_LatArea', modeCfg, P1, idx)

        # # wangpeng add 2018-12-10 增加云检测标识
        self.__init_class_array_LonLat(
            D1, 'CloudMask', 'S1_CloudMaskArea', modeCfg, P1, idx)

        # 保存辅助信息，开始计算中心点圈的37个点位置 (精匹配增加)
        if not hasattr(self, 'S2_FootLat'):
            self.S2_FootLat = []
            self.S2_FootLon = []

        if hasattr(self, 'S2_FootLat'):
            FootSahpe = D2.Lats.shape + (37,)
            dfai = modeCfg.S2_Fov_fov
            for i in xrange(self.row):
                for j in xrange(self.col):

                    fLon, fLat = pb_space.ggp_footpoint(D2.L_pos[i, j],
                                                        D2.P_pos[i, j],
                                                        D2.satZenith[i, j],
                                                        D2.satAzimuth[i, j],
                                                        D2.Lats[i, j],
                                                        D2.Lons[i, j], dfai)

                    self.S2_FootLat.append(fLat)
                    self.S2_FootLon.append(fLon)

            self.S2_FootLat = np.array(self.S2_FootLat)
            self.S2_FootLon = np.array(self.S2_FootLon)
            self.S2_FootLat = self.S2_FootLat.reshape(FootSahpe)
            self.S2_FootLon = self.S2_FootLon.reshape(FootSahpe)
        # end  画圈圈结束

        # 其他辅助信息
        self.__init_class_array_geo(
            D1, 'LandCover', 'S1_LandCover', i1, j1, i2, j2)
        self.__init_class_array_geo(
            D1, 'LandSeaMask', 'S1_LandSeaMask', i1, j1, i2, j2)

        self.__init_class_array_geo(
            D2, 'LandSeaMask', 'S2_LandSeaMask', i2, j2, i2, j2)
        self.__init_class_array_geo(
            D2, 'LandSeaMask', 'S2_LandSeaMask', i2, j2, i2, j2)

        self.__init_class_dict_geo(
            D1, 'SV', 'S1_SV', modeCfg, i1, j1, i2, j2)
        self.__init_class_dict_geo(
            D1, 'BB', 'S1_BB', modeCfg, i1, j1, i2, j2)
        self.__init_class_dict_geo(
            D1, 'cal_coeff2', 'S1_IrCalSlope', modeCfg, i1, j1, i2, j2)
        self.__init_class_dict_geo(
            D2, 'SV', 'S2_SV', modeCfg, i2, j2, i2, j2)
        self.__init_class_dict_geo(
            D2, 'BB', 'S2_BB', modeCfg, i2, j2, i2, j2)

        # 关键物理量的均值和std计算
        self.__init_class_dict_dn(D1, 'Dn', 'S1', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D1, 'Ref', 'S1', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D1, 'Rad', 'S1', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D1, 'Tbb', 'S1', modeCfg, P1, P2, idx)
        # D2的数据直接赋值 std 就是0 均值就是自己
        self.__init_class_dict_dn(D2, 'Dn', 'S2', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D2, 'Ref', 'S2', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D2, 'Rad', 'S2', modeCfg, P1, P2, idx)
        self.__init_class_dict_dn(D2, 'Tbb', 'S2', modeCfg, P1, P2, idx)

    def __init_class_dict_dn(self, idata, name, sensor, modeCfg, P1, P2, idx):

        member1 = '%s_Fov%sArea' % (sensor, name)
        member2 = '%s_Fov%sMean' % (sensor, name)
        member3 = '%s_Fov%sStd' % (sensor, name)
        member4 = '%s_Env%sMean' % (sensor, name)
        member5 = '%s_Env%sStd' % (sensor, name)

        if hasattr(idata, name) and not hasattr(self, member2):

            areaShape = (self.row, self.col) + \
                (modeCfg.FovWind1[0], modeCfg.FovWind1[1])
            if 'S1' in sensor:
                self.__dict__[member1] = {}
            self.__dict__[member2] = {}
            self.__dict__[member3] = {}
            self.__dict__[member4] = {}
            self.__dict__[member5] = {}

            for band1 in modeCfg.chan1:
                index = modeCfg.chan1.index(band1)
                band2 = modeCfg.chan2[index]
                if 'S1' in sensor:
                    band = band1
                elif 'S2' in sensor:
                    band = band2

                if band in eval('idata.%s.keys()' % name):
                    if 'S1' in sensor:
                        self.__dict__[member1][band1] = np.full(
                            areaShape, self.FillValue, self.dtype)
                    self.__dict__[member2][band1] = np.full(
                        (self.row, self.col), self.FillValue, self.dtype)
                    self.__dict__[member3][band1] = np.full(
                        (self.row, self.col), self.FillValue, self.dtype)
                    self.__dict__[member4][band1] = np.full(
                        (self.row, self.col), self.FillValue, self.dtype)
                    self.__dict__[member5][band1] = np.full(
                        (self.row, self.col), self.FillValue, self.dtype)

        if hasattr(self, member2):
            for band1 in modeCfg.chan1:
                index = modeCfg.chan1.index(band1)
                band2 = modeCfg.chan2[index]
                if 'S1' in sensor:
                    band = band1
                elif 'S2' in sensor:
                    band = band2
                if band in eval('idata.%s.keys()' % name):
                    # sat1 Fov和Env dn的mean和std
                    data = eval('idata.%s["%s"]' % (name, band))
                    if 'S1' in sensor:
                        # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                        area, mean, std = get_area_mean_std(
                            data, P1, modeCfg.FovWind1)
                        self.__dict__[member1][band1][idx] = area
                        self.__dict__[member2][band1][idx] = mean
                        self.__dict__[member3][band1][idx] = std
                        area, mean, std = get_area_mean_std(
                            data, P2, modeCfg.EnvWind1)
                        self.__dict__[member4][band1][idx] = mean
                        self.__dict__[member5][band1][idx] = std
                    else:
                        self.__dict__[member2][band1][idx] = data[idx]
                        self.__dict__[member3][band1][idx] = 0.
                        self.__dict__[member4][band1][idx] = data[idx]
                        self.__dict__[member5][band1][idx] = 0.

    def __init_class_dict_geo(self, idata, name1, name2, modeCfg, i, j, x, y):
        '''
        直接使用查找表赋值，不需要计算均值std 和 窗区
        '''

        if hasattr(idata, name1) and not hasattr(self, name2):
            self.__dict__[name2] = {}

            for band1 in modeCfg.chan1:
                index = modeCfg.chan1.index(band1)
                band2 = modeCfg.chan2[index]
                if 'S1' in name2:
                    band = band1
                elif 'S2' in name2:
                    band = band2

                if band in eval('idata.%s.keys()' % name1):
                    self.__dict__[name2][band1] = np.full(
                        (self.row, self.col), self.FillValue, self.dtype)

        if hasattr(self, name2):
            for band1 in modeCfg.chan1:
                index = modeCfg.chan1.index(band1)
                band2 = modeCfg.chan2[index]
                if 'S1' in name2:
                    band = band1
                elif 'S2' in name2:
                    band = band2
                if band in eval('idata.%s.keys()' % name1):
                    data = eval('idata.%s["%s"]' % (name1, band))
                    self.__dict__[name2][band1][x, y] = data[i, j]

    def __init_class_array_geo(self, idata, name1, name2, i, j, x, y):

        if hasattr(idata, name1) and not hasattr(self, name2):
            self.__dict__[name2] = np.full((self.row, self.col), -999, 'i2')
        if hasattr(self, name2):
            data = eval('idata.%s' % name1)
            self.__dict__[name2][x, y] = data[i, j]

    def __init_class_array_LonLat(self, idata, name1, name2, modeCfg, P1, idx):

        if hasattr(idata, name1) and not hasattr(self, name2):
            areaShape = (self.row, self.col) + \
                (modeCfg.FovWind1[0], modeCfg.FovWind1[1])
            self.__dict__[name2] = np.full(areaShape, -999., 'f4')
        if hasattr(self, name2):
            data = eval('idata.%s' % name1)
            area, _, _ = get_area_mean_std(data, P1, modeCfg.FovWind1)
            self.__dict__[name2][idx] = area

    def save_fine_data(self, modeCfg):
        """
        第二轮匹配，根据各通道的的mean和std计以为，角度和距离等进行精细化过滤
        """

        # 最终的公共匹配点数量
        idx = np.where(self.PubIdx > 0)
        if len(idx[0]) == 0:
            return
        print u'所有粗匹配点数目 ', len(idx[0])

        # 掩码清零
        self.MaskRough[:] = 0

        # 计算共同区域的距离差 #########
        disDiff = np.full_like(self.S1_Time, '-1', dtype='i2')
        a = np.power(self.S2_Lon[idx] - self.S1_Lon[idx], 2)
        b = np.power(self.S2_Lat[idx] - self.S1_Lat[idx], 2)
        disDiff[idx] = np.sqrt(a + b) * 100.

        idx_Rough = np.logical_and(disDiff < modeCfg.distdif_max, disDiff >= 0)
        idx1 = np.where(idx_Rough)
        print u'1. 距离过滤后剩余点 ', len(idx1[0])

        timeDiff = np.abs(self.S1_Time - self.S2_Time)

        idx_Rough = np.logical_and(idx_Rough, timeDiff <= modeCfg.timedif_max)
        idx1 = np.where(idx_Rough)
        print u'2. 时间过滤后剩余点 ', len(idx1[0])
        # 过滤太阳天顶角 ###############
        idx_Rough = np.logical_and(
            idx_Rough, self.S1_SoZ <= modeCfg.solzenith_max)
        idx_Rough = np.logical_and(
            idx_Rough, self.S2_SoZ <= modeCfg.solzenith_max)
        idx1 = np.where(idx_Rough)
        print u'3. 太阳天顶角过滤后剩余点 ', len(idx1[0])

        # 计算耀斑角 ###############
        glint1 = np.full_like(self.S1_SatZ, -999.)
        glint2 = np.full_like(self.S1_SatZ, -999.)

        glint1[idx] = sun_glint_cal(
            self.S1_SatA[idx], self.S1_SatZ[idx], self.S1_SoA[idx], self.S1_SoZ[idx])
        glint2[idx] = sun_glint_cal(
            self.S2_SatA[idx], self.S2_SatZ[idx], self.S2_SoA[idx], self.S2_SoZ[idx])

        idx_Rough = np.logical_and(idx_Rough, glint1 > modeCfg.solglint_min)
        idx_Rough = np.logical_and(idx_Rough, glint2 > modeCfg.solglint_min)

        idx1 = np.where(idx_Rough)
        print u'4. 太阳耀斑角过滤后剩余点 ', len(idx1[0])

        # 角度均匀性 #################
        SatZRaio = np.full_like(self.S1_Time, 9999)
        SatZ1 = np.cos(self.S1_SatZ[idx] * np.pi / 180.)
        SatZ2 = np.cos(self.S2_SatZ[idx] * np.pi / 180.)
        SatZRaio[idx] = np.abs(SatZ1 / SatZ2 - 1.)

        idx_Rough = np.logical_and(idx_Rough, SatZRaio <= modeCfg.angledif_max)
        idx1 = np.where(idx_Rough)
        print u'5. 卫星天顶角均匀性过滤后剩余点 ', len(idx1[0])

        idx_Rough = np.logical_and(
            idx_Rough, self.S1_SatZ <= modeCfg.satzenith_max)
        idx1 = np.where(idx_Rough)
        print u'6. FY卫星观测角(天顶角)滤后剩余点 ', len(idx1[0])
        self.MaskRough[idx1] = 1

        # 添加spec, 粗匹配后剩余的点是要记录光谱谱线的。。。2维转1维下标
        idx_1d = np.ravel_multi_index(idx1, (self.row, self.col))

        if modeCfg.write_spec:
            # 定义spec_MaskRough_value 然后记录需要保存的谱线
            if hasattr(self, 'S2_Spec_all'):
                for i in idx_1d:
                    self.S2_Spectral.append(self.S2_Spec_all[i])
                self.S2_Spectral = np.array(self.S2_Spectral)

                # 记录根据MaskRough表记录的格点信息
                self.S2_Spec_row = idx1[0]
                self.S2_Spec_col = idx1[1]

        for Band1 in modeCfg.chan1:
            # 掩码清零
            self.MaskFine[Band1][:] = 0

            th_vaue_max = modeCfg.CH_threshold[Band1]['value_max']
            th1 = modeCfg.CH_threshold[Band1]['angledif_max']
            th2 = modeCfg.CH_threshold[Band1]['homodif_fov_max']
            th3 = modeCfg.CH_threshold[Band1]['homodif_env_max']
            th4 = modeCfg.CH_threshold[Band1]['homodif_fov_env_max']

            th_cld1 = modeCfg.CH_threshold[Band1]['cld_angledif_max']
            th_cld2 = modeCfg.CH_threshold[Band1]['cld_homodif_fov_max']
            th_cld3 = modeCfg.CH_threshold[Band1]['cld_homodif_env_max']
            th_cld4 = modeCfg.CH_threshold[Band1]['cld_homodif_fov_env_max']

            flag = 0
            # 如果 rad和tbb都有就用rad 做均匀性判断
            if hasattr(self, 'S1_FovRadMean')and Band1 in self.S1_FovRadMean.keys():
                flag = 'ir'
                # 固定通道值用于检测红外晴空和云
#                 irValue = self.FovTbbMean1[modeCfg.clear_band_ir]
                homoFov1 = np.abs(
                    self.S1_FovRadStd[Band1] / self.S1_FovRadMean[Band1])
                homoEnv1 = np.abs(
                    self.S1_EnvRadStd[Band1] / self.S1_EnvRadMean[Band1])
                homoFovEnv1 = np.abs(
                    self.S1_FovRadMean[Band1] / self.S1_EnvRadMean[Band1] - 1)
                homoValue1 = self.S1_FovRadMean[Band1]

                homoFov2 = np.abs(
                    self.S2_FovRadStd[Band1] / self.S2_FovRadMean[Band1])
                homoEnv2 = np.abs(
                    self.S2_EnvRadStd[Band1] / self.S2_EnvRadMean[Band1])
                homoFovEnv2 = np.abs(
                    self.S2_FovRadMean[Band1] / self.S2_EnvRadMean[Band1] - 1)
                homoValue2 = self.S2_FovRadMean[Band1]

            # 如果只有 tbb 就用tbb
            if hasattr(self, 'S1_FovTbbMean') and not hasattr(self, 'S1_FovRadMean'):
                if Band1 in self.S1_FovTbbMean.keys():
                    flag = 'ir'
                    # 固定通道值用于检测红外晴空和云
    #                 irValue = self.FovTbbMean1[modeCfg.clear_band_ir]
                    homoFov1 = np.abs(
                        self.FovTbbStd1[Band1] / self.FovTbbMean1[Band1])
                    homoEnv1 = np.abs(
                        self.EnvTbbStd1[Band1] / self.EnvTbbMean1[Band1])
                    homoFovEnv1 = np.abs(
                        self.FovTbbMean1[Band1] / self.EnvTbbMean1[Band1] - 1)
                    homoValue1 = self.FovTbbMean1[Band1]
                    homoFov2 = np.abs(
                        self.FovTbbStd2[Band1] / self.FovTbbMean2[Band1])
                    homoEnv2 = np.abs(
                        self.EnvTbbStd2[Band1] / self.EnvTbbMean2[Band1])
                    homoFovEnv2 = np.abs(
                        self.FovTbbMean2[Band1] / self.EnvTbbMean2[Band1] - 1)
                    homoValue2 = self.FovTbbMean2[Band1]
            # 可见光通道
            if hasattr(self, 'S1_FovRefMean') and Band1 in self.S1_FovRefMean.keys():
                flag = 'vis'
#                 visValue = self.FovRefMean1[modeCfg.clear_band_vis]
                homoFov1 = np.abs(
                    self.S1_FovRefStd[Band1] / self.S1_FovRefMean[Band1])
                homoEnv1 = np.abs(
                    self.S1_EnvRefStd[Band1] / self.S1_EnvRefMean[Band1])
                homoFovEnv1 = np.abs(
                    self.S1_FovRefMean[Band1] / self.S1_EnvRefMean[Band1] - 1)
                homoValue1 = self.S1_FovRefMean[Band1]

                homoFov2 = np.abs(
                    self.S2_FovRefStd[Band1] / self.S2_FovRefMean[Band1])
                homoEnv2 = np.abs(
                    self.S2_EnvRefStd[Band1] / self.S2_EnvRefMean[Band1])
                homoFovEnv2 = np.abs(
                    self.S2_FovRefMean[Band1] / self.S2_EnvRefMean[Band1] - 1)
                homoValue2 = self.S2_FovRefMean[Band1]

            # 云判识关闭状态 ####
            if (modeCfg.clear_min_ir == 0 and 'ir' in flag) or (modeCfg.clear_max_vis == 0 and 'vis' in flag):

                condition = np.logical_and(self.MaskRough > 0, True)
                condition = np.logical_and(SatZRaio < th1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,角度均匀性过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFov1 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoEnv1 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,环境过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFovEnv1 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区环境过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoValue1 < th_vaue_max, condition)
                condition = np.logical_and(homoValue1 > 0, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,饱和值过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                # sat 2过滤

                condition = np.logical_and(homoFov2 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoEnv2 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,环境2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFovEnv2 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区环境2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,饱和值2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                self.MaskFine[Band1][idx] = 1

            # 云判识开启状态 ####
            else:
                # 晴空判别
                if 'ir' in flag:
                    # 固定通道值用于检测可见晴空和云
                    irValue = self.S1_FovTbbMean[modeCfg.clear_band_ir]
                    condition = np.logical_and(
                        self.MaskRough > 0, irValue >= modeCfg.clear_min_ir)
                elif 'vis' in flag:
                    # 固定通道值用于检测可见晴空和云
                    visValue = self.S1_FovRefMean[modeCfg.clear_band_vis]
                    condition = np.logical_and(
                        self.MaskRough > 0, visValue < modeCfg.clear_max_vis)
                    condition = np.logical_and(visValue > 0, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(SatZRaio < th1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 角度 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFov1 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv1 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 环境1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv1 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区/环境 1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue1 > 0, condition)
                condition = np.logical_and(homoValue1 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFov2 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv2 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 环境2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv2 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区/环境 2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                idx_clear = np.where(condition)
                self.MaskFine[Band1][idx_clear] = 1

                # 云区判别
                if 'ir' in flag:
                    condition = np.logical_and(
                        self.MaskRough > 0, irValue < modeCfg.clear_min_ir)
                    condition = np.logical_and(irValue > 0, condition)
                elif 'vis' in flag:
                    condition = np.logical_and(
                        self.MaskRough > 0, visValue >= modeCfg.clear_max_vis)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(SatZRaio < th_cld1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 角度 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFov1 < th_cld2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv1 < th_cld3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 环境1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv1 < th_cld4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区/环境 1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue1 > 0, condition)
                condition = np.logical_and(homoValue1 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFov2 < th_cld2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv2 < th_cld3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 环境2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv2 < th_cld4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区/环境 2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                idx_cloud = np.where(condition)
                totalNums = len(idx_cloud[0]) + len(idx_clear[0])
                print u'%s %s 云判识开启，匹配点个数，晴空 %d 云区 %d 总计：%d' % (Band1, flag, len(idx_clear[0]), len(idx_cloud[0]), totalNums)
                self.MaskFine[Band1][idx_cloud] = 1

    def reload_data(self, ICFG, MCFG):
        """
        :param ICFG: 输入配置文件
        :param MCFG: 阈值配置文件
        :return:
        """
        print ' reaload data'
        i_file = ICFG.ofile

        with h5py.File(i_file, 'r') as h5file_r:
            for key in h5file_r.keys():
                pre_rootgrp = h5file_r.get(key)  # 获取根下名字
                if type(pre_rootgrp).__name__ == "Group":
                    # 组下的所有数据集
                    for key_grp in pre_rootgrp.keys():
                        if not hasattr(self, key_grp):
                            self.__dict__[key_grp] = {}
                        dset_path = '/' + key + '/' + key_grp
                        self.__dict__[key_grp][
                            key] = h5file_r.get(dset_path)[:]

                # 根下的数据集
                else:
                    self.__dict__[key] = h5file_r.get(key)[:]

    def rewrite_hdf5(self, ICFG, MCFG):
        """
        :param ICFG: 输入配置文件
        :param MCFG: 阈值配置文件
        :return:
        """
        i_file = ICFG.ofile
        with h5py.File(i_file, 'r+') as hdf5File:
            dset = hdf5File.get('MaskRough')
            dset[...] = self.MaskRough
#             # 增加云记录
#             hdf5File.create_dataset(
#                 'S1_CloudMaskArea', data=self.S1_CloudMaskArea, compression='gzip', compression_opts=5, shuffle=True)

            for band in MCFG.chan1:
                dset = hdf5File.get('%s/MaskFine' % band)
                dset[...] = self.MaskFine.get('%s' % band)

    def write_hdf5(self, ICFG, MCFG):

        print u'输出产品'
        for band in MCFG.chan1:
            idx = np.where(self.MaskFine[band] > 0)
            DCLC_nums = len(idx[0])
            if DCLC_nums > 0:
                break
        if DCLC_nums == 0:
            print('colloc point is zero')
            sys.exit(-1)

        # 创建文件夹
        MainPath, _ = os.path.split(ICFG.ofile)
        if not os.path.isdir(MainPath):
            os.makedirs(MainPath)

        # 创建hdf5文件
        h5File_W = h5py.File(ICFG.ofile, 'w')

        if hasattr(self, 'S1_orbit_direction'):
            h5File_W.attrs.create(
                'S1_orbit_direction', self.S1_orbit_direction, shape=(1,), dtype='S2')
            h5File_W.attrs.create(
                'S1_orbit_num', self.S1_orbit_num, shape=(1,), dtype='i4')

        if hasattr(self, 'S2_orbit_direction'):
            h5File_W.attrs.create(
                'S2_orbit_direction', self.S2_orbit_direction, shape=(1,), dtype='S2')
            h5File_W.attrs.create(
                'S2_orbit_num', self.S2_orbit_num, shape=(1,), dtype='i4')

        # wangpeng add 2018-06-30 增加系数文件
        if hasattr(self, 'S1_cal_coeff'):
            h5File_W.attrs.create(
                'S1_cal_coeff', self.S1_cal_coeff.values(), dtype='f4')

            for band in MCFG.chan1:
                group = h5File_W.create_group(band)
                group.attrs.create(
                    'S1_cal_coeff', self.S1_cal_coeff[band], dtype='f4')

        for member in self.__dict__.keys():
            dname = eval('self.%s' % member)
            if isinstance(dname, dict):
                # 注意，注意 ！！！ 记录光谱信息的字典要过滤掉，在重处理模式下这个没法更新，需要删除文件处理
                if 'S2_Spec_all' in member:
                    continue
                elif 'S1_cal_coeff' in member:
                    continue
                for band in dname.keys():
                    str_dname = '/' + band + '/' + member
                    dset = h5File_W.create_dataset(
                        str_dname, data=dname[band], compression='gzip', compression_opts=5, shuffle=True)
                    if 'MaskFine' in member:
                        dset.attrs.create(
                            'Long_name', 'after scene homogenous collocation', shape=(1,), dtype='S32')

            elif isinstance(dname, np.ndarray):
                dset = h5File_W.create_dataset(
                    member, data=dname, compression='gzip', compression_opts=5, shuffle=True)
                if 'S2_Spectral' in member:
                    dset.attrs.create(
                        'Long_name', 'Record spectral lines obtained from MaskRough dataset', shape=(1,), dtype='S64')
                elif 'MaskRough' in member:
                    dset.attrs.create(
                        'Long_name', 'after time and angle collocation', shape=(1,), dtype='S32')

        h5File_W.close()

    def draw_dclc(self, ICFG, MCFG):

        print u'回归图'

        for Band in MCFG.chan1:
            idx = np.where(self.MaskFine[Band] > 0)
            if hasattr(self, 'S1_FovRefMean'):
                if Band in self.S1_FovRefMean.keys() and Band in MCFG.axis_ref:
                    x = self.S1_FovRefMean[Band][idx]
                    y = self.S2_FovRefMean[Band][idx]
                    if len(x) >= 2:
                        value_min = value_max = None
                        flag = 'Ref'
                        print('ref', Band, len(x), np.min(x), np.max(x),
                              np.min(y), np.max(y))
                        if MCFG.AutoRange == 'ON':
                            value_min = np.min([np.min(x), np.min(y)])
                            value_max = np.max([np.max(x), np.max(y)])
                        elif len(MCFG.axis_ref) != 0:
                            value_min = MCFG.axis_ref[Band][0]
                            value_max = MCFG.axis_ref[Band][1]
                            print(Band, value_min, value_max)
                        if value_min is not None and value_max is not None:
                            regression(x, y, value_min, value_max,
                                       flag, ICFG, MCFG, Band)
            if hasattr(self, 'S1_FovRadMean'):
                if Band in self.S1_FovRadMean.keys() and Band in MCFG.axis_rad:
                    x = self.S1_FovRadMean[Band][idx]
                    y = self.S2_FovRadMean[Band][idx]
                    if len(x) >= 2:
                        value_min = value_max = None
                        flag = 'Rad'
                        print('rad', Band, len(x), np.min(x), np.max(x),
                              np.min(y), np.max(y))
                        if MCFG.AutoRange == 'ON':
                            value_min = np.min([np.min(x), np.min(y)])
                            value_max = np.max([np.max(x), np.max(y)])
                        elif len(MCFG.axis_rad) != 0:
                            value_min = MCFG.axis_rad[Band][0]
                            value_max = MCFG.axis_rad[Band][1]
                        if value_min is not None and value_max is not None:
                            regression(x, y, value_min, value_max,
                                       flag, ICFG, MCFG, Band)
            if hasattr(self, 'S1_FovTbbMean'):
                if Band in self.S1_FovTbbMean.keys() and Band in MCFG.axis_rad:
                    x = self.S1_FovTbbMean[Band][idx]
                    y = self.S2_FovTbbMean[Band][idx]
                    if len(x) >= 2:
                        value_min = value_max = None
                        flag = 'Tbb'
                        print('tbb', Band, len(x), np.min(x),
                              np.max(x), np.min(y), np.max(y))
                        if MCFG.AutoRange == 'ON':
                            value_min = np.min([np.min(x), np.min(y)])
                            value_max = np.max([np.max(x), np.max(y)])
                        elif len(MCFG.axis_tbb) != 0:
                            value_min = MCFG.axis_tbb[Band][0]
                            value_max = MCFG.axis_tbb[Band][1]
                        if value_min is not None and value_max is not None:
                            regression(x, y, value_min, value_max,
                                       flag, ICFG, MCFG, Band)
        if hasattr(self, 'S1_FovRefMean'):
            a = set(self.S1_FovRefMean.keys())
            b = set(['CH_01', 'CH_02', 'CH_03'])
            c = a.intersection(b)

            if len(c) == 3:
                print '真彩图  显示匹配位置'
                R = self.S1_FovRefMean['CH_03']
                G = self.S1_FovRefMean['CH_02']
                B = self.S1_FovRefMean['CH_01']
                mask = self.MaskFine['CH_01']

                # 把匹配点进行补点，图像好看
                for i in xrange(3):
                    fill_points_2d(R, -999.)
                    fill_points_2d(G, -999.)
                    fill_points_2d(B, -999.)

                MainPath, _ = os.path.split(ICFG.ofile)
                if not os.path.isdir(MainPath):
                    os.makedirs(MainPath)
                file_name = '%s+%s_%s_Map321.png' % (
                    ICFG.sat1, ICFG.sensor1, ICFG.ymd)
                path_name = os.path.join(MainPath, file_name)
                dv_img.dv_rgb(R, G, B, path_name, 2, 1, mask)
