# coding: utf-8
'''
匹配类
@author: wangpeng
'''
import os
import sys

import h5py
import yaml

from DV import dv_plt
from PB import pb_space
from PB.pb_sat import sun_glint_cal
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

        # 记录文件属性
        self.obrit_direction1 = None
        self.obrit_num1 = None
        self.obrit_direction2 = None
        self.obrit_num2 = None

        # 全部匹配点和粗匹配点掩码信息记录,每个通道的精匹配掩码信息记录
        self.PubIdx = np.full((row, col), 0, 'i1')
        self.MaskRough = np.full((row, col), 0, 'i1')
        self.MaskFine = {}

        # SAT1 的经纬度区域信息，使用时初始化
        self.Lon_area1 = None
        self.Lat_area1 = None
        # SAT1 add area data save, Initialization when used
        self.FovDnArea1 = {}
        self.FovRefArea1 = {}
        self.FovTbbArea1 = {}
        self.FovRadArea1 = {}

        # SAT1 的全局变量信息
        self.Time1 = np.full((row, col), self.FillValue, self.dtype)
        self.Lon1 = np.full((row, col), self.FillValue, self.dtype)
        self.Lat1 = np.full((row, col), self.FillValue, self.dtype)
        self.SatA1 = np.full((row, col), self.FillValue, self.dtype)
        self.SatZ1 = np.full((row, col), self.FillValue, self.dtype)
        self.SunA1 = np.full((row, col), self.FillValue, self.dtype)
        self.SunZ1 = np.full((row, col), self.FillValue, self.dtype)
        self.LandCover1 = np.full((row, col), -999, 'i2')
        self.LandSeaMask1 = np.full((row, col), -999, 'i2')

        # SAT1 FOV ENV
        # wangpeng add 2018-06-30
        self.ir_cal_slope1 = {}
        self.cal_coeff1 = {}

        self.FovDnMean1 = {}
        self.FovDnStd1 = {}
        self.FovRefMean1 = {}
        self.FovRefStd1 = {}
        self.FovRadMean1 = {}
        self.FovRadStd1 = {}
        self.FovTbbMean1 = {}
        self.FovTbbStd1 = {}
        self.EnvDnMean1 = {}
        self.EnvDnStd1 = {}
        self.EnvRefMean1 = {}
        self.EnvRefStd1 = {}
        self.EnvRadMean1 = {}
        self.EnvRadStd1 = {}
        self.EnvTbbMean1 = {}
        self.EnvTbbStd1 = {}
        self.SV1 = {}
        self.BB1 = {}

        # SAT2 的全局变量信息
        self.Time2 = np.full((row, col), self.FillValue, self.dtype)
        self.Lon2 = np.full((row, col), self.FillValue, self.dtype)
        self.Lat2 = np.full((row, col), self.FillValue, self.dtype)
        self.SatA2 = np.full((row, col), self.FillValue, self.dtype)
        self.SatZ2 = np.full((row, col), self.FillValue, self.dtype)
        self.SunA2 = np.full((row, col), self.FillValue, self.dtype)
        self.SunZ2 = np.full((row, col), self.FillValue, self.dtype)
        self.LandCover2 = np.full((row, col), -999, 'i2')
        self.LandSeaMask2 = np.full((row, col), -999, 'i2')
        # 光普响应和探元号信息
        self.spec_radiance2 = None  # 记录所有光谱
        self.pixel_num2 = None
        # 记录中心点的画圆的37个像素点经纬度位置
        self.FootLons2 = []
        self.FootLats2 = []

        # SAT2 FOV ENV
        self.FovDnMean2 = {}
        self.FovDnStd2 = {}
        self.FovRefMean2 = {}
        self.FovRefStd2 = {}
        self.FovRadMean2 = {}
        self.FovRadStd2 = {}
        self.FovTbbMean2 = {}
        self.FovTbbStd2 = {}

        self.EnvDnMean2 = {}
        self.EnvDnStd2 = {}
        self.EnvRefMean2 = {}
        self.EnvRefStd2 = {}
        self.EnvRadMean2 = {}
        self.EnvRadStd2 = {}
        self.EnvTbbMean2 = {}
        self.EnvTbbStd2 = {}
        self.SV2 = {}
        self.BB2 = {}

        # 初始化字典内的存放每个通道的数据空间
        for band in BandLst:
            # wangpeng add 2018-06-30
            self.ir_cal_slope1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.MaskFine[band] = np.full((row, col), 0, 'i1')
            self.FovDnMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovDnStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRefMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRefStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRadMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRadStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovTbbMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovTbbStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)

            self.EnvDnMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvDnStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRefMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRefStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRadMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRadStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvTbbMean1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvTbbStd1[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.SV1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.BB1[band] = np.full((row, col), self.FillValue, self.dtype)

            # SAT2 FOV ENV
            self.FovDnMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovDnStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRefMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRefStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRadMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovRadStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovTbbMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.FovTbbStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)

            self.EnvDnMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvDnStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRefMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRefStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRadMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvRadStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvTbbMean2[band] = np.full(
                (row, col), self.FillValue, self.dtype)
            self.EnvTbbStd2[band] = np.full(
                (row, col), self.FillValue, self.dtype)

            self.SV2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.BB2[band] = np.full((row, col), self.FillValue, self.dtype)

    def correct_target_ref_data(self):
        """
        :订正第二颗传感器可见光通道的ref值 
        """
        coeff = np.cos(np.deg2rad(self.SunZ1)) / np.cos(np.deg2rad(self.SunZ2))
        for band in sorted(self.FovRefMean1.keys()):
            if self.FovRefMean2[band] is not None:
                print '订正 sensor2 ref band %s' % band
                idx = np.where(self.FovRefMean2[band] > 0)
                self.FovRefMean2[band][idx] = self.FovRefMean2[
                    band][idx] * coeff[idx]
                idx = np.where(self.EnvRefMean2[band] > 0)
                self.EnvRefMean2[band][idx] = self.EnvRefMean2[
                    band][idx] * coeff[idx]

    def reload_data(self, ICFG, MCFG):
        """
        :param ICFG: 输入配置文件
        :param MCFG: 阈值配置文件
        :return:
        """
        i_file = ICFG.ofile
        row = MCFG.row
        col = MCFG.col
        with h5py.File(i_file, 'r') as hdf5File:
            global_keys = hdf5File.keys()
            if 'MaskRough' in global_keys:
                self.MaskRough = np.full((row, col), 0, 'i1')
            if 'PubIdx' in global_keys:
                self.PubIdx = _get_band_dataset('PubIdx', hdf5File)
            self.FileAttrs = hdf5File.attrs
            self.Time1 = _get_band_dataset('S1_Time', hdf5File)
            self.Lon1 = _get_band_dataset('S1_Lon', hdf5File)
            self.Lat1 = _get_band_dataset('S1_Lat', hdf5File)
            self.Lat_area1 = _get_band_dataset('S1_LatArea', hdf5File)
            self.Lon_area1 = _get_band_dataset('S1_LonArea', hdf5File)
            self.SatA1 = _get_band_dataset('S1_SatA', hdf5File)
            self.SatZ1 = _get_band_dataset('S1_SatZ', hdf5File)
            self.SunA1 = _get_band_dataset('S1_SoA', hdf5File)
            self.SunZ1 = _get_band_dataset('S1_SoZ', hdf5File)
            self.LandCover1 = _get_band_dataset('S1_LandCover', hdf5File)
            self.LandSeaMask1 = _get_band_dataset('S1_LandSeaMask', hdf5File)

            self.Time2 = _get_band_dataset('S2_Time', hdf5File)
            self.Lon2 = _get_band_dataset('S2_Lon', hdf5File)
            self.Lat2 = _get_band_dataset('S2_Lat', hdf5File)
            self.SatA2 = _get_band_dataset('S2_SatA', hdf5File)
            self.SatZ2 = _get_band_dataset('S2_SatZ', hdf5File)
            self.SunA2 = _get_band_dataset('S2_SoA', hdf5File)
            self.SunZ2 = _get_band_dataset('S2_SoZ', hdf5File)
            self.LandCover2 = _get_band_dataset('S2_LandCover', hdf5File)
            self.LandSeaMask2 = _get_band_dataset('S2_LandSeaMask', hdf5File)
            self.S2_Spec = _get_band_dataset('S2_Spec', hdf5File)
            self.S2_PixelNum = _get_band_dataset('S2_PixelNum', hdf5File)
            self.S2_FootLat = _get_band_dataset('S2_FootLat', hdf5File)
            self.S2_FootLon = _get_band_dataset('S2_FootLon', hdf5File)

            for band in MCFG.chan1:
                self.MaskFine[band] = np.full((row, col), 0, 'i1')
                self.FovDnArea1[band] = _get_band_dataset(
                    'S1_FovDnArea', hdf5File, band)
                self.FovDnMean1[band] = _get_band_dataset(
                    'S1_FovDnMean', hdf5File, band)
                self.FovDnStd1[band] = _get_band_dataset(
                    'S1_FovDnStd', hdf5File, band)

                self.FovRefArea1[band] = _get_band_dataset(
                    'S1_FovRefArea', hdf5File, band)
                self.FovRefMean1[band] = _get_band_dataset(
                    'S1_FovRefMean', hdf5File, band)
                self.FovRefStd1[band] = _get_band_dataset(
                    'S1_FovRefStd', hdf5File, band)

                self.FovRadArea1[band] = _get_band_dataset(
                    'S1_FovRadArea', hdf5File, band)
                self.FovRadMean1[band] = _get_band_dataset(
                    'S1_FovRadMean', hdf5File, band)
                self.FovRadStd1[band] = _get_band_dataset(
                    'S1_FovRadStd', hdf5File, band)

                self.FovTbbArea1[band] = _get_band_dataset(
                    'S1_FovTbbArea', hdf5File, band)
                self.FovTbbMean1[band] = _get_band_dataset(
                    'S1_FovTbbMean', hdf5File, band)
                self.FovTbbStd1[band] = _get_band_dataset(
                    'S1_FovTbbStd', hdf5File, band)

                self.EnvDnMean1[band] = _get_band_dataset(
                    'S1_EnvDnMean', hdf5File, band)
                self.EnvDnStd1[band] = _get_band_dataset(
                    'S1_EnvDnStd', hdf5File, band)
                self.EnvRefMean1[band] = _get_band_dataset(
                    'S1_EnvRefMean', hdf5File, band)
                self.EnvRefStd1[band] = _get_band_dataset(
                    'S1_EnvRefStd', hdf5File, band)
                self.EnvRadMean1[band] = _get_band_dataset(
                    'S1_EnvRadMean', hdf5File, band)
                self.EnvRadStd1[band] = _get_band_dataset(
                    'S1_EnvRadStd', hdf5File, band)
                self.EnvTbbMean1[band] = _get_band_dataset(
                    'S1_EnvTbbMean', hdf5File, band)
                self.EnvTbbStd1[band] = _get_band_dataset(
                    'S1_nvTbbStd', hdf5File, band)
                self.SV1[band] = _get_band_dataset('S1_SV', hdf5File, band)
                self.BB1[band] = _get_band_dataset('S1_BB', hdf5File, band)

                # SAT2 FOV ENV
                self.FovDnMean2[band] = _get_band_dataset(
                    'S2_FovDnMean', hdf5File, band)
                self.FovDnStd2[band] = _get_band_dataset(
                    'S2_FovDnStd', hdf5File, band)
                self.FovRefMean2[band] = _get_band_dataset(
                    'S2_FovRefMean', hdf5File, band)
                self.FovRefStd2[band] = _get_band_dataset(
                    'S2_FovRefStd', hdf5File, band)
                self.FovRadMean2[band] = _get_band_dataset(
                    'S2_FovRadMean', hdf5File, band)
                self.FovRadStd2[band] = _get_band_dataset(
                    'S2_FovRadStd', hdf5File, band)
                self.FovTbbMean2[band] = _get_band_dataset(
                    'S2_FovTbbMean', hdf5File, band)
                self.FovTbbStd2[band] = _get_band_dataset(
                    'S2_FovTbbStd', hdf5File, band)

                self.EnvDnMean2[band] = _get_band_dataset(
                    'S2_EnvDnMean', hdf5File, band)
                self.EnvDnStd2[band] = _get_band_dataset(
                    'S2_EnvDnStd', hdf5File, band)
                self.EnvRefMean2[band] = _get_band_dataset(
                    'S2_EnvRefMean', hdf5File, band)
                self.EnvRefStd2[band] = _get_band_dataset(
                    'S2_EnvRefStd', hdf5File, band)
                self.EnvRadMean2[band] = _get_band_dataset(
                    'S2_EnvRadMean', hdf5File, band)
                self.EnvRadStd2[band] = _get_band_dataset(
                    'S2_EnvRadStd', hdf5File, band)
                self.EnvTbbMean2[band] = _get_band_dataset(
                    'S2_EnvTbbMean', hdf5File, band)
                self.EnvTbbStd2[band] = _get_band_dataset(
                    'S2_nvTbbStd', hdf5File, band)
                self.SV2[band] = _get_band_dataset('S2_SV', hdf5File, band)
                self.BB2[band] = _get_band_dataset('S2_BB', hdf5File, band)

    def save_rough_data(self, P1, P2, D1, D2, modeCfg):
        """
        第一轮匹配，根据查找表进行数据的mean和std计算，并且对全局物理量复制（角度，经纬度，时间等）
        """
        print u'对公共区域位置进行数据赋值......'
        condition = np.logical_and(P1.Lut_row >= 0, P1.Lut_col >= 0)
        idx = np.where(condition)

        # 记录公共匹配点
        self.PubIdx[idx] = 1

        # 根据朝找表 计算原数据行列信息
        i1 = P1.Lut_row[idx]
        j1 = P1.Lut_col[idx]
        i2 = idx[0]
        j2 = idx[1]

        # 开始计算中心点圈的37个点位置
        dfai = modeCfg.S2_Fov_fov
        for i in xrange(self.row):
            for j in xrange(self.col):

                fLon, fLat = pb_space.ggp_footpoint(P1.L_pos2[i, j],
                                                    P1.P_pos2[i, j],
                                                    D2.satZenith[i, j],
                                                    D2.satAzimuth[i, j],
                                                    D2.Lats[i, j],
                                                    D2.Lons[i, j], dfai)

                self.FootLats2.append(fLat)
                self.FootLons2.append(fLon)

        FootSahpe = D2.Lats.shape + (37,)
        self.FootLats2 = np.array(self.FootLats2)
        self.FootLons2 = np.array(self.FootLons2)
        self.FootLats2 = self.FootLats2.reshape(FootSahpe)
        self.FootLons2 = self.FootLons2.reshape(FootSahpe)

        # 存放区域数据的空间维度
        areaShape = (self.row, self.col) + \
            (modeCfg.FovWind1[0], modeCfg.FovWind1[1])
        # 记录成像仪的在圈里的位置信息
        self.Lon_area1 = np.full((areaShape), self.FillValue, self.dtype)
        self.Lat_area1 = np.full((areaShape), self.FillValue, self.dtype)
        area, mean, std = get_area_mean_std(D1.Lons, P1, modeCfg.FovWind1)
        self.Lon_area1[idx] = area
        area, mean, std = get_area_mean_std(D1.Lats, P1, modeCfg.FovWind1)
        self.Lat_area1[idx] = area

        # 记录高光谱文件属性
        LstIdx1 = int(len(D1.obrit_direction) / 2)
        self.obrit_direction1 = D1.obrit_direction[LstIdx1]
        self.obrit_num1 = D1.obrit_num[LstIdx1]
        LstIdx2 = int(len(D2.obrit_direction) / 2)
        self.obrit_direction2 = D2.obrit_direction[LstIdx2]
        self.obrit_num2 = D2.obrit_num[LstIdx2]

        # 保存传感器1 的公共数据信息
        self.Time1[idx] = D1.Time[i1, j1]
        self.Lon1[idx] = D1.Lons[i1, j1]
        self.Lat1[idx] = D1.Lats[i1, j1]
        self.SatA1[idx] = D1.satAzimuth[i1, j1]
        self.SatZ1[idx] = D1.satZenith[i1, j1]
        self.SunA1[idx] = D1.sunAzimuth[i1, j1]
        self.SunZ1[idx] = D1.sunZenith[i1, j1]

        if D1.LandCover is not None:
            self.LandCover1[idx] = D1.LandCover[i1, j1]
        else:
            self.LandCover1 = None
        if D1.LandSeaMask is not None:
            self.LandSeaMask1[idx] = D1.LandSeaMask[i1, j1]
        else:
            self.LandSeaMask1 = None

        # 保存传感器2  的公共数据信息
        self.Time2[idx] = D2.Time[i2, j2]
        self.Lon2[idx] = D2.Lons[i2, j2]
        self.Lat2[idx] = D2.Lats[i2, j2]
        self.SatA2[idx] = D2.satAzimuth[i2, j2]
        self.SatZ2[idx] = D2.satZenith[i2, j2]
        self.SunA2[idx] = D2.sunAzimuth[i2, j2]
        self.SunZ2[idx] = D2.sunZenith[i2, j2]
        self.spec_radiance2 = D2.radiance
        self.pixel_num2 = D2.pixel_num

        if D2.LandCover is not None:
            self.LandCover2[idx] = D2.LandCover[i2, j2]
        else:
            self.LandCover2 = None
        if D2.LandSeaMask is not None:
            self.LandSeaMask2[idx] = D2.LandSeaMask[i2, j2]
        else:
            self.LandSeaMask2 = None

        # 各项值计算 #############
        for Band1 in modeCfg.chan1:
            index = modeCfg.chan1.index(Band1)
            Band2 = modeCfg.chan2[index]
            print Band1, Band2

            # sat1 DN #############
            if Band1 in D1.DN.keys():
                data = D1.DN['%s' % Band1]
                # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                area, mean, std = get_area_mean_std(data, P1, modeCfg.FovWind1)
                # 开辟区域空间存放区域数据
                self.FovDnArea1[Band1] = np.full(
                    (areaShape), self.FillValue, self.dtype)
                self.FovDnArea1[Band1][idx] = area
                self.FovDnMean1[Band1][idx] = mean
                self.FovDnStd1[Band1][idx] = std
                area, mean, std = get_area_mean_std(data, P2, modeCfg.EnvWind1)
                self.EnvDnMean1[Band1][idx] = mean
                self.EnvDnStd1[Band1][idx] = std
            else:
                self.FovDnArea1[Band1] = None
                self.FovDnMean1[Band1] = None
                self.FovDnStd1[Band1] = None
                self.EnvDnMean1[Band1] = None
                self.EnvDnStd1[Band1] = None

            ############# sat1 Ref #############
            if Band1 in D1.Ref.keys():
                data = D1.Ref['%s' % Band1]
                # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                area, mean, std = get_area_mean_std(data, P1, modeCfg.FovWind1)
                self.FovRefArea1[Band1] = np.full(
                    (areaShape), self.FillValue, self.dtype)
                self.FovRefArea1[Band1][idx] = area
                self.FovRefMean1[Band1][idx] = mean
                self.FovRefStd1[Band1][idx] = std
                area, mean, std = get_area_mean_std(data, P2, modeCfg.EnvWind1)
                self.EnvRefMean1[Band1][idx] = mean
                self.EnvRefStd1[Band1][idx] = std
            else:
                self.FovRefArea1[Band1] = None
                self.FovRefMean1[Band1] = None
                self.FovRefStd1[Band1] = None
                self.EnvRefMean1[Band1] = None
                self.EnvRefStd1[Band1] = None

            ############# sat1 Rad #############
            if Band1 in D1.Rad.keys():
                data = D1.Rad[Band1]
                # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                area, mean, std = get_area_mean_std(data, P1, modeCfg.FovWind1)
                self.FovRadArea1[Band1] = np.full(
                    (areaShape), self.FillValue, self.dtype)
                self.FovRadArea1[Band1][idx] = area
                self.FovRadMean1[Band1][idx] = mean
                self.FovRadStd1[Band1][idx] = std
                area, mean, std = get_area_mean_std(data, P2, modeCfg.EnvWind1)
                self.EnvRadMean1[Band1][idx] = mean
                self.EnvRadStd1[Band1][idx] = std
            else:
                self.FovRadArea1[Band1] = None
                self.FovRadMean1[Band1] = None
                self.FovRadStd1[Band1] = None
                self.EnvRadMean1[Band1] = None
                self.EnvRadStd1[Band1] = None

            ############# sat1 Tbb #############
            if Band1 in D1.Tbb.keys():
                data = D1.Tbb['%s' % Band1]
                # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                area, mean, std = get_area_mean_std(data, P1, modeCfg.FovWind1)
                self.FovTbbArea1[Band1] = np.full(
                    (areaShape), self.FillValue, self.dtype)
                self.FovTbbArea1[Band1][idx] = area
                self.FovTbbMean1[Band1][idx] = mean
                self.FovTbbStd1[Band1][idx] = std
                area, mean, std = get_area_mean_std(data, P2, modeCfg.EnvWind1)
                self.EnvTbbMean1[Band1][idx] = mean
                self.EnvTbbStd1[Band1][idx] = std
            else:
                self.FovTbbArea1[Band1] = None
                self.FovTbbMean1[Band1] = None
                self.FovTbbStd1[Band1] = None
                self.EnvTbbMean1[Band1] = None
                self.EnvTbbStd1[Band1] = None

            # sat1 sv和 bb的赋值
            if D1.SV[Band1] is not None:
                self.SV1[Band1][idx] = D1.SV[Band1][i1, j1]
            else:
                self.SV1[Band1] = None

            if D1.BB[Band1] is not None:
                self.BB1[Band1][idx] = D1.BB[Band1][i1, j1]
            else:
                self.BB1[Band1] = None

            # wangpeng add 2018-06-30   add vis ir cal ceoff 25 band
            if D1.cal_coeff1 is not None:
                self.cal_coeff1[Band1] = D1.cal_coeff1[Band1]

            # wangpeng add 2018-06-30
            if D1.cal_coeff2 is not None:
                if Band1 in D1.cal_coeff2.keys():
                    self.ir_cal_slope1[Band1][
                        idx] = D1.cal_coeff2[Band1][i1, j1]
                else:
                    self.ir_cal_slope1[Band1] = None

            ############# sat2 DN #############
            if Band2 in D2.DN.keys():
                data = D2.DN[Band2]
                self.FovDnMean2[Band1][idx] = data[i2, j2]
                self.FovDnStd2[Band1][idx] = 0
                self.EnvDnMean2[Band1][idx] = data[i2, j2]
                self.EnvDnStd2[Band1][idx] = 0
            else:
                self.FovDnMean2[Band1] = None
                self.FovDnStd2[Band1] = None
                self.EnvDnMean2[Band1] = None
                self.EnvDnStd2[Band1] = None

            ############# sat2 Ref #############
            if Band2 in D2.Ref.keys():
                data = D2.Ref[Band2]
                self.FovRefMean2[Band1][idx] = data[i2, j2]
                self.FovRefStd2[Band1][idx] = 0
                self.EnvRefMean2[Band1][idx] = data[i2, j2]
                self.EnvRefStd2[Band1][idx] = 0
            else:
                self.FovRefMean2[Band1] = None
                self.FovRefStd2[Band1] = None
                self.EnvRefMean2[Band1] = None
                self.EnvRefStd2[Band1] = None

            ############# sat2 Rad #############
            if Band2 in D2.Rad.keys():
                data = D2.Rad[Band2]
                self.FovRadMean2[Band1][idx] = data[i2, j2]
                self.FovRadStd2[Band1][idx] = 0
                self.EnvRadMean2[Band1][idx] = data[i2, j2]
                self.EnvRadStd2[Band1][idx] = 0
            else:
                self.FovRadMean2[Band1] = None
                self.FovRadStd2[Band1] = None
                self.EnvRadMean2[Band1] = None
                self.EnvRadStd2[Band1] = None

            ############# sat2 Tbb #############
            if Band2 in D2.Tbb.keys():
                data = D2.Tbb[Band2]
                self.FovTbbMean2[Band1][idx] = data[i2, j2]
                self.FovTbbStd2[Band1][idx] = 0
                self.EnvTbbMean2[Band1][idx] = data[i2, j2]
                self.EnvTbbStd2[Band1][idx] = 0
            else:
                self.FovTbbMean2[Band1] = None
                self.FovTbbStd2[Band1] = None
                self.EnvTbbMean2[Band1] = None
                self.EnvTbbStd2[Band1] = None

            # sat2 sv和 bb的赋值
            if D2.SV[Band2] is not None:
                self.SV2[Band1][idx] = D2.SV[Band2][i2, j2]
            else:
                self.SV2[Band1] = None

            if D2.BB[Band2] is not None:
                self.BB2[Band1][idx] = D2.BB[Band2][i2, j2]
            else:
                self.BB2[Band1] = None

    def save_fine_data(self, modeCfg):
        """
        第二轮匹配，根据各通道的的mean和std计以为，角度和距离等进行精细化过滤
        """

        # 最终的公共匹配点数量
        idx = np.where(self.PubIdx > 0)
        if len(idx[0]) == 0:
            return
        print u'所有粗匹配点数目 ', len(idx[0])

        ############### 计算共同区域的距离差 #########
        disDiff = np.full_like(self.Time1, '-1', dtype='i2')
        a = np.power(self.Lon2[idx] - self.Lon1[idx], 2)
        b = np.power(self.Lat2[idx] - self.Lat1[idx], 2)
        disDiff[idx] = np.sqrt(a + b) * 100.

        idx_Rough = np.logical_and(disDiff < modeCfg.distdif_max, disDiff >= 0)
        idx1 = np.where(idx_Rough)
        print u'1. 距离过滤后剩余点 ', len(idx1[0])

        timeDiff = np.abs(self.Time1 - self.Time2)

        idx_Rough = np.logical_and(idx_Rough, timeDiff <= modeCfg.timedif_max)
        idx1 = np.where(idx_Rough)
        print u'2. 时间过滤后剩余点 ', len(idx1[0])
        ############### 过滤太阳天顶角 ###############
        idx_Rough = np.logical_and(
            idx_Rough, self.SunZ1 <= modeCfg.solzenith_max)
        idx_Rough = np.logical_and(
            idx_Rough, self.SunZ2 <= modeCfg.solzenith_max)
        idx1 = np.where(idx_Rough)
        print u'3. 太阳天顶角过滤后剩余点 ', len(idx1[0])

        ############### 计算耀斑角 ###############
        glint1 = np.full_like(self.SatZ1, -999.)
        glint2 = np.full_like(self.SatZ1, -999.)
#         print 'self.SatA1[idx]=', np.nanmin(self.SatA1[idx]), np.nanmax(self.SatA1[idx])
#         print 'self.SatZ1[idx]=', np.nanmin(self.SatZ1[idx]), np.nanmax(self.SatZ1[idx])
#         print 'self.SunA1[idx]=', np.min(self.SunA1[idx]), np.max(self.SunA1[idx])
# print 'self.SunZ1[idx]=', np.min(self.SunZ1[idx]),
# np.max(self.SunZ1[idx])

        glint1[idx] = sun_glint_cal(
            self.SatA1[idx], self.SatZ1[idx], self.SunA1[idx], self.SunZ1[idx])
        glint2[idx] = sun_glint_cal(
            self.SatA2[idx], self.SatZ2[idx], self.SunA2[idx], self.SunZ2[idx])
        idx_Rough = np.logical_and(idx_Rough, glint1 > modeCfg.solglint_min)
        idx_Rough = np.logical_and(idx_Rough, glint2 > modeCfg.solglint_min)

        idx1 = np.where(idx_Rough)
        print u'4. 太阳耀斑角过滤后剩余点 ', len(idx1[0])

        ############### 角度均匀性 #################
        SatZRaio = np.full_like(self.Time1, 9999)
        SatZ1 = np.cos(self.SatZ1[idx] * np.pi / 180.)
        SatZ2 = np.cos(self.SatZ2[idx] * np.pi / 180.)
        SatZRaio[idx] = np.abs(SatZ1 / SatZ2 - 1.)

        idx_Rough = np.logical_and(idx_Rough, SatZRaio <= modeCfg.angledif_max)
        idx1 = np.where(idx_Rough)
        print u'5. 卫星天顶角均匀性过滤后剩余点 ', len(idx1[0])

        idx_Rough = np.logical_and(
            idx_Rough, self.SatZ1 <= modeCfg.satzenith_max)
        idx1 = np.where(idx_Rough)
        print u'6. FY卫星观测角(天顶角)滤后剩余点 ', len(idx1[0])
        self.MaskRough[idx1] = 1

        # 添加spec, 粗匹配后剩余的点是要记录光谱谱线的。。。2维转1维下标
#         idx_1d = np.ravel_multi_index(idx1, (self.row, self.col))
#
#         if modeCfg.write_spec and self.spec_MaskRough_value is None:
#             # 定义spec_MaskRough_value 然后记录需要保存的谱线
#             self.spec_MaskRough_value = []
#             for i in idx_1d:
#                 self.spec_MaskRough_value.append(self.spec_MaskRough_all[i])
#             self.spec_MaskRough_value = np.array(self.spec_MaskRough_value)
#
#             # 记录根据MaskRough表记录的格点信息
#             self.spec_MaskRough_row = idx1[0]
#             self.spec_MaskRough_col = idx1[1]

        for Band1 in modeCfg.chan1:
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
            if (self.FovRadMean1[Band1] is not None) and (self.FovTbbMean1[Band1]is not None):
                flag = 'ir'
                homoFov1 = np.abs(
                    self.FovRadStd1[Band1] / self.FovRadMean1[Band1])
                homoEnv1 = np.abs(
                    self.EnvRadStd1[Band1] / self.EnvRadMean1[Band1])
                homoFovEnv1 = np.abs(
                    self.FovRadMean1[Band1] / self.EnvRadMean1[Band1] - 1)
                homoValue1 = self.FovRadMean1[Band1]
                homoFov2 = np.abs(
                    self.FovRadStd2[Band1] / self.FovRadMean2[Band1])
                homoEnv2 = np.abs(
                    self.EnvRadStd2[Band1] / self.EnvRadMean2[Band1])
                homoFovEnv2 = np.abs(
                    self.FovRadMean2[Band1] / self.EnvRadMean2[Band1] - 1)
                homoValue2 = self.FovRadMean2[Band1]
            # 如果只有 tbb 就用tbb
            elif (self.FovTbbMean1[Band1] is not None) and (self.FovRadMean1[Band1] is None):
                flag = 'ir'
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
            elif self.FovRefMean1[Band1] is not None:
                flag = 'vis'
                homoFov1 = np.abs(
                    self.FovRefStd1[Band1] / self.FovRefMean1[Band1])
                homoEnv1 = np.abs(
                    self.EnvRefStd1[Band1] / self.EnvRefMean1[Band1])
                homoFovEnv1 = np.abs(
                    self.FovRefMean1[Band1] / self.EnvRefMean1[Band1] - 1)
                homoValue1 = self.FovRefMean1[Band1]
                homoFov2 = np.abs(
                    self.FovRefStd2[Band1] / self.FovRefMean2[Band1])
                homoEnv2 = np.abs(
                    self.EnvRefStd2[Band1] / self.EnvRefMean2[Band1])
                homoFovEnv2 = np.abs(
                    self.FovRefMean2[Band1] / self.EnvRefMean2[Band1] - 1)
                homoValue2 = self.FovRefMean2[Band1]
            #### 云判识关闭状态 ####
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

            #### 云判识开启状态 ####
            else:
                # 晴空判别
                if 'ir' in flag:
                    # 固定通道值用于检测可见晴空和云
                    irValue = self.FovTbbMean1[modeCfg.clear_band_ir]
                    condition = np.logical_and(
                        self.MaskRough > 0, irValue >= modeCfg.clear_min_ir)
                elif 'vis' in flag:
                    # 固定通道值用于检测可见晴空和云
                    visValue = self.FovRefMean1[modeCfg.clear_band_vis]
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

        # 根据卫星性质来命名数据集，固定标识，避免命名烦恼 烦恼 烦恼
        NameHead1 = 'S1_'
        NameHead2 = 'S2_'
        # 创建文件夹
        MainPath, MainFile = os.path.split(ICFG.ofile)
        if not os.path.isdir(MainPath):
            os.makedirs(MainPath)

        # 创建hdf5文件
        h5File_W = h5py.File(ICFG.ofile, 'w')

        if self.obrit_direction1 is not None:
            h5File_W.attrs.create(
                'obrit Direction1', self.obrit_direction1, shape=(1,), dtype='S2')
            h5File_W.attrs.create(
                'obrit Num1', self.obrit_num1, shape=(1,), dtype='i4')
            h5File_W.attrs.create(
                'obrit Direction2', self.obrit_direction2, shape=(1,), dtype='S2')
            h5File_W.attrs.create(
                'obrit Num2', self.obrit_num2, shape=(1,), dtype='i4')

        # wangpeng add 2018-06-30 增加系数文件
        if self.cal_coeff1 is not None:
            h5File_W.attrs.create(
                'cal_ceoff', self.cal_coeff1.values(), dtype='f4')

        # wangpeng add 2018-06-30 增加组属性
        if self.cal_coeff1 is not None:
            for band in MCFG.chan1:
                group = h5File_W.create_group(band)
                group.attrs.create(
                    'cal_ceoff', self.cal_coeff1[band], dtype='f4')

        if MCFG.write_spec:
            dset = h5File_W.create_dataset(
                '%sSpec' % NameHead2, dtype='f4', data=self.spec_radiance2, compression='gzip', compression_opts=5, shuffle=True)
            dset.attrs.create(
                'Long_name', 'Record spectral lines obtained from src dataset', shape=(1,), dtype='S64')
        # 生成 h5,首先写入全局变量
        # 第一颗传感器的全局数据信息
        h5File_W.create_dataset('%sLonArea' % NameHead1, dtype='f4',
                                data=self.Lon_area1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sLatArea' % NameHead1, dtype='f4',
                                data=self.Lat_area1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sLon' % NameHead1, dtype='f4', data=self.Lon1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sLat' % NameHead1, dtype='f4', data=self.Lat1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sTime' % NameHead1, dtype='f4',
                                data=self.Time1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sSatA' % NameHead1, dtype='f4',
                                data=self.SatA1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sSatZ' % NameHead1, dtype='f4',
                                data=self.SatZ1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sSoA' % NameHead1, dtype='f4', data=self.SunA1, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sSoZ' % NameHead1, dtype='f4', data=self.SunZ1, compression='gzip', compression_opts=5, shuffle=True)

        if self.LandCover1 is not None:
            h5File_W.create_dataset('%sLandCover' % NameHead1, dtype='f4',
                                    data=self.LandCover1, compression='gzip', compression_opts=5, shuffle=True)
        if self.LandSeaMask1 is not None:
            h5File_W.create_dataset('%sLandSeaMask' % NameHead1, dtype='f4',
                                    data=self.LandSeaMask1, compression='gzip', compression_opts=5, shuffle=True)

        # 第二颗传感器的全局数据信息
        h5File_W.create_dataset(
            '%sLon' % NameHead2, dtype='f4', data=self.Lon2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sLat' % NameHead2, dtype='f4', data=self.Lat2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sTime' % NameHead2, dtype='f4',
                                data=self.Time2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sSatA' % NameHead2, dtype='f4',
                                data=self.SatA2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sSatZ' % NameHead2, dtype='f4',
                                data=self.SatZ2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sSoA' % NameHead2, dtype='f4', data=self.SunA2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset(
            '%sSoZ' % NameHead2, dtype='f4', data=self.SunZ2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sFootLon' % NameHead2, dtype='f4',
                                data=self.FootLons2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sFootLat' % NameHead2, dtype='f4',
                                data=self.FootLats2, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sPixelNum' % NameHead2, dtype='i1',
                                data=self.pixel_num2, compression='gzip', compression_opts=5, shuffle=True)

        if self.LandCover2 is not None:
            h5File_W.create_dataset('%sLandCover' % NameHead2, dtype='f4',
                                    data=self.LandCover2, compression='gzip', compression_opts=5, shuffle=True)
        if self.LandSeaMask2 is not None:
            h5File_W.create_dataset('%sLandSeaMask' % NameHead2, dtype='f4',
                                    data=self.LandSeaMask2, compression='gzip', compression_opts=5, shuffle=True)

        # 写入掩码属性
        dset = h5File_W.create_dataset(
            'MaskRough', dtype='u2', data=self.MaskRough, compression='gzip', compression_opts=5, shuffle=True)
        dset.attrs.create(
            'Long_name', 'after time and angle collocation', shape=(1,), dtype='S32')
        # 写入公共区域属性
        h5File_W.create_dataset(
            'PubIdx', dtype='u2', data=self.PubIdx, compression='gzip', compression_opts=5, shuffle=True)

        # 写入1通道数据信息
        for Band in MCFG.chan1:
            # 第一颗传感器通道数据
            # wangpeng add 2018-06-30
            if self.ir_cal_slope1[Band] is not None:
                h5File_W.create_dataset('/%s/%sir_cal_slope' % (Band, NameHead1), dtype='f4',
                                        data=self.ir_cal_slope1[Band], compression='gzip', compression_opts=5, shuffle=True)
            if self.SV1[Band] is not None:
                h5File_W.create_dataset('/%s/%sSV' % (Band, NameHead1), dtype='f4', data=self.SV1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
            if self.BB1[Band] is not None:
                h5File_W.create_dataset('/%s/%sBB' % (Band, NameHead1), dtype='f4', data=self.BB1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovDnMean1[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovDnArea' % (Band, NameHead1), dtype='f4', data=self.FovDnArea1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovDnMean' % (Band, NameHead1), dtype='f4', data=self.FovDnMean1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovDnStd' % (Band, NameHead1), dtype='f4', data=self.FovDnStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvDnMean' % (Band, NameHead1), dtype='f4', data=self.EnvDnMean1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvDnStd' % (Band, NameHead1), dtype='f4', data=self.EnvDnStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovRefMean1[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovRefArea' % (Band, NameHead1), dtype='f4',
                                        data=self.FovRefArea1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRefMean' % (Band, NameHead1), dtype='f4',
                                        data=self.FovRefMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRefStd' % (Band, NameHead1), dtype='f4', data=self.FovRefStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRefMean' % (Band, NameHead1), dtype='f4',
                                        data=self.EnvRefMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRefStd' % (Band, NameHead1), dtype='f4', data=self.EnvRefStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovRadMean1[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovRadArea' % (Band, NameHead1), dtype='f4',
                                        data=self.FovRadArea1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRadMean' % (Band, NameHead1), dtype='f4',
                                        data=self.FovRadMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRadStd' % (Band, NameHead1), dtype='f4', data=self.FovRadStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRadMean' % (Band, NameHead1), dtype='f4',
                                        data=self.EnvRadMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRadStd' % (Band, NameHead1), dtype='f4', data=self.EnvRadStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovTbbMean1[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovTbbArea' % (Band, NameHead1), dtype='f4',
                                        data=self.FovTbbArea1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovTbbMean' % (Band, NameHead1), dtype='f4',
                                        data=self.FovTbbMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovTbbStd' % (Band, NameHead1), dtype='f4', data=self.FovTbbStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvTbbMean' % (Band, NameHead1), dtype='f4',
                                        data=self.EnvTbbMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvTbbStd' % (Band, NameHead1), dtype='f4', data=self.EnvTbbStd1[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            ###################### 第二颗传感器通道数据 ########################
            if self.SV2[Band] is not None:
                h5File_W.create_dataset('/%s/%sSV' % (Band, NameHead2), dtype='f4', data=self.SV2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
            if self.BB2[Band] is not None:
                h5File_W.create_dataset('/%s/%sBB' % (Band, NameHead2), dtype='f4', data=self.BB2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovDnMean2[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovDnMean' % (Band, NameHead2), dtype='f4', data=self.FovDnMean2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovDnStd' % (Band, NameHead2), dtype='f4', data=self.FovDnStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvDnMean' % (Band, NameHead2), dtype='f4', data=self.EnvDnMean2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvDnStd' % (Band, NameHead2), dtype='f4', data=self.EnvDnStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovRefMean2[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovRefMean' % (Band, NameHead2), dtype='f4',
                                        data=self.FovRefMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRefStd' % (Band, NameHead2), dtype='f4', data=self.FovRefStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRefMean' % (Band, NameHead2), dtype='f4',
                                        data=self.EnvRefMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRefStd' % (Band, NameHead2), dtype='f4', data=self.EnvRefStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovRadMean2[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovRadMean' % (Band, NameHead2), dtype='f4',
                                        data=self.FovRadMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovRadStd' % (Band, NameHead2), dtype='f4', data=self.FovRadStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRadMean' % (Band, NameHead2), dtype='f4',
                                        data=self.EnvRadMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvRadStd' % (Band, NameHead2), dtype='f4', data=self.EnvRadStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            if self.FovTbbMean2[Band] is not None:
                h5File_W.create_dataset('/%s/%sFovTbbMean' % (Band, NameHead2), dtype='f4',
                                        data=self.FovTbbMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sFovTbbStd' % (Band, NameHead2), dtype='f4', data=self.FovTbbStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvTbbMean' % (Band, NameHead2), dtype='f4',
                                        data=self.EnvTbbMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
                h5File_W.create_dataset('/%s/%sEnvTbbStd' % (Band, NameHead2), dtype='f4', data=self.EnvTbbStd2[
                                        Band], compression='gzip', compression_opts=5, shuffle=True)

            dset = h5File_W.create_dataset('/%s/MaskFine' % Band, dtype='u2', data=self.MaskFine[
                                           Band], compression='gzip', compression_opts=5, shuffle=True)
            dset.attrs.create(
                'Long_name', 'after scene homogenous collocation', shape=(1,), dtype='S32')

        h5File_W.close()

    def draw_dclc(self, ICFG, MCFG):

        print u'产品绘图'

        for Band in MCFG.chan1:
            idx = np.where(self.MaskFine[Band] > 0)
            if self.FovRefMean1[Band] is not None and Band in MCFG.axis_ref:
                x = self.FovRefMean1[Band][idx]
                y = self.FovRefMean2[Band][idx]
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

            if self.FovRadMean1[Band] is not None and Band in MCFG.axis_rad:
                x = self.FovRadMean1[Band][idx]
                y = self.FovRadMean2[Band][idx]
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

            if self.FovTbbMean1[Band] is not None and Band in MCFG.axis_rad:
                x = self.FovTbbMean1[Band][idx]
                y = self.FovTbbMean2[Band][idx]
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
