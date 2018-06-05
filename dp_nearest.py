# coding: utf-8
import numpy as np
from numpy import linalg as LA
from scipy.spatial.ckdtree import cKDTree
from pykdtree.kdtree import KDTree
from PB import pb_space

__author__ = 'wangpeng'

'''
Description:  using KD-tree to find best matched points
Author:       wangpeng
Date:         2018-04-18
version:      1.0
Input:
Output:       (^_^)
'''


class dp_nearest():
    '''
    投影公共类
    '''

    def __init__(self):

        # 查找表的行和列，使用Lons2 Lats2的维度进行反回，目前暂时支持2维
        self.Lut_row = None
        self.Lut_col = None

        # 存放 D1距离D2最近中心点的矩形区域中，符合条件的D1位置
        self.Lut_row_if = []
        self.Lut_col_if = []

        # 存放 D1距离D2最近中心点的 整个矩形区域的D1位置
        self.Lut_row_b = []
        self.Lut_row_e = []
        self.Lut_col_b = []
        self.Lut_col_e = []

        self.G_pos1 = None
        self.G_pos2 = None
        self.P_pos2 = None
        self.L_pos2 = None

    def FindNearest(self, G_pos1, G_pos2, P_pos2, L_pos2, HalfFov, window):

        shape1 = G_pos1.shape[:2]
        shape2 = G_pos2.shape[:2]

        # 放到类里
        self.G_pos1 = G_pos1
        self.G_pos2 = G_pos2
        self.P_pos2 = P_pos2
        self.L_pos2 = L_pos2

        # 前提条件
        assert (G_pos2.shape == P_pos2.shape), "G_pos2 P_pos2 must have same shape"
        assert (P_pos2.shape == L_pos2.shape), "P_pos2 L_pos2 must have same shape"

#         # G_pos1  和 G_pos2 转成2维
#         pytree_los = cKDTree(G_pos1.reshape(G_pos1.size / 3, 3))
#         # 返回最近距离和索引，数据大小G_pos2.size的1维长度,里面的值是G_pos1.size的1维值
#         dist, idx = pytree_los.query(G_pos2.reshape(G_pos2.size / 3, 3))

        # G_pos1  和 G_pos2 转成2维 ,使用 kdTree linux并行版本
        pytree_los = KDTree(G_pos1.reshape(G_pos1.size / 3, 3))
        # 返回最近距离和索引，数据大小G_pos2.size的1维长度,里面的值是G_pos1.size的1维值
        dist, idx = pytree_los.query(G_pos2.reshape(G_pos2.size / 3, 3), sqr_dists=False)

        # 把idx中值转成 Lons1.shape的维度
        i1, j1 = np.unravel_index(idx, shape1)

        # 把idx的下标转成Lons2.shape的维度
        idx_idx = np.arange(0, idx.size)
        i2, j2 = np.unravel_index(idx_idx, shape2)

        # 制作以Lons2维度大小的查找表
        self.Lut_row = np.full(shape2, -999, 'i2')
        self.Lut_col = np.full(shape2, -999, 'i2')
        self.Lut_row[i2, j2] = i1
        self.Lut_col[i2, j2] = j1

        # 计算D1距离D2最近中心点的矩形区域中，符合条件的位置

        if P_pos2 is not None:
            # 制作一个矩阵存放 D1距离D2最近中心点的矩形区域中，符合条件的位置
            nc = int(window[0] / 2)
            nLine, nPixel = shape1

            for i in xrange(idx.size):
                xd = j1[i]
                yd = i1[i]
                # 这里注意切片时 a:b 是不包含b的
                xb = 0 if xd - nc < 0 else xd - nc
                xe = nPixel - 1 if xd + nc > nPixel - 1 else xd + nc
                yb = 0 if yd - nc < 0 else yd - nc
                ye = nLine - 1 if yd + nc > nLine - 1 else yd + nc
                xe = xe + 1
                ye = ye + 1
                # 成像仪的矩形区域
                G_pos1_area = G_pos1[yb:ye, xb:xe, :]
                L_pos1_area = G_pos1_area - P_pos2[i2[i], j2[i], :]
                temp = np.dot(L_pos1_area, L_pos2[i2[i], j2[i], :])
                temp = temp / LA.norm(L_pos1_area, axis=2)  # 矢量相乘
                cos_angle = temp / LA.norm(L_pos2[i2[i], j2[i], :])  # 长度
                iy, ix = np.where(cos_angle > HalfFov)

                self.Lut_row_if.append(iy)
                self.Lut_col_if.append(ix)
                self.Lut_row_b.append(yb)
                self.Lut_row_e.append(ye)
                self.Lut_col_b.append(xb)
                self.Lut_col_e.append(xe)

            self.Lut_row_if = np.array(self.Lut_row_if)
            self.Lut_col_if = np.array(self.Lut_col_if)

if __name__ == '__main__':
    print 'test'
