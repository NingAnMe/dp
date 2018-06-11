# coding: UTF-8
import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate.interpolate import interp2d
'''
二维矩阵操作
Created on 2016年12月28日

@author: zhangtao
'''


def shrink_2d(array2d, mask, (row_small, col_small), fillValue=-9999.):
    '''
    2维矩阵 等比降分辨率
    array2d  2维矩阵
    mask     无效值掩模矩阵
    row_small, col_small  缩小后的维度
    '''
    assert len(array2d.shape) == 2, \
        "array2d must be 2d array."
    assert array2d.shape == mask.shape, \
        "array2d and musk must have same shape."
    row, col = array2d.shape
    assert row_small <= row and row % row_small == 0, \
        u"row_small 必须整数倍缩小."
    assert col_small <= col and col % col_small == 0, \
        u"row_small 必须整数倍缩小."

    masked_ary = np.ma.masked_where(mask, array2d)
    ary_small = masked_ary.reshape(
        row_small, row // row_small, col_small, col // col_small).mean(3).mean(1)
    return np.ma.filled(ary_small, fillValue)


def interp_2d(array2d, mask, (row_big, col_big),
              offsetx=None, offsety=None):
    '''
    2维矩阵 插值 升分辨率 （只支持小矩阵）
    array2d  2维矩阵
    mask     无效值掩模矩阵
    row_new, col_new： 放大后的维度
    multiplex, offsetx：x轴放大后的位置关系
    multipley, offsety：y轴放大后的位置关系
    '''
    assert len(array2d.shape) == 2, \
        "array2d must be 2d array."
    assert array2d.shape == mask.shape, \
        "array2d and musk must have same shape."
    row, col = array2d.shape
    assert row_big >= row and row_big % row == 0 and row_big / row % 2 == 1, \
        u"row_big 必须奇数倍放大."
    assert col_big >= col and col_big % col == 0 and col_big / col % 2 == 1, \
        u"col_big 必须奇数倍放大."

    multiplex = col_big // col
    multipley = row_big // row
    if offsetx is None:
        offsetx = (multiplex - 1) // 2
    if offsety is None:
        offsety = (multipley - 1) // 2

    x = multiplex * np.arange(col) + offsetx
    y = multipley * np.arange(row) + offsety
    xx, yy = np.meshgrid(x, y)

    # interp1
    idx = ~mask
    func = Rbf(xx[idx], yy[idx], array2d[idx], function="linear")
    x1 = np.arange(col_big)
    y1 = np.arange(row_big)
    xx1, yy1 = np.meshgrid(x1, y1)
    return func(xx1, yy1)

#     # interp2
#     idx = ~mask
#     func = interp2d(xx[idx], yy[idx], array2d[idx])
#     x1 = np.arange(col_big)
#     y1 = np.arange(row_big)
#     return func(x1, y1)


def enlarge_2d(array2d, mask, (row_big, col_big),
               offsetx=None, offsety=None, fillValue=-9999.):
    '''
    2维矩阵 等比扩大格点 升分辨率
    array2d  2维矩阵
    mask     无效值掩模矩阵
    row_new, col_new： 放大后的维度
    multiplex, offsetx：x轴放大后的位置关系
    multipley, offsety：y轴放大后的位置关系
    '''
    assert len(array2d.shape) == 2, \
        "array2d must be 2d array."
    assert array2d.shape == mask.shape, \
        "array2d and musk must have same shape."
    row, col = array2d.shape
    assert row_big >= row and row_big % row == 0 and row_big / row % 2 == 1, \
        u"row_big 必须奇数倍放大."
    assert col_big >= col and col_big % col == 0 and col_big / col % 2 == 1, \
        u"col_big 必须奇数倍放大."

    multiplex = col_big // col
    multipley = row_big // row
    if offsetx is None:
        offsetx = (multiplex - 1) // 2
    if offsety is None:
        offsety = (multipley - 1) // 2

    x = multiplex * np.arange(col) + offsetx
    y = multipley * np.arange(row) + offsety
    xx, yy = np.meshgrid(x, y)
    array2d[mask] = fillValue
    # enlarge
    ret = np.full((row_big, col_big), fillValue)
    ret[yy, xx] = array2d[:, :]
    for n in xrange(offsetx):
        fill_2d(ret, ret == fillValue, 'l')
        fill_2d(ret, ret == fillValue, 'r')
    for n in xrange(offsety):
        fill_2d(ret, ret == fillValue, 'u')
        fill_2d(ret, ret == fillValue, 'd')
    return ret


def fill_2d(array2d, mask, useFrom):
    '''
    2维矩阵无效值补点
    array2d  2维矩阵
    mask     无效值掩模矩阵
    useFrom  u/d/l/r, 用上/下/左/右的点来补点 
    '''
    assert len(array2d.shape) == 2, \
        "array2d must be 2d array."
    assert array2d.shape == mask.shape, \
        "array2d and musk must have same shape."

    condition = np.empty_like(mask)
    # 用上方的有效点补点
    if useFrom == 'up' or useFrom == 'u':
        condition[1:, :] = mask[1:, :] * (~mask)[:-1, :]
        condition[0, :] = False
        index = np.where(condition)
        array2d[index[0], index[1]] = array2d[index[0] - 1, index[1]]

    # 用右方的有效点补点
    elif useFrom == 'right' or useFrom == 'r':
        condition[:, :-1] = mask[:, :-1] * (~mask)[:, 1:]
        condition[:, -1] = False
        index = np.where(condition)
        array2d[index[0], index[1]] = array2d[index[0], index[1] + 1]

    # 用下方的有效点补点
    elif useFrom == 'down' or useFrom == 'd':
        condition[:-1, :] = mask[:-1, :] * (~mask)[1:, :]
        condition[-1, :] = False
        index = np.where(condition)
        array2d[index[0], index[1]] = array2d[index[0] + 1, index[1]]

    # 用左方的有效点补点
    elif useFrom == 'left' or useFrom == 'l':
        condition[:, 1:] = mask[:, 1:] * (~mask)[:, :-1]
        condition[:, 0] = False
        index = np.where(condition)
        array2d[index[0], index[1]] = array2d[index[0], index[1] - 1]


def smooth_2d(array2d, mask):
    m = ~mask
    # set all 'masked' points to 0. so they aren't used in the smoothing
    r = array2d * m
    a = 4 * r[1:-1, 1:-1] + r[2:, 1:-1] + \
        r[:-2, 1:-1] + r[1:-1, 2:] + r[1:-1, :-2]
    # a divisor that accounts for masked points
    b = 4 * m[1:-1, 1:-1] + m[2:, 1:-1] + \
        m[:-2, 1:-1] + m[1:-1, 2:] + m[1:-1, :-2]
    b[b ==
        0] = 1.  # for avoiding divide by 0 error (region is masked so value doesn't matter)
    array2d[1:-1, 1:-1] = a / b


def add_extra_rowcol_around_2d(ary2d, urow=1, drow=1, lcol=1, rcol=1, fillValue=np.NaN):

    row, col = ary2d.shape
    # add extra col
    ary2d = np.c_[
        np.full((row, lcol), fillValue), ary2d, np.full((row, rcol), fillValue)]

    # add extra row
    newcol = col + lcol + rcol
    return np.r_[np.full((urow, newcol), fillValue), ary2d, np.full((drow, newcol), fillValue)]


def rolling_2d_window(ary2d, window=(3, 3)):
    '''
    2维矩阵滑动窗口
    '''
    r, c = window
    shape = (ary2d.shape[0] - r + 1, ary2d.shape[1] - c + 1) + window

    strides = ary2d.strides * 2
    return np.lib.stride_tricks.as_strided(ary2d, shape=shape, strides=strides)


def rolling_2d_window_pro(ary2d, window, d_i, d_j, p_i, p_j):
    '''
    2维矩阵滑动窗口,升级版本，指定部分需要计算的行列和对应投影位置的行列，返回新的投影行列
    '''
    r, c = window
    shape = (ary2d.shape[0] - r + 1, ary2d.shape[1] - c + 1) + window

    strides = ary2d.strides * 2
    ary4d = np.lib.stride_tricks.as_strided(
        ary2d, shape=shape, strides=strides)
    # 根据新窗口大小 和 原窗口大小计算步长
    row, col = ary4d.shape[:2]
    step = (ary2d.shape[0] - ary4d.shape[0]) / 2
    step2 = (ary2d.shape[1] - ary4d.shape[1]) / 2

    # 建立投影位置和新的窗口位置对应关系
    condition = np.logical_and(d_i >= step, d_j >= step2)
    condition = np.logical_and(condition, d_i <= row)
    condition = np.logical_and(condition, d_j <= col)
    idx = np.where(condition)
    di = d_i[idx] - step
    dj = d_j[idx] - step2
    pi = p_i[idx]
    pj = p_j[idx]

    # 计算均值和std
    a = np.nanmean(ary4d[di, dj], (1, 2))
    b = np.nanstd(ary4d[di, dj], (1, 2))

#     return a, b, pi, pj
#     a = np.mean(ary4d[di, dj], (1, 2))
#     b = np.std(ary4d[di, dj], (1, 2))
#     print pi[0], pj[0], d_i[0], d_j[0]
    # 过滤掉nan值
    idx2 = np.where(~np.isnan(a))
    return a[idx2], b[idx2], pi[idx2], pj[idx2]

if __name__ == '__main__':
    a = np.array([-999.])
    print np.nanmean(a)
    print np.nanstd(a)
#     a = np.array([[0, 1], [2, 3], [4, 5]])
#     print a
#     a = np.ma.masked_where(a < 4, a)
#     print a
#     b = interp_2d(a, a < 0, (9, 6))
#     print b
#     c = enlarge_2d(a, a < 0, (9, 10))
#     print c
#
#     print "--------"
#     print shrink_2d(b, b < 0, (3, 5))
#     print shrink_2d(c, c < 0, (3, 5))
#     func = Rbf([1, 3], [1, 3], [[1, 2], [3, 4]], function='linear')
#     print func(np.arange(0, 5), np.arange(0, 5))
