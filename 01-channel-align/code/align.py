from numpy import array, dstack, roll
from skimage.transform import rescale
from skimage.io import imshow
from math import *
from copy import deepcopy

METRIC_TYPE = 0

def align(bgr_image, g_coord):
    b_row = b_col = r_row = r_col = 0
    height = bgr_image.shape[0] // 3
    channels = channel_division(bgr_image)
    b_eps, r_eps = shift_search(channels)
    bgr_image = dstack((roll(roll(channels[2], r_eps[0], 1), r_eps[1], 0), channels[1],
                        roll(roll(channels[0], b_eps[0], 1), b_eps[1], 0)))
    b_row, b_col = g_coord[0] - b_eps[1] - height, \
                   g_coord[1] - b_eps[0]
    r_row, r_col = g_coord[0] - r_eps[1] + height, \
                   g_coord[1] - r_eps[0]
    return bgr_image, (b_row, b_col), (r_row, r_col)


def channel_division(bgr_image):
    height = bgr_image.shape[0] // 3
    channels = []
    x1, x2 = 0, bgr_image.shape[1]
    y1, y2 = 0, height
    eps_y = height // 20
    eps_x = x2 // 20
    for _ in range(3):
        channels.append(bgr_image[(y1 + eps_y):(y2 - eps_y),
                        (x1 + eps_x):(x2 - 2 * eps_x)])
        y1 += height
        y2 += height
    return channels


def shift_search(channels):
    height, width = channels[0].shape
    if height < 500 and width < 500:
        r = 15
        range_xy = [[-r, r], [-r, r]]
        if METRIC_TYPE:
            b_eps = MSE_align(channels[1], channels[0], range_xy)
            r_eps = MSE_align(channels[1], channels[2], range_xy)
        else:
            b_eps = cross_cor_align(channels[1], channels[0], range_xy)
            r_eps = cross_cor_align(channels[1], channels[2], range_xy)
    else:
        r = 2
        copy_channels = deepcopy(channels)
        for i in range(3):
            copy_channels[i] = rescale(copy_channels[i], 0.5)
        b_eps, r_eps = shift_search(copy_channels)
        b_eps = tuple(map(lambda x: x * 2, b_eps))
        r_eps = tuple(map(lambda x: x * 2, r_eps))
        range_blue = [[b_eps[0] - r, b_eps[0] + r],
                      [b_eps[1] - r, b_eps[1] + r]]
        range_red = [[r_eps[0] - r, r_eps[0] + r],
                      [r_eps[1] - r, r_eps[1] + r]]
        if METRIC_TYPE:
            b_eps = MSE_align(channels[1], channels[0], range_blue)
            r_eps = MSE_align(channels[1], channels[2], range_red)
        else:
            b_eps = cross_cor_align(channels[1], channels[0], range_blue)
            r_eps = cross_cor_align(channels[1], channels[2], range_red)
    return (b_eps, r_eps)


def MSE_align(img1, img2, range_xy):
    optimal = 1e6
    eps_x, eps_y = 0, 0
    for x in range(range_xy[0][0], range_xy[0][1] + 1):
        for y in range(range_xy[1][0], range_xy[1][1] + 1):
            cur = MSE_metric(img1, img2, x, y)
            if cur < optimal:
                optimal = cur
                eps_x, eps_y = x, y
    return (eps_x, eps_y)


def MSE_metric(img1, img2, x, y):
    img2_shift = roll(roll(img2, y, 0), x, 1)
    img2_intersec = img2_shift[max(0, y):min(img2_shift.shape[0], img2_shift.shape[0] + y),
                    max(0, x):min(img2_shift.shape[1], img2_shift.shape[1] + x)]
    img1_intersec = img1[max(0, y):min(img2_shift.shape[0], img2_shift.shape[0] + y),
                    max(0, x):min(img2_shift.shape[1], img2_shift.shape[1] + x)]
    return ((img1_intersec - img2_intersec) ** 2).sum() / img1_intersec.size


def cross_cor_align(img1, img2, range_xy):
    optimal = 0
    eps_x, eps_y = 0, 0
    for x in range(range_xy[0][0], range_xy[0][1] + 1):
        for y in range(range_xy[1][0], range_xy[1][1] + 1):
            cur = cross_cor_metric(img1, img2, x, y)
            if cur >= optimal:
                optimal = cur
                eps_x, eps_y = x, y
    return (eps_x, eps_y)


def cross_cor_metric(img1, img2, x, y):
    img2_shift = roll(roll(img2, y, 0), x, 1)
    img2_intersec = img2_shift[max(0, y):min(img2_shift.shape[0], img2_shift.shape[0] + y),
                    max(0, x):min(img2_shift.shape[1], img2_shift.shape[1] + x)]
    img1_intersec = img1[max(0, y):min(img2_shift.shape[0], img2_shift.shape[0] + y),
                    max(0, x):min(img2_shift.shape[1], img2_shift.shape[1] + x)]
    return (img1_intersec * img2_intersec).sum() / \
           sqrt((img1_intersec ** 2).sum() * (img2_intersec ** 2).sum())

