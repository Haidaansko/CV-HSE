import numpy as np
from math import sqrt

def img_energy(img):
    brightness = img[:, :, 0] * 0.299 \
                 + img[:, :, 1] * 0.587 \
                 + img[:, :, 2] * 0.114
    gx, gy = np.gradient(brightness)
    energy = np.sqrt(np.add(np.square(gx * 2), np.square(gy * 2)))
    for y in range(0, img.shape[0]):
        if y == 0:
            energy[y, 0] = sqrt((brightness[1, 0] - brightness[0, 0]) ** 2 + (brightness[y, 1] - brightness[y, 0]) ** 2)
            energy[y, img.shape[1] - 1] = sqrt(
                (brightness[1, img.shape[1] - 1] - brightness[0, img.shape[1] - 1]) ** 2 + (
                    brightness[y, img.shape[1] - 1] - brightness[y, img.shape[1] - 2]) ** 2)
        elif y == img.shape[0] - 1:
            energy[y, 0] = sqrt(
                (brightness[y, 0] - brightness[y - 1, 0]) ** 2 + (brightness[y, 1] - brightness[y, 0]) ** 2)
            energy[y, img.shape[1] - 1] = sqrt(
                (brightness[y, img.shape[1] - 1] - brightness[y - 1, img.shape[1] - 1]) ** 2 + (
                    brightness[y, img.shape[1] - 1] - brightness[y, img.shape[1] - 2]) ** 2)
        else:
            energy[y, 0] = sqrt(
                (brightness[y + 1, 0] - brightness[y - 1, 0]) ** 2 + (brightness[y, 1] - brightness[y, 0]) ** 2)
            energy[y, img.shape[1] - 1] = sqrt(
                (brightness[y + 1, img.shape[1] - 1] - brightness[y - 1, img.shape[1] - 1]) ** 2 + (
                    brightness[y, img.shape[1] - 1] - brightness[y, img.shape[1] - 2]) ** 2)
    for x in range(1, img.shape[1] - 1):
        energy[0, x] = sqrt(
            (brightness[1, x] - brightness[0, x]) ** 2 + (brightness[0, x - 1] - brightness[0, x + 1]) ** 2)
        energy[img.shape[0] - 1, x] = sqrt(
            (brightness[img.shape[0] - 1, x] - brightness[img.shape[0] - 2, x]) ** 2 + (
                brightness[img.shape[0] - 1, x - 1] - brightness[img.shape[0] - 1, x + 1]) ** 2)
    return energy


def seam_carve(img, mode, mask=None):
    direction, action = mode.split()
    energy = img_energy(img)
    if mask is not None:
        energy = np.add(energy, mask * (energy.size * 256))
    energy_map = np.zeros(img.shape[0:2])
    carve_mask = np.zeros(img.shape[0:2])
    if direction == 'horizontal':
        energy_map[0, :] = energy[0, :]
        for y in range(1, energy_map.shape[0]):
            for x in range(0, energy_map.shape[1]):
                energy_map[y, x] = min(
                    energy_map[y - 1, max(x - 1, 0)],
                    energy_map[y - 1, x],
                    energy_map[y - 1, min(x + 1, energy_map.shape[1] - 1)]) + energy[y, x]
        x = list(energy_map[-1, :]).index(min(energy_map[-1, :]))

        if action == 'shrink':
            resized_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]))
            if mask is None:
                resized_mask = None
            else:
                resized_mask = np.zeros((img.shape[0], img.shape[1] - 1))
        else:
            resized_img = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]))
            if mask is None:
                resized_mask = None
            else:
                resized_mask = np.zeros((img.shape[0], img.shape[1] + 1))


        for y in range(energy_map.shape[0] - 1, -1, -1):
            carve_mask[y, x] = 1
            cur_x = x
            if action == 'shrink':
                resized_img[y, :, :] = np.concatenate((img[y, :cur_x, :],
                                                    img[y, (cur_x + 1):, :]), axis=0)
                if mask is not None:
                    resized_mask[y, :] = np.concatenate((mask[y, :cur_x],
                                                      mask[y, (cur_x + 1):]), axis=0)
            else:
                insert = np.array([(img[y, cur_x, :] + img[y, cur_x + 1, :]) / 2])
                resized_img[y, :, :] = np.concatenate((img[y, :(cur_x + 1), :], insert,
                                                    img[y, (cur_x + 1):, :]), axis=0)
                if mask is not None:
                    insert = np.array([mask[y, cur_x]])
                    resized_mask[y, :] = np.concatenate((mask[y, :(cur_x + 1)],
                                                         insert, mask[y, (cur_x + 1):]), axis=0)
            if y:
                if energy_map[y - 1, max(0, x - 1)] <= energy_map[y - 1, x]:
                    x = max(0, x - 1)

                if energy_map[y - 1, min(cur_x + 1, energy_map.shape[1] - 1)] < energy_map[y - 1, x]:
                    x = min(cur_x + 1, energy_map.shape[1] - 1)

    else:
        energy_map[:, 0] = energy[:, 0]
        for x in range(1, energy_map.shape[1]):
            for y in range(0, energy_map.shape[0]):
                energy_map[y, x] = min(energy_map[max(0, y - 1), x - 1],
                                       energy_map[y, x - 1],
                                       energy_map[min(y + 1, energy_map.shape[0] - 1), x - 1]) \
                                   + energy[y, x]
        y = list(energy_map[:, -1]).index(min(energy_map[:, -1]))
        if action == 'shrink':
            resized_img = np.zeros((img.shape[0] - 1, img.shape[1], img.shape[2]))
            if mask is None:
                resized_mask = None
            else:
                resized_mask = np.zeros((img.shape[0] - 1, img.shape[1]))
        else:
            resized_img = np.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]))
            if mask is None:
                resized_mask = None
            else:
                resized_mask = np.zeros((img.shape[0] + 1, img.shape[1]))

        for x in range(energy_map.shape[1] - 1, -1, -1):
            carve_mask[y, x] = 1
            cur_y = y
            if action == 'shrink':
                resized_img[:, x, :] = np.concatenate((img[:cur_y, x, :],
                                                    img[(cur_y + 1):, x, :]), axis=0)
                if mask is not None:
                    resized_mask[:, x] = np.concatenate((mask[:cur_y, x],
                                                      mask[(cur_y + 1):, x]), axis=0)
            else:
                if cur_y + 1 == energy_map.shape[0]:
                    insert = np.array([list(img[cur_y, x, :])])
                else:
                    insert = np.array([list((img[cur_y, x, :] + img[cur_y + 1, x, :]) / 2)])
                resized_img[:, x, :] = np.concatenate((img[:(cur_y + 1), x, :], insert,
                                                    img[(cur_y + 1):, x, :]), axis=0)
                if mask is not None:
                    insert = np.array([mask[cur_y, x]])
                    resized_mask[:, x] = np.concatenate((mask[:(cur_y + 1), x], insert,
                                                         mask[cur_y + 1:, x]), axis=0)
            if x:
                if energy_map[max(0, y - 1), x - 1] <= energy_map[y, x - 1]:
                    y = max(0, y - 1)

                if energy_map[min(cur_y + 1, energy_map.shape[0] - 1), x - 1] < energy_map[y, x - 1]:
                    y = min(cur_y + 1, energy_map.shape[0] - 1)

    return (img, resized_mask, carve_mask)
