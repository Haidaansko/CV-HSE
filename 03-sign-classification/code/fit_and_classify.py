import numpy as np
import math
from sklearn.svm import LinearSVC
from skimage.transform import resize

def extract_hog(RGB_img):
    img_size = 64

    RGB_img = resize(RGB_img, (img_size, img_size))
    img = RGB_img[:, :, 0] * 0.299 + RGB_img[:, :, 1] * 0.587 + RGB_img[:, :, 2] * 0.114
    rows, cols = img.shape

    first_col, right = np.hsplit(img, [1])
    right = np.column_stack((right, np.zeros(rows)))
    left, last_col = np.hsplit(img, [-1])
    left = np.column_stack((np.zeros(rows), left))

    Gx = right - left

    first_col.shape = (rows,)
    last_col.shape = (rows,)
    Gx[:, 0] -= first_col
    Gx[:, -1] += last_col

    first_row, down = np.vsplit(img, [1])
    down = np.row_stack((down, np.zeros(cols)))
    up, last_row = np.vsplit(img, [-1])
    up = np.row_stack((np.zeros(cols), up))

    Gy = down - up

    first_row.shape = (cols,)
    last_row.shape = (cols,)
    Gy[0, :] -= first_row
    Gy[-1, :] += last_row

    orientations = np.arctan2(Gy, Gx)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    cell_size = 8
    block_size = 2
    n_bins = 12
    cell_x = img_size // cell_size
    cell_y = img_size // cell_size

    cell_hist = np.zeros((cell_y, cell_x, n_bins))
    n_bins = (np.arange(n_bins + 1) * 2 * math.pi / (n_bins) - math.pi)
    for i in range(cell_y):
        for j in range(cell_x):
            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            if j == cell_x - 1:
                x1 = img.shape[1] - 1
            if i == cell_y - 1:
                y1 = img.shape[0] - 1

            cell_hist[i, j, :], bin_edges = \
                np.histogram(orientations[y0:y1, x0:x1],
                             weights=magnitude[y0:y1, x0:x1],
                             bins=n_bins)

    eps = 1e-6
    for i in range(cell_y - 1):
        for j in range(cell_x - 1):
            sum = 0
            for y in range(block_size - 1):
                for x in range(block_size - 1):
                    sum += np.sum(cell_hist[i + y, j + x, :])

            norm = math.sqrt(sum * sum + eps)

            for y in range(0, block_size - 1):
                for x in range(0, block_size - 1):
                    cell_hist[i + y, j + x, :] /= norm

    cell_hist = np.reshape(cell_hist, cell_hist.size)
    return cell_hist


def fit_and_classify(train_features, train_labels, test_features):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    LinearSVC(C=20, class_weight=None, max_iter=-1, random_state=None, verbose=False)
    return clf.predict(test_features)
