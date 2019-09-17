import cv2
import os
import sys
import numpy as np
# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
from skimage.feature import local_binary_pattern

OUTPUTS = './Outputs'
PATH = './Hands33'
RADIUS = 3
NEIGHBORS = 8
eps = 0.0000001
TEST= './test'
FEATURES = os.path.join(OUTPUTS, './lbp_features.csv')
ORDER = os.path.join(OUTPUTS,'./lbp_order.csv')


def change_shape(img):
    img_bgr = cv2.imread(img, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    windows = []
    for i in range(0, img_gray.shape[0], 100):
        new_shape = []
        for j in range(0, img_gray.shape[1], 100):
            # new_shape = []
            part = img_gray[i:i + 100, j:j + 100]
            new_shape.append(part)
        windows.append(new_shape)
    # print(windows)
    return np.array(windows)


def lbp_and_show(filename, RADIUS, NEIGHBORS):
    windows = change_shape(os.path.join(TEST,filename))

    lbps = []
    hists = []
    for window in windows:
        # print(len(windows))
        for hsquare in window:
            feat = local_binary_pattern(hsquare, NEIGHBORS, RADIUS, method='uniform')
            lbps.append(feat)
            hist, _ = np.histogram(feat.ravel(), bins=np.arange(0, NEIGHBORS + 3), range=(0, NEIGHBORS + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hists.append(hist)
    return hists

def compare(data, test):
    # print(len(data), len(test))
    res = []
    for i in range(len(data)):
        temp = 0
        for j in range(len(data[0])):
            temp += (test[j] - data[i][j])**2
        temp = temp**.5
        res.append(temp)
    res /= (sum(res)+eps)
    return res

def get_rank(res2):
    lines = [line for line in open(ORDER, 'r')]
    for i in range(len(lines)):
        res2[i] = lines[res2[i]]
    # print(res2)
    return res2

def main():
    filename = 'Hand_'+sys.argv[1]+'.jpg'
    # files = [filename]
    hist = lbp_and_show(filename, RADIUS, NEIGHBORS)
    hist = np.ravel(np.array(hist))

    data = np.loadtxt(FEATURES, delimiter=',')
    # print('Data shape: ',data.shape)
    res = compare(data, hist)
    # print(res)
    dummy = np.sort(res)
    try:
        number_of_results = min(len(res), int(sys.argv[2]))
    except:
        number_of_results = len(res)
    res2 = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    res2 = get_rank(res2)
    for i in range(len(dummy)):
        x = np.nonzero(res==dummy[i])[0][0]
        dummy[i] = res[x]
    for i in range(number_of_results):
        print(i+1, res2[i], 1-dummy[i], '\n')


if __name__ == '__main__':
    main()