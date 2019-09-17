import cv2
import os
import sys
import numpy as np
# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
from skimage.feature import local_binary_pattern

OUTPUTS = './Outputs'
PATH = './Hands'
TEST= './test'
RADIUS = 3
NEIGHBORS = 8
eps = 0.0000001
FEATURES = os.path.join(OUTPUTS, './lbp_features.csv')
ORDER = os.path.join(OUTPUTS, './lbp_order.csv')

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
    windows = change_shape(os.path.join(PATH,filename))

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

def main():
    hists = []
    try:
        file = sys.argv[1]
    except:
        file = '0000000'
    # print(file)
    files = []
    if file == '0000000':
        for filename in os.listdir(PATH):
            if filename.endswith('.jpg'):
                files.append(filename)
                print(filename)
                # filename = os.path.join(path,filename)
                hists.append(lbp_and_show(filename, RADIUS, NEIGHBORS))
    else:
        filename = 'Hand_'+sys.argv[1]+'.jpg'
        files = [filename]
        hists.append(lbp_and_show(filename, RADIUS, NEIGHBORS))
        print(hists)
        # hists = hists.reshape(1920,)
    hists = np.array(hists)
    if len(hists) == 1:
        print(hists)
    else:
        print(hists[:10])
    print('Number of hists: ',hists.shape)
    print('Length of each histogram: ',hists[0].shape)
    # temp = np.array([])
    temp = []
    for hist in hists:
        temp.append(hist.ravel())
    # print(len(temp[0]))
    hists = np.array(temp)
    # df = pd.DataFrame(hists)
    # print(df.head())
    print('Number of histograms: ',hists.shape)
    # if len(hists)<100:
        # FEATURES = os.path.join(OUTPUTS, './lbp_small_features.csv')
    np.savetxt(FEATURES, hists, delimiter=',')
    with open(ORDER, 'w') as f:
        for item in files:
            f.write("%s\n" % item)
    # with file('lbp_features.txt', 'w') as outfile:
    #     for slice in hists:
    #         np.savetxt(outfile, slice)

if __name__ == '__main__':
    main()