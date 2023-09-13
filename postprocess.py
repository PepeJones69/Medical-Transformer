import os
import pickle
import numpy as np
import skimage.io
from skimage import morphology

FILE_PATH = 'runs/test/0/'
ROI_PATH = 'data_test/roi/'
SAVE_PATH = 'runs/test/destretched'
os.makedirs(SAVE_PATH, exist_ok=True)


def destretch(label_filename, roi_filename):
    with open(roi_filename, 'rb') as fp:
        roi_start_list_smooth = pickle.load(fp)
    label = skimage.io.imread(label_filename)
    label = skimage.transform.resize(label, (256, 1100), order=0, preserve_range=True)
    label = label.astype(np.uint8)

    label_fin = np.ones((800, 1100)) * 255
    label_fin = label_fin.astype(np.uint8)
    for i, roi_start in enumerate(roi_start_list_smooth):
        label_fin[roi_start:roi_start+256, i] = label[:, i]

    return label_fin
    

for img_filename in sorted(os.listdir(FILE_PATH)):
    if img_filename.endswith('.png'):
        label_filename = os.path.join(FILE_PATH, img_filename)
        roi_filename = ROI_PATH + img_filename[:-4]
        label = destretch(label_filename, roi_filename)

        label_mor = morphology.label(label, background=255)
        roi_labels = np.argsort(np.unique(label_mor, return_counts=True)[1])

        if roi_labels.shape[0] > 4:
            for noise in roi_labels[:-4]:
                mask = (label_mor == noise)
                dilated = morphology.binary_dilation(mask)
                neighbors_temp, neighbor_counts = np.unique(label[np.logical_xor(mask, dilated)], return_counts=True)
                best_neighbor = neighbors_temp[np.argmax(neighbor_counts)]
                label[mask] = best_neighbor
                skimage.io.imsave(SAVE_PATH+img_filename, label)
        else:
            skimage.io.imsave(SAVE_PATH+img_filename, label)
print("De-stretched to " + SAVE_PATH)
