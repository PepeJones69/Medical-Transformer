import cv2
import os
import numpy as np
from scipy.ndimage import uniform_filter

image_dir = "data_original/val/img"
target_dir = "data_original/val/img"
method = "crop"
dim = 256, 256

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

fnames = os.listdir(image_dir)

if method == "mov_avg":
    first_img = cv2.imread(os.path.join(image_dir, fnames[0]))
    y, x, c = first_img.shape
    all_imgs = np.empty((len(fnames), y, x, c), dtype=np.uint8)

    for i, fname in enumerate(fnames):
        img = cv2.imread(os.path.join(image_dir, fname))
        all_imgs[i, ...] = img

    conv_all_imgs = uniform_filter(all_imgs, size=(7, 0, 0, 0), mode="reflect")

    for i, fname in enumerate(fnames):
        img_denoised = conv_all_imgs[i]
        img_denoised = cv2.resize(img_denoised, dim)
        cv2.imwrite(os.path.join(target_dir, fname), img_denoised)


elif method == "bilateral":
    for i, fname in enumerate(fnames):
        img = cv2.imread(os.path.join(image_dir, fname))
        img_denoised = cv2.bilateralFilter(img, 9, 40, 100)
        img_denoised = cv2.resize(img_denoised, dim)
        cv2.imwrite(os.path.join(target_dir, fname), img_denoised)

elif method == "crop":
    for i, fname in enumerate(fnames):
        img = cv2.imread(os.path.join(image_dir, fname))
        img_cropped = img[:, 150:950, :]
        cv2.imwrite(os.path.join(target_dir, fname), img_cropped)

else:
    for i, fname in enumerate(fnames):
        img = cv2.imread(os.path.join(image_dir, fname))
        img = cv2.resize(img, dim)
        cv2.imwrite(os.path.join(target_dir, fname), img)

