import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os

from skimage.morphology import binary_opening, disk
from keras.models import load_model
from skimage.io import imread

from utils import masks_as_color, multi_rle_encode

# Open U-net model
fullres_model = load_model('fullres_model.h5')

def raw_prediction(img, path='prediction'):
    c_img = imread(os.path.join(path, 'test.jpg'))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, c_img[0]

def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

def predict(img, path='prediction'):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img

fig, m_axs = plt.subplots(1, 2, figsize=(15, 6))
[c_ax.axis('off') for c_ax in m_axs.flatten()]



#first_seg, first_img = raw_prediction('test.jpg')
test_image = imread('prediction/test.jpg')
test_image = np.expand_dims(test_image, 0)/255.0

m_axs[0].imshow(test_image[0])
m_axs[0].set_title('Image: ' + 'test.jpg')

model_pred = imread('prediction/test.jpg')
model_pred = np.expand_dims(model_pred, 0)/255.0
model_pred = fullres_model.predict(model_pred)[0]

reencoded = masks_as_color(multi_rle_encode(smooth(model_pred)[:, :, 0]))
m_axs[1].imshow(reencoded)
m_axs[1].set_title('Predicted mask')

fig.savefig('prediction/predicted_mask.png')