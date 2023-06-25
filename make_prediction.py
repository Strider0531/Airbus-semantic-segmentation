import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

from skimage.morphology import binary_opening, disk
from keras.models import load_model
from tkinter import filedialog
from skimage.io import imread
from utils import masks_as_color, multi_rle_encode

# Ask for a file
picture = filedialog.askopenfile(mode='r', title='Select activity picture', defaultextension=".jpg")

# Stop executing script if open canceled
if picture is None:
    print("Open canceled")
    exit()
else:
    folder_dir = 'prediction/input'
    pic_name = 'example.jpg'  # change to desc_to_img_name() later
    path = os.path.join(folder_dir, f'{pic_name}')

    # Making the folder if it does not exist yet
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

# Copy file to folder
shutil.copy(picture.name, path)

# Open U-net model
fullres_model = load_model('fullres_model.h5')

# Make original pictures + predicted mask
fig, m_axs = plt.subplots(1, 2, figsize=(15, 6))

test_image = imread('prediction/input/example.jpg')
test_image = np.expand_dims(test_image, 0)/255.0

m_axs[0].imshow(test_image[0])
m_axs[0].set_title('Image: ' + 'test.jpg')

model_pred = imread('prediction/input/example.jpg')
model_pred = np.expand_dims(model_pred, 0)/255.0
model_pred = fullres_model.predict(model_pred)[0]

def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

reencoded = masks_as_color(multi_rle_encode(smooth(model_pred)[:, :, 0]))
m_axs[1].imshow(reencoded)
m_axs[1].set_title('Predicted mask')

# Save resulted pictures
fig.savefig('prediction/output/original_img+predicted_mask.png')