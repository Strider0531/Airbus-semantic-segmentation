TRAIN_BATCH_SIZE = 48
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'              # Set 'DECONV' for Conv2DTranspose layer. Simple = UpSampling2D.

NET_SCALING = (1, 1)                  # Downsampling inside the network
IMG_SCALING = (3, 3)                  # Downsampling in preprocessing
VALID_IMG_COUNT = 900                 # Number of validation images to use
MAX_TRAIN_STEPS = 9                   # Maximum number of steps_per_epoch in training

MAX_TRAIN_EPOCHS = 99
PATIENCE = 10                         # Patience before early stopping
TRAIN_DATA_FOLDER = 'train_v2'        # Folder contain training set of images inside data/ folder
TEST_DATA_FOLDER = 'test_v2'          # Folder contain testing set of images inside data/ folder