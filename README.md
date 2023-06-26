Airbus ship semantic segmentation
================================================================

Project developed in Python 3.9

Current project configured for airbus ship semantic segmentation.

Airbus Ship dataset https://www.kaggle.com/c/airbus-ship-detection/data

Using of 'Airbus ship segmentation' project:

- Image segmentation:

1. Install requirements;
2. Open make_prediction.py and run it;
3. Chose .jpg image with ship (image must have 768x768 resolution)
4. Resulted image with original image and predicted mask writes to prediction/output folder. 

- U-Net model training:

1. Install requirements;
2. Prepare train and validation pandas dataframes with 'ImageId' and 'EncodedPixels' columns 
(Preparing_train_test_frames.ipynb notebook contains this step);
3. Change configs.py if it's needed
(Batch size, maximum train epochs etc.);  
4. Model's weights, resulted "fullres_model.h5" model and training graph (training_hist_plot.png) will be saved
to the main project folder;

- Other files:

1. Data_analysis.ipynb show short dataset analysis;
2. Observe_results.ipynb show results of model predicting.

Results:
U-Net model trained on current dataset can segment 'simple' images well:
![alt text](https://github.com/Strider0531/Airbus-semantic-segmentation/blob/master/Readme_Files/good_1.jpg?raw=true)

![alt text](https://github.com/Strider0531/Airbus-semantic-segmentation/blob/master/Readme_Files/good_2.jpg?raw=true)

But it's not good for segment ships from image with non-trivial background:
![alt text](https://github.com/Strider0531/Airbus-semantic-segmentation/blob/master/Readme_Files/bad_1.jpg?raw=true)

![alt text](https://github.com/Strider0531/Airbus-semantic-segmentation/blob/master/Readme_Files/bad_2.jpg?raw=true)

For better segmentation it should be added more confident ships masks into train dataset, images with highest resolution and so on.
Also, training process of resulted U-Net model excledes images wthout ships. Adding ones can highly increase performance.
=======================================================================

Main project development resource:
https://www.kaggle.com/code/hmendonca/u-net-model-with-submission#Decode-all-the-RLEs-into-Images  
