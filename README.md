# Weather Image Recognition
###### Erbo Shan 

									
#### This project includes four .py file
* WebCam.py trains the data set and does the prediction without relabling and normalization
* WebCamNormalized.py trains the data set and does the prediction with normalization
* Relable.py trains the data set and does relabling and normalization.
* predict.py trains the data set and takes your input image to predict the weather and time by using greyscale method and              	 colourful image method.
    ```
    python3 predict.py katkam-20170203150000.jpg 
    ```
    
#### If you are intresting in using Tensorflow in this project :

 * Download **retrained_graph.pb**
 * Download tf_files.zip and uncompress it
 * Open terminal
 * Get testing script
   ```python
   curl -L https://goo.gl/3lTKZs > label_image.py
   ```
 
 * Type this command to test
   ```python
   python label_image.py tf_files/Cloudy/Cloudy329.jpg
   ```
 
 * You will see something like this:
 ```python
 	mostlycloudy (score = 0.34056)
 	clear (score = 0.14641)
 	mainly clear (score = 0.14396)
 	cloudy (score = 0.12643)
 	rain showers (score = 0.08014)
 	drizzle (score = 0.06460)
 	moderate rain (score = 0.03207)
 	rain (score = 0.03091)
 	fog (score = 0.01696)
 	rain fog (score = 0.01162)
 	snow (score = 0.00633)
 ```
 
>* We have already changed the name of image by its label(weather)[if there is no related weather from CSV file, we just ignore them], and classified them into different labeled(weather) folders.
 ```
 eg. if we can find a specific weather for katkam-20160605080000.jpg Let's say it is Rain. 
 The name of this images will be changed into Rainxx.jpg
 ```
 
>* If you want to rename each image. The function below will help (uncommend it in **WebCam.py** ):+1:
    ```
    # renaming(FileName, Classifier)
    ```
    
#### If you want to retrain you own data set

* Tensorflow is required, Docker is strongly recommended.
* Work under the same directory with tf_files

* Download traing script with this command
```
curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```
* Start image retraining with this command and you will get your own **retrained_graph.pb** file
```
python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=tf_files
  ```
  
#### Stuffs you may want to use for this(References):
 * [Tensorflow](https://www.tensorflow.org) for Trainning data
 * [Dcoker](https://www.docker.com) for training data on container
 * [References](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0)

