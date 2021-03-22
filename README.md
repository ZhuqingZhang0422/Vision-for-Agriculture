# Vision-for-Agriculture
Vision for Agriculture Segmenting and classifying aerial images of US farmland
Dataset intro
The challenge dataset contains 21,061 aerial farmland images captured throughout 2019 across the US. Each image consists of four 512x512 color channels, which are RGB and Near Infra-red (NIR). Each image also has a boundary map and a mask. The boundary map indicates the region of the farmland, and the mask indicates valid pixels in the image. Regions outside of either the boundary map or the mask are not evaluated.

This dataset contains six types of annotations: Cloud shadow, Double plant, Planter skip, Standing Water, Waterway, and Weed cluster. These types of field anomalies have great impacts on the potential yield of farmlands, therefore it is extremely important to accurately locate them. In the Agriculture-Vision dataset, these six patterns are stored separately as binary masks due to potential overlaps between patterns. Users are free to decide how to use these annotations.

Each field image has a file name in the format of (field id)_(x1)-(y1)-(x2)-(y2).(jpg/png). Each field id uniquely identifies the farmland that the image is cropped from, and (x1, y1, x2, y2) is a 4-tuple indicating the position in which the image is cropped. Please refer to our paper for more details regarding how we construct the dataset.

Your task
Using the dataset described above your task is to train a model to predict field anomalies on new images. Given a new input from the test set your task is to predict what class does each pixel belong to (one of the six anomalies or the background).

Submission
This year your submissions will not go through the Kaggle website. Due to issues with the privacy of the test set, you will use a platform codalab: https://competitions.codalab.org/competitions/23732

It is straightforward to use - you need to register your team and you can upload your predictions (up to 2 per day and 999 in total). You will see your scores instantly. Make sure your team name on codalab is in the following format "comp540_netid_netid_*". This will be important for us to be able to see your results, keep track of your progress and extract the class leaderboard. Also beware - the scoring process on codalab takes about 6h!

The server expects you to format the predictions in the following way:

- for each image in the test set you should produce a prediction image in .png format, with filename **field-id_x1-y1-x2-y2.png**. The image ID 'field-id_x1-y1-x2-y2' must match the ID of the predicted image exactly. Each png file in your submission will be loaded using

numpy.array(PIL.Image.open(‘field-id_x1-y1-x2-y2.png’))

So we recommend you save the predictions using

PIL.Image.fromarray(pred).save(‘field-id_x1-y1-x2-y2.png’)

- the prediction image should have the same size as the input image and each pixel should contain a label from 0-6 indicating the anomaly type/background. Specifically:

0 - background

1 - cloud_shadow

2 - double_plant

3 - planter_skip

4 - standing_water

5 - waterway

6 - weed_cluster

This label order will be strictly followed during evaluation.

- then you should zip the folder with prediction images and upload it to codalab.

Evaluation metric
We use mean Intersection-over-Union (mIoU) as our main quantitative evaluation metric, which is one of the most commonly used measures in semantic segmentation datasets. The mIoU is computed as:



Where c is the number of annotation types (c = 7 in our dataset, with 6 patterns + background), Pc and Tc are the predicted mask and ground truth mask of class c respectively.

Since our annotations may overlap, we modify the canonical mIoU metric to accommodate this property. For pixels with multiple labels, a prediction of either label will be counted as a correct pixel classification for that label, and a prediction that does not contain any ground truth labels will be counted as an incorrect classification for all ground truth labels.

A more detailed explanation can be found here: https://github.com/SHI-Labs/Agriculture-Vision .

First steps
We want to encourage you to take the following steps first:

1) Check out the "Data" tab, download the data and get familiar with the available dataset. More details about the dataset are under the Data tab.

2) Check out the following page where the original competition instructions are placed: https://github.com/SHI-Labs/Agriculture-Vision

3) Make an account on codalab and register your team here: https://competitions.codalab.org/competitions/23732

4) Start experimenting with simple models and get your submissions up on the server

IMPORTANT: make sure your team name on codalab is in the following format "comp540_netid_netid_*". This will be important for us to be able to see your results and extract the class leaderboard!
