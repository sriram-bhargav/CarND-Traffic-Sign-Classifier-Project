#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: examples/visualization.jpg "Visualization"
[image2]: examples/original.jpg "Pre-Normalization"
[image3]: examples/normalized.jpg "Post-Normalization"
[image4]: test/n1.png "Traffic Sign 1"
[image5]: test/n2.png "Traffic Sign 2"
[image6]: test/n3.png "Traffic Sign 3"
[image7]: test/n4.png "Traffic Sign 4"
[image8]: test/n5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training and test data set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

sklearn.preprocessing's normalize func is used to scale data individually into unit norm. The results were far better than converting to grayscale and using (x - 128)/128 to normalize the data.

Here is an example of a traffic sign image before and after normalizing (scaled x 100 to display).

![alt text][image2]
![alt text][image3]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model has modified LeNet architecture (has an additional conv layer).
My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x50  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x50 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x100  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x200			    |     									|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 3x3x200  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 1x1x200 				    |     									|
| Fully connected		| inputs 200, outputs 100   					|
| RELU					|												|
| Fully connected		| inputs 100, outputs 84    					|
| RELU					|												|
| Fully connected		| inputs 84 , outputs 43    					|
| Softmax				| Cross Entropy									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer. 
- Batch Size was 128. 
- Number of EPOCHS was 50
- mu was 0
- sigma was 0.1
- training rate was 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.955
* test set accuracy of 0.938

If an iterative approach was chosen:
- What was the first architecture that was tried and why was it chosen?
- What were some problems with the initial architecture?
- How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
- Which parameters were tuned? How were they adjusted and why?
- What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I have used iterative approach to get validation accuracy > 0.93. These are steps i followed:
* Started with LeNet neural network.
* Converting images to grayscale didn't help the validation accuracy. So, I sticked with RGB images for training.
* Initially LeNet model was underfitting the data, so i played with epochs and learning rate.
* Increasing learning_rate, made the model osciallate around non-optimal model accuracy (~ 0.85).
* Now my task was to improve training set accuracy, so I played with model hyper parameters like batch_size, epoch to see how it affected training accuracy.
* Finally, modified LeNet architecture by adding convolution and pooling layers; increasing size of each layer (in the network).
* I was able to reach 0.999 training accuracy and 0.88 validation accuracy, which means model is overfitting.
* Adding dropout to fully connected layers, tinkering with Lenet architecture and other hyper parameters helped me reach > 0.93 validation accuracy.

If a well known architecture was chosen:
- What architecture was chosen?
- Why did you believe it would be relevant to the traffic sign application?
- How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

LeNet was chosen because it is straightforward and small, used on 32x32 images, for a similar problem - character recognition. This project used slighlty modified Lenet to classify 32x32 traffic images.

Due to time constraints, I was not able to augment the data (rotation, translation, scale etc.,). That might have increased model accuracy by some amount without changing LeNet architectur (making model training and testing faster :)

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

1. First image, 20 mph sign, has some brown dirt on the image, could have been harder to classify because we are using rgb images.
2. Second image, stop sign, is slightly tilted left.
3. Third image, stay straight or right, has very thin white boundary and looks merging with blue sky in the background.
4. Fourth image, road work, also tilted left, looks similar to Wild animals crossing traffic sign.
5. I actually liked used test image for "Speed limit 70km/h" because it also looks like an image for "Speed limit 120km/h" (even for a naked eye) because of scaled down image and 7 looks like digits 12. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| Speed limit (20km/h)   									| 
| Stop    			| Stop										|
| Go straight or right				| Go straight or right											|
| Road work     		| Road work					 				|
|	Speed limit (70km/h)		| Speed limit (120km/h)     							|


The model correctly guesses 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.8%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Ipython notebook.

For the first image, the model is 100% sure that this is a Speed limit (20km/h) sign (probability of 1), and the image does contain a Speed limit (20km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Speed limit (20km/h) 									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (50km/h)													|
| 0	      			| Speed limit (60km/h)					 				|
| 0				    | Speed limit (70km/h)     							|

For the second image, the model is 100% sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Stop 									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (50km/h)													|
| 0	      			| Speed limit (60km/h)					 				|
| 0				    | Speed limit (70km/h)     							|

For the third image, the model is 100% sure that this is a Go straight or right sign (probability of 1), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Go straight or right 									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (50km/h)													|
| 0	      			| Speed limit (60km/h)					 				|
| 0				    | Speed limit (70km/h)     							|

For the second image, the model is 100% sure that this is a Road work sign (probability of 1), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Road work 									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (50km/h)													|
| 0	      			| Speed limit (60km/h)					 				|
| 0				    | Speed limit (70km/h)     							|

For the second image, the model is 0% sure that this is a Speed limit (70km/h) sign (probability of 0), and the image does not contain a Speed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Speed limit (120km/h) 									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (50km/h)													|
| 0	      			| Speed limit (60km/h)					 				|
| 0				    | Speed limit (70km/h)     							|

