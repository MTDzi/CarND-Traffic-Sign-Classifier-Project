# **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/class_imbalance_train.png ""
[image2]: ./img/class_imbalance_valid.png ""
[image3]: ./img/class_imbalance_test.png ""
[image4]: ./img/before_preprocessing.png "Original image"
[image5]: ./img/after_preprocessing.png "Normalized image"
[image6]: ./downloaded_images/original_15.jpg "No vehicles"
[image7]: ./downloaded_images/original_36.jpg "Go straight or right"
[image8]: ./downloaded_images/original_12.jpg "Priority road"
[image9]: ./downloaded_images/original_7.jpg "Speed limit (100km/h)"
[image10]: ./downloaded_images/original_40.jpg "Roundabout mandatory"
[image11]: ./downloaded_images/original_33.jpg "No entry, class frequency"
[image12]: ./downloaded_images/original_17.jpg "Turn right ahead"
[image13]: ./downloaded_images/original_2.jpg "Speed limit (50km/h)"


---
## Writeup

The link to my [project code](https://github.com/MTDzi/CarND-Traffic-Sign-Classifier-Project) where you'll find the `"writeup.md"` file, and the `"Traffic_Sign_Classifier.ipynb"` notebook with the code and analysis.

### Data Set Summary & Exploration

#### Basic summary of the data set

I used the `numpy` library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

In the notebook, for each class in the dataset I've printed out an example image.
I also checked the class imbalance in the training, validation, and test set, because if there were any discrepancies between the sets, because accuracy (as a measure of model performance) strongly depends on the distribution of classes. For example, if the validation set had an over-representation of easier classes than the test set, the accuracies would not be comparable.

Here are the bar charts for class counts:

![alt text][image1]

![alt text][image2]

![alt text][image3]

### Design and Test a Model Architecture

I did NOT convert the images to grayscale. I think color helps people distinguish
between signs, and therefore it might also convey something to the model.

On the other hand, I guess traffic signs are perfectly readable for color-blind people. But still, color carries information, and I didn't want to throw that information away.

Anyway, I did use normalization because from what I've learned in the lecture, gradient-descent-based techniques benefit when the input vectors have similar distributions across all features. Here's an example of an image before preprocessing:

![alt text][image4]

and after:

![alt text][image5]

I did not use image augmentation, because I knew this will lead to a longer learning process, I needed to use less computationally expensive regularization techniques (hence the dropout). But still, at first I thought of flipping the signs horizontally, but then I realized that some (most?) of the signs have horizontal orientation, e.g. all of the "Speed limit (X km/h)" signs are read from left to right.

But I do realize that if I wanted to regularize the model, I might have considered other image augmentation techniques like: ZCA whitening, zooming in and out, (gentle) shearing, and probably many others.


#### Model architecture

I used a standard LeNet architecture, with a few more convolutional kernels, and dropout:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 8 filters, `strides=[1,1,1,1]`, `padding='VALID'`, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| `ksize=[1,2,2,1]`, `strides=[1,2,2,1]`,  outputs 14x14x8 				|
| Convolution 5x5	    | 20 filters, `strides=[1,1,1,1]`, `padding='VALID'`, outputs 10x10x20     									|
| RELU					|												|
| Fully connected		| 120 neurons        									|
| RELU					|												|
| Dropout | `keep_prob=.5` |
| Fully connected		| 84 neurons        									|
| Dropout | `keep_prob=.5` |
| RELU					|												|
| Logits				| 42 output neurons (same number as the number of classes)        									|



#### Training the model

I used the LeNet architecture with as few modifications possible that would get me over the 93% validation set threshold. I don't have a GPU with compute capability >= 3.0, so training truly deep architectures and a grid search for the optimal set of hyperparameters wasn't an option for me.

However, I had to modify the input so that the model would accept RGB images.

I used the Adam optimizer with a learning rate of **0.001**, a batch of size **150**, and with early stopping (with patience 5).

I had to slightly modify the LeNet architecture to cross the 93% barrier, and I describe that in the following section.



#### Finding the solution

I started off with the standard LeNet architecture, a learning rate of 0.01, and a batch size of 150. I wanted to manipulate the learning rate, and keep the batch size fixed, but not too small, so that there was a high chance that in a given batch I would have examples of all the classes. Also, I set the number of epochs to 100, but never intended to run the calculations that long, instead I used early stopping (based on the validation set accuracy).

I was first aiming for a 100% accuracy on the training set, and for that I reduced the learning rate to 0.001, and changed the number of filters in the convolutional layers (from 6 and 16, to 8 and 20).

OK, so I had 100% accuracy on the training set, but my validation accuracy was now at most 88% -- my model was overfitting the training set. This is when I started adding regularization, and I decided to stick to dropout. I dropped the neurons from both fully connected layers (last two hidden layers in my network), with a `keep_prob` of 0.5. This boosted the validation set accuracy up to 96%. WOW!

My final model results were:
* training set accuracy of: **99.7%**
* validation set accuracy of: **95.8%**
* test set accuracy of: **93.0%**
* downloaded images accuracy of: **75.0%**

I used the LeNet architecture because it's a convolutional neural network (CNN) that I knew worked fine on the MNIST dataset. The fact it's convolutional was crucial because they are shift invariant, and the traffic signs are not always perfectly centered.

There's still the question of: "How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?"
The accuracy on the training set doesn't provide enough evidence about the model's performance -- it might have as well learned idiosyncrasies of that dataset alone.

The accuracy on the validation set was helpful for picking the right hyperparameters, but again -- I might have just picked a lucky combination of hyperparameters that for that particular dataset worked well.

The test set accuracy provides much stronger proof that the model generalizes well, that I captured some valuable *features* (in a more general sense of the word) of the traffic signs. But again, that might have worked well only for images that were prepared pretty much the same way as the training and validation data (same lighting, cropping, image quality, weather conditions, etc.).

What was much stronger evidence was the accuracy on the 8 images downloaded from the web (which was 75.0%), which I'm showing in the next section. This set was chosen by me, cropped pretty much randomly "by hand", images were originally of different qualities, etc.




### Test a Model on new images

Here are eight German traffic signs that I found on the web:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

The first image might be confused with some of the speed limit signs, the second one

#### Discussion of model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No vehicles      		| **Priority road**   									|
| Go straight or right     			| Go straight or right 										|
| Speed limit (100km/h)					| **Speed limit (80km/h)**											|
| Roundabout mandatory	      		| Roundabout mandatory					 				|
| No entry			| No entry      							|
| Turn right ahead | Turn right ahead |
| Speed limit (50km/h) | Speed limit (50km/h) |




The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75.0%. That's not so bad, considering that the images looked a bit different than those in the original test set. It's way lower than the 93.0% accuracy on the test set, but note that the distribution of classes was hardly comparable.

#### How certain the model was when predicting on each of the eight new images

The code for making predictions on my final model is located in the **TOP 5 predictions** section.

For the first image,

![alt text][image6]

the model was confused, it isn't sure about any one class (all predictions are lower than 50%), and the right label ("No vehicles") didn't even make it to the top 5 predictions. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 49.34%  | Priority road    									|
| 40.08% | End of all speed and passing limits |
| 6.91% | Speed limit (30km/h) |
| 3.41% |  End of no passing |
| 0.10% | Speed limit (70km/h) |


For the second image,

![alt text][image7]

the model was spot on: it was 100% certain about the correct label "Go straight or right":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100.00% | Go straight or right |
| 0.00% | End of no passing |
| 0.00% | End of all speed and passing limits |
| 0.00% | Roundabout mandatory |
| 0.00% | Turn left ahead |

For the third image,

![alt text][image8]

the model again was very confident ("Priority road" was indeed the right label):

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100.00% | Priority road |
| 0.00% | End of all speed and passing limits |
| 0.00% | End of no passing |
| 0.00% | End of no passing by vehicles over 3.5 metric tons |
| 0.00% | Traffic signals |

For the fourth image,

![alt text][image9]

the model failed to recognize that the image was "Speed limit (100km/h)" and instead "thought" that it was "Speed limit (80km/h)", but it wasn't very confident about it (all predictions <50%):

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 45.30% | Speed limit (80km/h) |
| 39.98% | Speed limit (60km/h) |
| 12.08% | Speed limit (30km/h) |
| 1.26% | Speed limit (50km/h) |
| 0.48% | Road work |

The fifth image,

![alt text][image10]

was easy for the model ("Roundabout mandatory" was indeed the right label):

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100.00% | Roundabout mandatory |
| 0.00% | Go straight or right |
| 0.00% | Dangerous curve to the right |
| 0.00% | End of no passing |
| 0.00% | Keep right |

The sixth image,

![alt text][image11]

 was correctly identified as "No entry":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100.00% | No entry |
| 0.00% | Turn right ahead |
| 0.00% | Stop |
| 0.00% | Speed limit (20km/h) |
| 0.00% | Speed limit (30km/h) |

The seventh image,

![alt text][image12]

was also correctly labeled, the label was: "Turn right ahead":

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100.00% | Turn right ahead |
| 0.00% | Roundabout mandatory |
| 0.00% | Go straight or left |
| 0.00% | Traffic signals |
| 0.00% | Keep right |

And the final, eight image,

![alt text][image13]

was also correctly identified ("Speed limit (50km/h)"):

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 99.74% | Speed limit (50km/h) |
| 0.26% | Speed limit (30km/h) |
| 0.00% | Speed limit (80km/h) |
| 0.00% | Speed limit (60km/h) |
| 0.00% | End of speed limit (80km/h) |
