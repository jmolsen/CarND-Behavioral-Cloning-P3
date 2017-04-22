
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Loss Graph"
[image2]: ./center_2017_04_16_14_48_37_606.jpg "Forward Center Image"
[image3]: ./left_2017_04_16_14_48_37_606.jpg "Forward Left Image"
[image4]: ./right_2017_04_16_14_48_37_606.jpg "Forward Rigth Image"
[image5]: ./center_2017_04_16_14_56_13_436.jpg "Reverse Center Image"
[image6]: ./left_2017_04_16_14_56_13_436.jpg "Reverse Left Image"
[image7]: ./right_2017_04_16_14_56_13_436.jpg "Reverse Right Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run3.mp4 containing a video of my model driving the car around track 1
* writeup_report.md summarizing the results

Additional files of a model which drives track 2:
* model_4.py containing the script to create and train the model
* model_4.h5 containing a trained convolutional neural network
* run4.mp4 containing a video of my model driving the car around track 2

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2 preprocessing steps, 3 convolutional layers with 5x5 filter sizes and depths between 24 and 48 with a stride of 2, 2 more convolutional layers with 3x3 filter sizes and depths of 64, a flatten step, and 4 fully connected layers with outputs of 100, 50, 10, and 1. (model.py lines 81-94).

The model includes ReLU activation functions on each of the convolutional layers to introduce nonlinearity (code lines 85-89), and the data is normalized in the model using a Keras lambda layer (code line 83). Also, the images are cropped to remove as much irrelevant data as possible such as the hood of the car and things beyond the road like hills, the sky, and trees use a cropping layer (code line 84).

####2. Attempts to reduce overfitting in the model

To prevent overfitting, the model was trained and validated on different data sets captured when I drove the track a couple of times both forwards and backwards (code line 6-11). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected a combination of center lane driving on both tracks and reverse course driving on both tracks.  For my successful model I ended up only using the forward and reverse course driving on track 1.  I used the images from all three cameras with a steering correction for the left and right cameras.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to generate some test data, start with a known model, check training and validation loss, and make changes to respond to high training and/or validation loss.  When both losses were high, I would increase the number of epochs; when the training loss was low, and the validation loss was high, I experimented with dropout layers and collecting more data.

My first step was to use a convolution neural network model similar to the NVIDIA Architecture presented in the lesson. I thought this model might be appropriate because it was used for training a self-driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model a high mean squared error on both the training and the validation set. This implied that the model was underfitting, so I increased the number of epochs. 

With the additional epochs, I was able to get the loss to around 0.03 for both training and validation.

![loss graph][image1]

I was pretty happy with the loss so I decided to try to drive track 1 autonomously, and it was successful in driving the track without leaving the road.
 

####2. Final Model Architecture

The final model architecture (model.py lines 81-94) consisted of a convolution neural network with the following layers and layer sizes:

* Keras lambda layer to normalize the image data
* Cropping layers to remove irrelevant image data
* 3 Convolutional layers with 5x5 filter sizes, stride of 2, and depths of 24, 36, and 48 - each with a ReLU activation
* 2 Convolutional layers with 3x3 fitler sizes and depths of 64 - each with a ReLU activation
* a Flatten layer
* 4 Fully Connected layers with outputs of 100, 50, 10, and 1


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here are example images of center lane driving from each camera:

![track 1 forward center][image2]
![track 1 forward left][image3]
![track 1 forward right][image4]

I then recorded two laps on track one going in reverse so that the vehicle would learn to turn in both directions since the track had a bit of a left turn bias. Here are example images of reverse driving from each camera:

![track 1 forward center][image5]
![track 1 forward left][image6]
![track 1 forward right][image7]

Then I repeated this process on track two in order to get more data points, and to hopefully be able to successfully train a model to drive track two.

In order to get more variety of data without actually having to drive the course, I augmented the data sat by flipping images and angles, and using images from the left and right cameras with a steering correction of 0.3.

After the collection process, I had 4273 data points from track one and 6454 data points from track two. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs I ended up using was 20 which got both training and validation loss down to around 0.03 as seen in the graph above. I used an adam optimizer so that manually training the learning rate wasn't necessary.


###4. Track Two

Even though I had successfully trained a model to drive track 1 autonomously, I wanted to try to see if I could get a model to drive track 2.

I added in the training data that I collected from driving track 2 both forwards and backwards a couple times.  I noticed that upon training that I was getting a higher mean squared error for the validation set than I was for the training set suggesting overfitting.  So, I added in a dropout layer after the convolutional errors with a dropout rate of 0.7.  Then I noticed that I was getting high squared error on both training and validation so I increased my epochs to 30.  This got the loss down to just below 0.08.  That wasn't as low as I would have liked, but I decided to test it on the track anyway.  To my surprise, it successfully navigated track 2.  Unfortunately, it had some problems on the bridge of track 1.  

I have included model_4.py, model_4.h5, and run4.mp4 as evidence of the successful track 2 run.

I attempted several other tweaks to my model in order to see if I could achieve a single model that successfully ran both tracks, but ran out of time.  


