# Reproduction of Driver Identiﬁcation Based on Vehicle Telematics Data using LSTM-Recurrent Neural Network by Girma et. al.
## Group 48

Authors:  
Sander Boers - 4670299 - s.h.boers@student.tudelft.nl

Mathijs van Geerenstein - 4598660 - m.r.vangeerenstein@student.tudelft.nl 

Tom Weinans - 4445449 - t.weimans@student.tudelft.nl

## Abstract
The aim of this blogpost is to describe the reproduction of the LSTM-Recurrent Neural Network proposed by Abenezer Girma et. al. [1]. This deep learning model is able to identify drivers by their driving behavior based on vehicle telematics data. Specifically, the proposed method is robust to noise and anomalies in this time-series data. In our attempt to reproduce the results, we show similar trends to Abenezer Girma et.al., but do not fully reproduce their observations. 


## Introduction
Over the years car technology has improved a lot, but the cars security systems did not evolve that much. The amount of car thefts has not dropped over the years. More specifically, the relative part of “digital” car thefts (where no brute force is used, but where the car is unlocked and started by hacking it) has increased. To fight this development, car alarms could operate by detecting who is actually driving the car. If the driving style does not match that of the owner, a car alarm can still identify a thief. The paper from Abenezer Girma et. al.[1] describes a deep learning model, which is able to identify drivers based on vehicle telematics. Specifically, it proposes a model structure that is especially robust to noise and other data anomalies, as is common with car sensors. The paper compares the achieved results on driver identification with more popular models. This blogpost will describe our attempt to reproduce the results achieved in the paper. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image6.png?raw=true)

The proposed method is a deep LSTM model. Long-Short Term Memory (LSTM) models are a class of Recurrent Neural Networks (RNNs) that are able to learn the order dependence in sequence prediction problems. This approach is applicable on driver identification, because there is sequential input data available to do the prediction. The input data includes various internal vehicle sensor readings for a certain time window. The model is a classifier which predicts the driver as output. A top-level overview of the approach is shown in the figure below. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image10.png?raw=true)

## Methods
In this problem of driver identification, we make use of OBD-II data, which is sequential sensor data collected during a driving trip. Each input to the model is a snippet of time-series ODB-II data, which includes a variety of up to 50 sensor data points per time instance. For robustness in the driver prediction, we want a model that has little sensitivity to noise (white gaussian noise) and sensor anomalies (broken sensor, bad connection, etc). 

### LSTM Model
Recurrent Neural Networks (RNNs) are one of the most successful techniques to do time-series classification tasks such as speech recognition. RNNs take sequential input data, and give a certain output based on a decision made by the internal state. The internal state extracts features and can capture temporal dynamics of time-series data. The problem with RNNs, is that it can suffer from the “vanishing gradient” problem whenever the number of internal (hidden) states increases. This results in the RNN being unable to learn. 

The proposed method uses a Long-Short Term Memory (LSTM) network, which does not suffer from this problem. Because of a memory block in the hidden layer, the model can capture long-term dependencies. Because of this memorizing ability, LSTM has shown great performance on time-series tasks. A visual representation of an LSTM memory block is shown in the figure below. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image4.png?raw=true)

### Evaluation
To evaluate the proposed method, the LSTM network is compared with three other methods, namely a fully connected neural network (FCNN), a decision tree model and a random forest model. All models are trained on the same training data. To test the robustness, each model is tested several times, each with more artificial noise added to the test data. Without any noise added to the test data, all models perform similarly in the driver classification task. However, when artificial noise is added, the performance of the FCNN, decision tree and random forest drops steeply, while the LSTM model does not. The results achieved by the paper are shown in the figure below. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image9.png?raw=true)

Additionally, the proposed method is evaluated by training on noisy sensor data, and then testing on noisy sensor data. Again, the LSTM model shows superior performance over the other three methods it is compared with. These results are shown in the figure below. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image5.png?raw=true)

## Results
This section shows our attempts in reproducing the comparison of the proposed LSTM model with three other methods: a fully connected neural network (FCNN), a decision tree and a random forest. 

### Reproduction

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image2.png?raw=true)

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image3.png?raw=true)

### New data
In addition to the reproduced results the choice was made to also check this implementation while training on a different dataset. For this the Vehicular-trace dataset [2] is used. This dataset consists of two parts. The first part has 10 different drivers and the second part has 4 different drivers. In order to compare this new dataset with the original we have used the first part of this dataset with 10 drivers and 21 comparable features. Training the LSTM, FCNN, Decision Tree and Random Forest on this dataset yields the following result for the accuracy vs noise induced data:

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image7.png?raw=true)

Comparing the figure above with the one trained on the original dataset shows similar trends for the FCNN, Decision Tree and Random Forest. THe LSTM however is performing significantly worse with an accuracy that is less than 60% at 2*STD compared to over 70% for the original dataset.

The other figure that needed to be reproduced is a figure comparing the accuracy of LSTM, FCNN, Decision Tree and Random Forest trained and tested on noisy data. The result of training with the new dataset is shown below.


![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image8.png?raw=true)

The accuracy of the four models trained on the new dataset is very similar to the ones trained on the original dataset. The LSTM and CNN are a bit lower while the DT has a slightly higher accuracy but overall the changes are minor and the ratios are very similar.


### New code variant
As part of the reproducibility assignment, a new code variant has been developed. The existing code by Abenezer Girma et.al. is written with the TenserFlow framework. We wanted to see if we could achieve the same results or even better with shorter runtime, with another machine learning framework. To do this, the TensorFlow model is completely rewritten in PyTorch. The PyTorch framework is the standard in the Deep Learning course, that is why this framework has been chosen as the new code variant. 

The new PyTorch model is able to train with the vehicular telematics data and can evaluate results on the testset. However, the model gets stuck in training after the first 10 epochs, not being able to decrease the loss any further. It could be that the PyTorch LSTM layers are different from the TensorFlow layers, or that we suffer from the vanishing gradient problem after all. It could also be because of a flaw in the way the training loop is done, which is a much more difficult task in PyTorch compared to TensorFlow.The resulting accuracy curve is terrible, as is to be expected (see figure below). We were not able to improve on these results within reasonable time. 

![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image12.png?raw=true)

```python
class LSTMmodel(torch.nn.Module):
   def __init__(self):
       super(LSTMmodel, self).__init__()
       self.lstm1 = nn.LSTM(hidden_size=160, input_size=NUM_FEATURES)
       self.batchNorm1 = nn.LazyBatchNorm1d()
       self.dropout1 = nn.Dropout(p=0.2)
       self.lstm2 = nn.LSTM(hidden_size=120, input_size=160)
       self.batchNorm2 = nn.BatchNorm1d(num_features=16)
       self.dropout2 = nn.Dropout(p=0.2)
       self.linear1 = nn.LazyLinear(NUM_CLASSES)
       self.softmax = nn.Softmax(dim=1)
```

## Discussion

### General discussion

TEXT
![alt text](https://github.com/sanderboers48/ReproducibilityPaperDeepLearning48/blob/main/figures/image1.png?raw=true)
### Results discussion

TEXT

## References

TEXT

## links from nice looking blogs:
* https://crisalixsa.github.io/h3d-net/
* https://github.com/RILEY-BLUE/Deep-Learning-Reproductiton/blob/master/Blog.ipynb
* https://medium.com/mlearning-ai/reproducibility-for-deep-fruit-detection-in-orchards-e518367ccf35
* https://bilal-attar.medium.com/deep-fruit-detection-in-orchards-a-reproduction-b3ca83ee0846
* https://dlorchardrepro.wordpress.com/
* https://www.notion.so/A-NEURAL-NETWORK-BASED-CORNER-DETECTION-METHOD-7fd032fbeb5043f18d9b858ca8eacd5a

## stuff from someone else:
Corner detection is a very useful and common computer vision procedure.
In "A Neural Network Based Corner Detection Method" a technique is proposed that uses neural networks to find these corners.
In this article, we will reproduce the technique described in this paper and test it on real-world data.
Specifically, we will use the corner detection method to do motion detection on a video.
For this reproduction Python 3.9 is used with PyTorch.

This corner detection technique works by looking at patches of 8x8 pixels and classifying them as containing a corner in the middle 4x4 pixels or not.

![A patch of 8 by 8 pixels](window_8x8.jpg)

*A patch of 8 by 8 pixels, If there is a corner in the red area, the whole patch is classified to be a corner patch*

The paper describes a very simple architecture with a single hidden layer of size 16.

![arechitecture](argitecture.jpg)

In the paper, it is not discussed which activation function they use.
In this reproduction, we use the relu activation function for the hidden layer.
To train the neural network, they use two sets of 8x8 images.
One set for which there is a corner in the center 4x4 pixels, and one for which there is not.
The corners are always a multiple of 45 degrees.

The paper does not give a clear description of how the training data was produced.
In this article we will use the following technique:

Generate a set of images and add random lines with corners to those images.
While placing the random lines, a list of all the corner locations is made.
Now, take random 8x8 patches from those images, and divide them into two groups.
One with corners, and one without.
Finally, throw away random samples from the largest group, to balance the data.

These images were used to train the neural network.
To generate the test data 10.000 images were generated, resulting in about 150.000 8x8 patches.
The test set contained about 1500 8x8 patches.

The initial results did not look good at all.
Every time the network was done training, it either classified all samples as having a corner or classified all samples as not having a corner.
Dropping the learning rate from 0.001 to 0.00005 solved that problem!
The accuracy also looked very good, but there was still a little problem.
The neural network learned to cheat!

To understand what was happening we should take a look at an example.
We will show a randomly generated line with corners, and next to it an image of where the neural network thinks the corners are.

![Everything is a corner](everything_is_corner.jpg)

As you can see, the neural network thinks that all parts of the line are corners!
Looking at the training samples, this makes a lot of sense.
Nearly all the images without corners are exactly the same.
Just completely black 8x8 patches.
In order to solve this problem the number of empty 8x8 patches in the training data is limited.
After running lots of experiments it turned out that having about one empty patch per 15 non-empty patches worked well.

The learning curve looks very smooth, and after 4 epochs the curve flattens out.
Every vertical line in the learning curve plot represents the end of an epoch.

![Learning curve](learning_curve.png)

With these changes the network performs very well, getting an accuracy of 93% on the test set.
Manually looking at the output of the network also shows that it actually learned to detect corners now.
In the following image all locations where the network thinks there is a corner are made red.

![Corner detection](good_corner_detection.jpg)

All the data that has been used for training and testing in the original paper, as well as in this reproduction, up to this point, is synthetically generated.
We thought it would be a good way to validate how well it works in real-world scenarios.
In the original paper, motion detection is mentioned as a use for corner detection.
We created a motion detection algorithm with this deep learning corner detection technique at the core to test the real-world performance of the algorithm.

First, every frame of the video is processed with an edge detection filter.
Then the neural network is used to find all the locations that contain a corner.
Because the neural network outputs for every pixel if it contains a corner or not, we need a method to convert this 2d array to a list of corner locations.
We first blur this 2d array and apply a threshold function. 
Then on every pixel in this array that is true, a flood-fill is used to find all its neighbours that make up the same corner.
The average location of all those neighbours is taken and added to the list of corners.

In the following image, a random frame of a video is shown.
Next to it, on the top right, there is the same frame with the edge detection applied to it.
On the bottom right, the locations of the detected corners are shown.
On this specific frame, the network did a perfect job on the 'H', but it missed the bottom left corner of the 'i'.

![](video_detection.jpg)

Then for every two consecutive frames, each corner from the first frame is matched to the closest corner in the second frame.
When the distance is within 15 pixels, an arrow is drawn in the direction of the movement, with a length equal to 5 times the moved distance.
For a single frame this looks like this:

![](arrow_frame.jpg)

All the arrows point to the left.
This is correct!
In this video, the text was indeed moving to the left (actually the camera was moving to the right).
This shows that this deep learning corner detection method can be used on real-world problems.
Not every frame is as perfect as this one, but on average the motion detection works very well.

The complete video can be seen [here](https://youtu.be/3Wegz9sT6Ig) (https://youtu.be/3Wegz9sT6Ig).
It is slowed down from 60fps to 10 fps to make it easier to see what happens.

### Conclusion
Just like the original paper, we implemented a corner detection method using deep learning.
The original paper got an accuracy of 97.55% while we got 93%.
This difference in performance can be explained by differences in the difficulty of the test set.
Our implementation also works well on real-world images.
The implementation that we made is however not very optimized for motion detection, because it took about 5 hours to analyse a 4-second video.
So, while analysing video with this method, make sure to have plenty of tea (or other beverage of your choosing) to ease the wait :)
