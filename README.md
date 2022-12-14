# TrackManiaAutonomousDriving
TrackMania Convolutional Neural Net 

# Table of Contents
1. [Introduction](#introduction) 
3. [Data Collection](#data-collection) 
4. [Model Usage](#model-usage)
5. [Blockers](#blockers)
6. [Fine Tuning and Training](#fine-tuning-and-training)
7. [Results](#results)
8. [What I Would Like to Continue](#what-i-would-like-to-continue)
9. [Sources and Citations](#sources-and-citations) 

# Introduction 

For the Independent Study I’ve begun constructing a CNN that can take in images from a car
racing game (Track Mania) and outputs speed and steering information to control the vehicle. I
plan to train the car on a custom course to be able to replicate driving behavior.


# Data Collection 

Using python and OpenPlanet, a scripting platform for the video game, I was able to access
information from the game needed as data. With this data I was able to access the speed of the car, and its progress to help automate the beginning and end of the recording process. 

I used a simple version of a map I created (no elevation changes) in order to get good data. Eventually I was able to expose the model to multiple different lighting conditions in order to explore how well the model can generalize regardless of lighting. 

One implementation which took awhile to understand was getting data from the OpenPlanet script and forwarding it to my recorder.py script. To do this I used network sockets to communicate the telemetry data from the car to the recorder. Using d3dshot helped capture screenshots at a fixed angle to not minimize any impact the angle might have on the training. This was cool because initially I had done that based on assumption after thinking back on in class lectures, however it ended up being a valuable decision. 

To get the steering angle input done, I implemented controller monitoring. I also implemented a datapoint cache that would cache around 10 seconds of results before saving them to the disk. This was a neat concept to learn about because it allowed me to implement keybinds to revert data when I would crash, take bad corners, or just make a mistake translating to collecting bad data.

In total, about ~50,000 data points (about 1.25 hours) of driving were recorded to make the dataset. I had to preprocess the data by flipping the image and steering angles in order to ensure that there would not be any bias in one direction; this actually boosted my data points by almost 100% (99,000+ datapoints). With this data in hand I split it up into training and test data and resize each image to be 64x64. By doing this to reduce the size of the model, and converting each image to grayscale to reduce the size since color of the image is irrelevant in thai instance, I was able to improve the runtime performance. 

Here is an example of what the data looks like post processing:

![speed ](https://user-images.githubusercontent.com/72223941/207712744-0d24dda3-3e00-47d8-a348-446f7457fc0e.png)
 
 [[Back To Top](#table-of-contents)]
 
# Model Usage 

When researching models to find a base type I decided that Nvidia’s End to End Learning for
Self-Driving Cars was a good pick since it is a strongly supported and realistic architecture. It is
a regression model with 5 convolutional layers and 5 connected dense layers, along with
steering. I’ve just begun implementation since it took a long time to understand what it does
alongside regular Computer Vision classwork material. Nvidia Architecture.

I used a Relu activation function for each densely connected layer and expanded the output to be two values instead of one for both steering and speed. After some reviewing on its condition, I decided to add batch normalization layers between each convolutional layer to improve performance on my dataset. With this I decided to include a dropout regularization between each densely connected layer in order to improve the models generalization. Concluding with a final atan operation I found slightly improved the performance when compared to not including an operation or softsign. Finally, I scaled it to work with my 64x64 image input. Here are examples of the neural net Nvidia described and the neural net that I used (visualization library doesn't support the training configuration of the network):

![nvidia model](https://user-images.githubusercontent.com/72223941/207716475-2092c282-09e8-4beb-82d8-2f772b821fa5.png)

![model_diagram](https://user-images.githubusercontent.com/72223941/207716028-c118a944-90aa-498b-b239-fedfee6dac49.png)

[[Back To Top](#table-of-contents)]

# Blockers 

One challenge was the restriction on the speed data (0 -~300). This mostly impacted the performance on the prediction processing. By normalizing the data to be between (0-1), as I mentioned in my Status Report 1 Conclusion, I was able to fix this. 

Another big issue was the amount of holes in my dataset. An example being when recording ideal driving, the network often did not process how to recover from unideal situations like facing a wall or right turns against a wall. This resulted in bad performance which required a recording of all the data and manual monitoring and testing of the car when put into unideal situations. 

Another issue which was hard to grasp until towards the end was trying to replicate the behavior of driving in a  video game while working around the noise and unpredictable data it returns. This led to many data points returning unideal returns in scenarios where it should have obviously made a specific move; and this was noticeable in the model accuracy being fairly low. 

Besides that the other challenges revolved mostly around connecting different platforms and softwares together for different purposes. 

[[Back To Top](#table-of-contents)]

# Fine-Tuning and Training

For this project, I wanted to explore automated hyperparameter tuning, since on PyTorch, RayTune would let me automate the parameter sweep process while simultaneously parallelizing the training using fractional GPU’s and early termination of bad trials. 

![batch](https://user-images.githubusercontent.com/72223941/207721387-83e0e156-afd9-4b76-ac95-172ebb94a872.png)

[[Back To Top](#table-of-contents)]

# Results 

Since the model is a regression model, I expected within 0.10 of the expected value to be the margin of error. Result of testing the model on the test dataset: 

![ddataset](https://user-images.githubusercontent.com/72223941/207722255-f1a113d5-5239-4618-82a0-32842cafec94.png)

The accuracy problem doesn't return the best results due to the high volume of noise and bad data, however it still preforms well when used to control the vehicle. Even after adding more data however to learn from bad situations, it would still often get stuck on basic manuevers.

Here's an example of it running well on the simple map: 

[YouTube Video](https://youtu.be/-1cM-NEqQuY).



# What I Would Like to Continue

If given the opportunity, I would like to collect more high-quality training data. I assume a reason for unpredictable and noisy performance was due to the lack of large amounts of data to learn from. One fix I was looking into was implementing a dual recording/inferencing script that would allow me to switch between the model and my own inputs. This would help record specific scenarios when the model is failing and let me get more relevant learning data. On top of this, I could experiment with adding more adjustments to the Nvidia model. 

[[Back To Top](#table-of-contents)]

# Sources and Citations 

| File | Work Description |
| --  | --- |
| `dataset.py` | Implemented by me. heavily based on examples from [PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). |
| `inference.py` | Implemented by me, except one line of steering smoothing from [this project](https://github.com/SullyChen/Autopilot-TensorFlow/). |
| `make_link.bat` | Implemented by me to link OpenPlanet plugin file to the plugin directory. |
| `train.py` | Implemented by me, inspiried from multiple online sources |
| `tune.py` | Implemented by me, modified from [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html). |
| `recorder.py` | Implemented by me, besides XboxController class modified slightly from [here](https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python). |
| `model.py` | Implemented by me, inspired by [Nvidia](https://arxiv.org/pdf/1604.07316.pdf). |
| `Plugin_TrackManiaCustomAPI.as` | Implemented by me using [this source](https://trackmania-api-node.netlify.app/); had to learn the basics of ActionScript to get this done |
| `process_data.py` | Implemented by me. |


[[Back To Top](#table-of-contents)]

