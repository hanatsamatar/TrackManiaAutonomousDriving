# TrackManiaAutonomousDriving
TrackMania Convolutional Neural Net 

# Table of Contents
1. [Introduction](#introduction) 
3. Data Collection 
4. Model Usage
5. Blockers / Difficulties
6. Fine-Tuning and Training
7. Results
8. What I Would Like to Continue On
9. Sources 

# Introduction 

For the Independent Study I’ve begun constructing a CNN that can take in images from a car
racing game (Track Mania) and outputs speed and steering information to control the vehicle. I
plan to train the car on a custom course to be able to replicate driving behavior.

[[Back To Top](#table-of-contents)]

# Data Collection 

Using python and OpenPlanet, a scripting platform for the video game, I was able to access
information from the game needed as data. With this data I was able to access the speed of the car, and its progress to help automate the beginning and end of the recording process. 

I used a simple version of a map I created (no elevation changes) in order to get good data. Eventually I was able to expose the model to multiple different lighting conditions in order to explore how well the model can generalize regardless of lighting. 

One implementation which took awhile to understand was getting data from the OpenPlanet script and forwarding it to my recorder.py script. To do this I used network sockets to communicate the telemetry data from the car to the recorder. Using d3dshot helped capture screenshots at a fixed angle to not minimize any impact the angle might have on the training. This was cool because initially I had done that based on assumption after thinking back on in class lectures, however it ended up being a valuable decision. 

To get the steering angle input done, I implemented controller monitoring. I also implemented a datapoint cache that would cache around 10 seconds of results before saving them to the disk. This was a neat concept to learn about because it allowed me to implement keybinds to revert data when I would crash, take bad corners, or just make a mistake translating to collecting bad data.

In total, about ~50,000 data points (about 1.25 hours) of driving were recorded to make the dataset. I had to preprocess the data by flipping the image and steering angles in order to ensure that there would not be any bias in one direction; this actually boosted my data points by almost 100% (99,000+ datapoints). With this data in hand I split it up into training and test data and resize each image to be 64x64. By doing this to reduce the size of the model, and converting each image to grayscale to reduce the size since color of the image is irrelevant in thai instance, I was able to improve the runtime performance. 

Here is an example of what the data looks like post processing:

![speed ](https://user-images.githubusercontent.com/72223941/207712744-0d24dda3-3e00-47d8-a348-446f7457fc0e.png)
 
# Model Usage 

When researching models to find a base type I decided that Nvidia’s End to End Learning for
Self-Driving Cars was a good pick since it is a strongly supported and realistic architecture. It is
a regression model with 5 convolutional layers and 5 connected dense layers, along with
steering. I’ve just begun implementation since it took a long time to understand what it does
alongside regular Computer Vision classwork material. Nvidia Architecture.

I used a Relu activation function for each densely connected layer and expanded the output to be two values instead of one for both steering and speed. After some reviewing on its condition, I decided to add batch normalization layers between each convolutional layer to improve performance on my dataset. With this I decided to include a dropout regularization between each densely connected layer in order to improve the models generalization. Concluding with a final atan operation I found slightly improved the performance when compared to not including an operation or softsign. Finally, I scaled it to work with my 64x64 image input. Here are examples of the neural net Nvidia described and the neural net that I used (visualization library doesn't support the training configuration of the network):

![nvidia model](https://user-images.githubusercontent.com/72223941/207716475-2092c282-09e8-4beb-82d8-2f772b821fa5.png)

![model_diagram](https://user-images.githubusercontent.com/72223941/207716028-c118a944-90aa-498b-b239-fedfee6dac49.png)


# Blockers / Difficulties

One challenge was the restriction on the speed data (0 -~300). This mostly impacted the performance on the prediction processing. By normalizing the data to be between (0-1), as I mentioned in my Status Report 1 Conclusion, I was able to fix this. 

Another big issue was the amount of holes in my dataset. An example being when recording ideal driving, the network often did not process how to recover from unideal situations like facing a wall or right turns against a wall. This resulted in bad performance which required a recording of all the data and manual monitoring and testing of the car when put into unideal situations. 

Another issue which was hard to grasp until towards the end was trying to replicate the behavior of driving in a  video game while working around the noise and unpredictable data it returns. This led to many data points returning unideal returns in scenarios where it should have obviously made a specific move; and this was noticeable in the model accuracy being fairly low. 

Besides that the other challenges revolved mostly around connecting different platforms and softwares together for different purposes. 

# Fine-Tuning and Training

For this project, I wanted to explore automated hyperparameter tuning, since on PyTorch, RayTune would let me automate the parameter sweep process while simultaneously parallelizing the training using fractional GPU’s and early termination of bad trials. 

![batch](https://user-images.githubusercontent.com/72223941/207721387-83e0e156-afd9-4b76-ac95-172ebb94a872.png)






