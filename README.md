# Getting Started

This guide will provide all the information you need to get started with our autonomous drone application using the DJI Tello drone. First you will need to make sure you have installed a few things so the application can run.

# Installation

## Git
Git must be installed to download and install the code repository to run on a computer. Instructions to do this are posted here: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
```
$ git clone https://github.com/HarryPretel/Quadcopter.git
```
## Python

The Python language and certain packages need to be installed before the application can run. Installation differs slightly depending on if you are using Windows or OS X. For complete instructions follow the steps provided here: [https://realpython.com/installing-python/](https://realpython.com/installing-python/)

Application tested with Python 3.7.1 but may work with other versions.

### Packages
You will need to install certain packages for python so the application can be carried out properly. 

First install pip according to these instructions: [https://pip.pypa.io/en/stable/installing/](https://pip.pypa.io/en/stable/installing/)

Now install the following packages by copy and pasting the following code in the command line.

To install **Numpy**:
```
$ pip install numpy
```
To install **OpenCV**:
```
$ pip install opencv-python
```
To install **PyGame**:
```
$ pip install opencv-python
```

# Connecting to the Drone
The computer connects to the drone using a WiFi UDP port. It is important to note that while this is a fast way to send information, this protocol does not guarantee commands will be delivered.

The connection scheme is already configured for you (found in djitellopy/tello.py). Full connection details as well as a list of valid commands for the DJI Tello can be found in DJI's documentation.

Controlling the drone can be done manually from the keyboard or scripted. Manual controls are listed below:

T: Takeoff
L: Land
W: Up
S: Down
A: CCW Rotation
D: CW Rotation
Arrow Keys: Planar movement

# References

 - https://github.com/damiafuentes/DJITelloPy
 - https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
