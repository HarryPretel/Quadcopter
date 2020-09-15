# Getting Started

This guide will provide all the information you need to get started with our autonomous drone application using the DJI Tello drone. First you will need to make sure you have installed a few things so the application can run.

# Installation
You will need to complete the following steps before running any code:
### Github
To **clone** this repository from github, run:
```
$ git clone https://github.com/HarryPretel/Quadcopter.git
```
To **clone** the repo that implements manual control, run:
```
$ git clone https://github.com/damiafuentes/DJITelloPy
```
Python 3.7.1 is required. Execute the following command to verify you are on 3.7.1. If you are not, this repo will not work.
```
$ python --version
```
### Packages
You will need to install certain packages for python so the application can be carried out properly. These packages may take a long time to install, be patient.
**Pip** is required to install these packages. If you do not have pip, download it.

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
$ pip install pygame
```

# Connecting to the Drone
You must connect to the drone over wifi. This means that you cannot access the internet through wifi while you are connected. If this is an issue, use a wired connection to access the internet.
1. Turn on drone
2. Connect to drone's wifi

### Aruco Landing:
A 10cm square, 6x6 format Aruco marker must be downloaded and printed from https://chev.me/arucogen/

After printing, tape your aruco marker somewhere visible to the drone immediately after takeoff. If the marker isn't visible immediately after takeoff, you will need to control the drone manually (using the controls listed under ***manual control***)until it is visible within frame.

You may move the marker, the drone will try to hover a certain distance away from the marker, once it is within that range and stable, it will land. If you don't want the drone to land, make sure that you don't it to get too close to the marker.

To begin ***semi-autonomous flight***, do the following after connecting to the drone:
```$ python PID.py ```
Press T to takeoff once the application has established a connection to the drone's camera. The drone will take off, while looking for an aruco marker. If an aruco marker is found, it will use the PID controller implemented to land a certain distance from the marker. These distance parameters, as well as parameters affecting the behavior of the PID controller can be set on lines 111-121 in PID.py.

### Manual Control:
T: Takeoff
L: Land
W: Up
S: Down
A: CCW Rotation
D: CW Rotation
Arrow Keys: Planar movement

To begin ***manual flight***, do the following after connecting to the drone:
```
$ python manual_control.py
```

### Easter Eggs:
There are many files in this repo that were used in the development of 

# Troubleshooting:
1. Verify that all packages have been installed correctly.
2. Verify that you are connected to the correct drone.
3. Verify that the drone has a full battery.
4. Inspect the drone for any damage to propellers, body, etc.

# References

 - https://github.com/damiafuentes/DJITelloPy
 - https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
