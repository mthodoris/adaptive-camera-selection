# adaptive-camera-selection

Best view selected from multicamera system 

ROS version: noetic Install ROS noetic instructions

http://wiki.ros.org/noetic/Installation/Ubuntu

follow 1.1, 1.2, 1.3, 1.4

1.4 --> sudo apt install ros-noetic-desktop

Add Gazebo for running the simulation environment

Instructions: https://classic.gazebosim.org/tutorials?tut=install_ubuntu

To run the simulation and the tool follow the next steps:

enter current directory cd active-camera-selection

run ./venv_setup.sh

activate venv

cd venv

source ./bin/activate

Build ROS packages
In $active-camera-selection directory

cd active-camera-selection

cd /catkin_ws

catkin_make

source devel/setup.bash

Launch the project
roslaunch robosapiens_tool active_camera_selection.launch

Demo Image
Here is a video showcasing our project.

Demo Video

OR

https://drive.google.com/file/d/1YyOXba8wrdovWMPNo5imzmLLTzOlbqh3/view?usp=drive_link

A graphical representation of the ros-nodes is depicted in the following graph

ROS nodes graph
