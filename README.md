
# yolov5_StrongSORT_ROS
Integration of  "Yolov5 StrongSort" with ROS for real time object tracking


# Installation

Clone the repository recursively:

```
cd catkin_workspace/src
git clone --recurse-submodules https://github.com/utopiacompression/yolov5_strongsort_ros.git
```
or if you cloned the repo without --recurse-submodules
```
git clone https://github.com/PardisTaghavi/yolov7_StrongSORT_ROS.git
cd yolov7_strongsort_ros
git submodule update --init
```



install requirements and build the package

```
cd Yolov7_StrongSORT_OSNet
pip install -r requirements.txt

cd ../..
catkin build yolov7_strongsort_ros or catkin_make
```
don't forget to source the package
```
source devel/setup.bash
```


run through python code:
```
chmod +x trackRos.py (do it one time to change accessibility of the file)
python trackRos.py 
```

or run through the launch file:
```
roslaunch yolov7_strongsort_ros trackingLaunch.launch
```

Name of the topic should be modified in the trackRos.py


# Speed
you can check frequency :
```
rostopic hz /trackResult
```
Currently it is around 7-8 Hz.
