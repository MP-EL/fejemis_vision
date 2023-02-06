# fejemis
Masters thesis repository

The system is tested on Ubuntu 20.04.5 LTS with ROS Noetic.

The following packages are needed to run the main script:

- OpenCV should be installed using "pip install opencv-python"
- ROS needs to be installed by following the guide here: \url{http://wiki.ros.org/noetic/Installation/Ubuntu} where the full instalation should be chosen. 
- The various python packages used can be installed using the requiremtents.txt file by running "pip install requiremtents.txt"
- jsk-visualization (sudo apt-get install ros-noetic-jsk-visualization) used for the markerArray message type and visualization for ROS

To run the scripts using roslaunch the following scripts will need executable permissions:

```
chmod +x path\_to\_ros\_package/pyth/main.py 
chmod +x path\_to\_ros\_package/pyth/net\_test.py
chmod +x path\_to\_ros\_package/pyth/people\_test\_basic.py
chmod +x path\_to\_ros\_package/pyth/cable\_test.py 
```

Build the package like you would build any other ros package. 

Running the main script needs the following topics published:
```
"/camera/color/image\_raw"
"/camera/color/camera\_info"
"/camera/aligned\_depth\_to\_color/image\_raw"
"/camera/aligned\_depth\_to\_color/camera\_info"
```

To run run the main script:
```
run "roslaunch fejemis\_vision main.launch"
```

To run any of the test scripts:
```
roslaunch fejemis\_vision cable\_test.launch
roslaunch fejemis\_vision people\_test.launch
roslaunch fejemis\_vision net\_test.launch
```
