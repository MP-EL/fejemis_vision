# fejemis
Masters thesis repository

3 detectors:

cable detection:
![image](https://user-images.githubusercontent.com/62695168/222127079-7999edd4-04d0-43c2-bdbc-c21c1be7e41e.png)

people detection + tracking:

![image_person_1](https://user-images.githubusercontent.com/62695168/222127316-6a0d7e1d-10e3-48f4-969f-1573ae5110e5.jpg)

netting detection:

![New_net_test_image2](https://user-images.githubusercontent.com/62695168/222127906-71d47e27-697c-4fd2-83b6-13d644883a3e.png)

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

To run any of the test scripts (these do not need the topics above to be published since they use images saved in the folder with the code):
```
roslaunch fejemis\_vision cable\_test.launch
roslaunch fejemis\_vision people\_test.launch
roslaunch fejemis\_vision net\_test.launch
```
