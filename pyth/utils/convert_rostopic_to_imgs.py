#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
import rosbag
import cv2
import time
from sensor_msgs.msg import Image, CameraInfo
import message_filters

bridge = CvBridge()

def handle_image(image, depth):
    timed = time.time()
    path_im = f'/home/mikkel/ros_pics/final_test_with_robot/{timed}_image.jpg'
    path_de = f'/home/mikkel/ros_pics/final_test_with_robot/{timed}_depth.PNG'
    cv2.imwrite(path_im, image)
    cv2.imwrite(path_de, depth)
    cv2.imshow('image', image)
    cv2.waitKey(500)
    # input('enter to save next picture')
    
def image_callback(image, depth):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        image = bridge.imgmsg_to_cv2(image, 'bgr8')
        depth = bridge.imgmsg_to_cv2(depth, "16UC1")
    except CvBridgeError as e:
        print(e)
    else:
        handle_image(image, depth)

def main():
    rospy.init_node('fejemisVisionMain')
    image_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    
    # Set up your subscriber and define its callback
    image_sub = message_filters.Subscriber(image_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)
    ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 1, 1)
    ts.registerCallback(image_callback)
    
    rospy.spin()
    
    
main()