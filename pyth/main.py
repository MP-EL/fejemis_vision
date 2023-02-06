#!/usr/bin/env python3

# from people_detection_depth_segmentation import HumanDetectorDL
# from people_detection_simple_segmentation import HumanDetectorDL
from people_detection_pose_segmentation import HumanDetectorDL
from cable_detection import CablesDetector
from net_detection import NetDetector

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs2

import message_filters
import cv2
import glob

class MasterNode:
    def __init__(self) -> None:
        self.cable_det = True
        self.people_det = True
        self.net_det = True
        
        self.bridge = CvBridge()
        self.HMdet = HumanDetectorDL()
        self.CBdet = CablesDetector()
        self.NTdet = NetDetector()
        
        self.intrinsics = None
        #### Only used if running with images from a folder:
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = 1280
        self.intrinsics.height = 720
        self.intrinsics.ppx = 629.3971557617188
        self.intrinsics.ppy = 358.9218444824219
        self.intrinsics.fx = 639.0593872070312
        self.intrinsics.fy = 638.0255126953125
        self.intrinsics.model = rs2.distortion.brown_conrady
        self.intrinsics.coeffs = [-0.05425437167286873, 0.06392815709114075, -0.0004175958165433258, 7.225961599033326e-05, -0.020541394129395485]
        self.CBdet.set_intrinsics(self.intrinsics)
        self.HMdet.set_intrinsics(self.intrinsics)
        self.NTdet.set_intrinsics(self.intrinsics)
        
        self.skipped_cable_detections = 0
        self.skipped_cable_detections_desired = 13
        self.skipped_net_detections = 0
        self.skipped_net_detections_desired = 30
        self.skipped_people_detections = 0
        self.skipped_people_detections_desired = 15
    
    def handle_people(self, image, depth, image_time_stamp):
        self.HMdet.new_frame(image, depth, image_time_stamp)
        
    def handle_cable(self, image, depth, image_time_stamp):
        cables = self.CBdet.find_cable(image, depth, image_time_stamp) 
            
    def handle_net(self, image, depth, image_time_stamp):
        self.NTdet.detect_net(image, depth, image_time_stamp)
        
    def handle_image(self, image, depth):
        image_time_stamp = rospy.Time()
        
        if self.people_det == True:
            if self.skipped_people_detections_desired <= self.skipped_people_detections:
                self.skipped_people_detections = 0
                self.handle_people(image, depth, image_time_stamp)
            else:
                self.skipped_people_detections += 1
        
        if self.net_det == True:
            if self.skipped_net_detections_desired <= self.skipped_net_detections:
                self.skipped_net_detections = 0
                self.handle_net(image, depth, image_time_stamp)
            else:
                self.skipped_net_detections += 1
        
        if self.cable_det == True:
            if self.skipped_cable_detections_desired <= self.skipped_cable_detections:
                self.skipped_cable_detections = 0
                self.handle_cable(image, depth, image_time_stamp)
            else:
                self.skipped_cable_detections += 1
            
    def image_callback(self, image, image_info, depth, depth_info):
        # print("Received an image!")
        try:
            if self.intrinsics == None:
                self.intrinsics = rs2.intrinsics()
                self.intrinsics.width = image_info.width
                self.intrinsics.height = image_info.height
                self.intrinsics.ppx = image_info.K[2]
                self.intrinsics.ppy = image_info.K[5]
                self.intrinsics.fx = image_info.K[0]
                self.intrinsics.fy = image_info.K[4]
                if image_info.distortion_model == 'plumb_bob':
                    self.intrinsics.model = rs2.distortion.brown_conrady
                elif image_info.distortion_model == 'equidistant':
                    self.intrinsics.model = rs2.distortion.kanela_brandt4
                self.intrinsics.coeffs = [i for i in image_info.D]
                
                self.CBdet.set_intrinsics(self.intrinsics)
                self.HMdet.set_intrinsics(self.intrinsics)
                self.NTdet.set_intrinsics(self.intrinsics)
                
        except CvBridgeError as e:
            print(e)
        
        try:
            # Convert your ROS Image message to OpenCV2
            image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError as e:
            print(e)
        else:
            self.handle_image(image, depth)

    def main(self):
        # Image and depth topics including their info topics to extract camera intrinsics.
        image_topic = "/camera/color/image_raw"
        image_info_topic = "/camera/color/camera_info"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        depth_info_topic = "/camera/aligned_depth_to_color/camera_info"
        
        # TimeSynchronizer so that all topics are gathered at the same time. 
        image_sub = message_filters.Subscriber(image_topic, Image)
        image_info_sub = message_filters.Subscriber(image_info_topic, CameraInfo)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        depth_info_sub = message_filters.Subscriber(depth_info_topic, CameraInfo)
        ts = message_filters.TimeSynchronizer([image_sub, image_info_sub, depth_sub, depth_info_sub], 1, 1)
        ts.registerCallback(self.image_callback)
        
        rospy.spin() 

if __name__ == '__main__':
    #ROS setup
    rospy.init_node('fejemisVisionMain')
    
    #Run the main class
    node = MasterNode()
    node.main()