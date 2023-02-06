#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped, Point
import tf
import rospkg
import time

import pyrealsense2 as rs2

from dataclasses import dataclass

@dataclass
class Person:
    head_stats: list
    torso_stats: list
    legs_stats: list
    height_stats: float
    x: float
    y: float
    z: float
    tm: float
    image_person: list

class HumanDetectorDL:
    def __init__(self):
        rospack = rospkg.RosPack()
        base_path = rospack.get_path('fejemis_vision')
        
        self.net = cv2.dnn.readNet(f"{base_path}/deep_learning/yolov4-tiny.weights", f"{base_path}/deep_learning/yolov4-tiny.cfg")
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.5
        self.updated_index = 0
        with open(f"{base_path}/deep_learning/classes.txt", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        
        self.people_per_frame_limit = 3
        self.intrinsics = None
        self.database = []
        self.tf_listener = tf.TransformListener()
        
        self.avg_limit = 10
        self.person_pub = rospy.Publisher('New_person', PointStamped, queue_size=10)
        self.minimum_time_detection_limit = 1
        self.maximum_time_detection_limit = 10
        self.maximum_distance_limit = 1.5
    
    def set_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics
        
    def convert_from_uvd(self, u, v, d):
        if d == 0:
            return 0,0,0
            
        u = u * (self.intrinsics.width / self.image_shape[1])
        v = v * (self.intrinsics.height / self.image_shape[0]) 
        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [u,v], d)
        
        X,Y,Z = result[0]*0.001, result[1]*0.001, result[2]*0.001
        
        return X,Y,Z
    
    def publish_to_ros(self, person):
        msg = PointStamped()
        msg.point = Point(person.x, person.y, person.z)
        
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        self.person_pub.publish(msg)
    
    def remove_long_time_members_from_database(self):
        current_time = time.time()
        for i, person in enumerate(self.database):
            if current_time - person.tm > self.maximum_time_detection_limit:
                self.database.pop(i)
    
    def compare_stats_to_database(self, person):
        def euclidean_dist(p1, p2):
            return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
        
        self.exists_index = -1
        dists = []
        if len(self.database) != 0:
            for i, per in enumerate(self.database):
                avg_dist = np.average([np.average(np.abs(np.array(per.head_stats) - np.array(person.head_stats))),
                                       np.average(np.abs(np.array(per.torso_stats) - np.array(person.torso_stats))),
                                       np.average(np.abs(np.array(per.legs_stats) - np.array(person.legs_stats)))])

                dists.append(avg_dist)
                                
        if len(dists) != 0:
            idx = np.argmin(dists)
            dist = dists[idx]
            if dist < self.avg_limit:
                self.exists_index = idx
            
        if self.exists_index != -1:
            if time.time() - person.tm > self.minimum_time_detection_limit and euclidean_dist(person, self.database[self.exists_index]) < self.maximum_distance_limit:
                self.publish_to_ros(person)
            #Update everything except for the location and time since we want to check if they are moving or not.
            self.database[self.exists_index].head_stats = person.head_stats
            self.database[self.exists_index].torso_stats = person.torso_stats
            self.database[self.exists_index].legs_stats = person.legs_stats      
            self.updated_index = self.exists_index+1
        else:
            self.database.append(person)
        return
    
    def segment_and_get_stats(self, img, distance, box, mask):
        def color_stats(p):
            h = []
            s = []
            v = []
            for x in p:
                h.append(x[0])
                s.append(x[1])
                v.append(x[2])
            return [np.average(h), np.average(s), np.average(v)]
        # print(box)
        x,y,w,h = box
        head = img[0:int((1/6)*img.shape[0]), 0:img.shape[1]]
        torso = img[int((1/6)*img.shape[0]): int((3/6)*img.shape[0]), 0:img.shape[1]]
        legs = img[int((3/6)*img.shape[0]): int((6/6)*img.shape[0]), 0:img.shape[1]]

        head_mask = mask[0:int((1/6)*mask.shape[0]), 0:mask.shape[1]]
        torso_mask = mask[int((1/6)*mask.shape[0]): int((3/6)*mask.shape[0]), 0:mask.shape[1]]
        legs_mask = mask[int((3/6)*mask.shape[0]): int((6/6)*mask.shape[0]), 0:mask.shape[1]]
        
        head_stats = color_stats(head[head_mask != 0])
        torso_stats = color_stats(torso[torso_mask != 0])
        legs_stats = color_stats(legs[legs_mask != 0])
        
        height_top = self.convert_from_uvd(x+w/2, y, distance)
        height_bottom = self.convert_from_uvd(x+w/2, y+h, distance)
        height_stats = height_bottom[1] - height_top[1]
        
        #Get point from tf's
        msg = PointStamped()
        msg.header.stamp = self.image_time_stamp
        msg.header.frame_id = "camera_link"
        msg.point.x = height_bottom[0]
        msg.point.y = height_bottom[1]
        msg.point.z = height_bottom[2]
        
        p1 = self.tf_listener.transformPoint('/map', msg)

        new_person = Person(head_stats, torso_stats, legs_stats, height_stats, p1.point.x, p1.point.y, p1.point.z, time.time(), img)
        self.compare_stats_to_database(new_person)
    
    def check_frame(self, classes, scores, boxes, image, depth):
        depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        detected_bboxes = []
            
        for (classid, score, box) in zip(classes, scores, boxes):
            if self.class_names[classid] == 'person':
                self.updated_index = -2
                label = "%s : %f" % (self.class_names[classid], score)
                detected_bboxes.append([box, label])
                x,y,w,h = box
                image_person = image[y:y+h+10, x:x+w+10]
                image_person_depth = depth[y:y+h+10, x:x+w+10]
                
                image_person_depth_color = cv2.cvtColor(image_person_depth.copy(), cv2.COLOR_GRAY2BGR)
                image_person = cv2.cvtColor(image_person, cv2.COLOR_BGR2HSV)
                pixel_values = image_person_depth_color.reshape((-1,3))
                pixel_values = np.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, (centers) = cv2.kmeans(pixel_values, 2, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
                
                centers = np.uint8(centers)
                labels = labels.flatten()
                largest_label = np.bincount(labels).argmax()
                segmented_image = centers[labels.flatten()]
                segmented_image[labels != largest_label] = [0,0,0]
                segmented_image = segmented_image.reshape(image_person_depth_color.shape)
                kernel = np.ones((15,15),np.uint8)
                si_mask = cv2.threshold(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                kernel = np.ones((5,5),np.uint8)
                si_mask = cv2.morphologyEx(si_mask, cv2.MORPH_OPEN, kernel)
                img = cv2.bitwise_and(image_person, image_person, mask = si_mask)
                distance = depth[int(y+h/2)][int(x+w/2)]

                self.segment_and_get_stats(img, distance, box, si_mask)

                image = cv2.rectangle(image, box, (0,255,0), 2)
                image = cv2.putText(image, label, (box[0]+box[2], box[1]+box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                depth_map = cv2.rectangle(depth_map, box, (0,255,0), 2)
        
        return detected_bboxes
        
    def model_inference(self, image, depth):
        self.image_shape = image.shape

        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        
        detected_bboxes = []
        
        if len(boxes) != 0:
            detected_bboxes = self.check_frame(classes, scores, boxes, image, depth)
        
        return detected_bboxes
    
    def new_frame(self, image, depth, image_time_stamp):
        self.image_time_stamp = image_time_stamp
        detected_bboxes = self.model_inference(image, depth)
        self.remove_long_time_members_from_database()
        
