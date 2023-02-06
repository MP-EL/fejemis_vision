#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from geometry_msgs.msg import PointStamped, Point
from scipy.spatial.distance import cdist
import tf
import time
import rospkg

import mediapipe as mp
import pyrealsense2 as rs2

from dataclasses import dataclass

@dataclass
class Person:
    right_thigh: list
    left_thigh: list
    right_calve: list
    left_calve: list
    torso: list
    right_back: list
    left_back: list
    shoulder: list
    right_arm_upper: list
    right_arm_lower: list
    left_arm_upper: list
    left_arm_lower: list
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
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.group_euclidean_distance = 1
        self.updated_index = 0
        
        self.pose = self.mp_pose.Pose(
            static_image_mode = True,
            model_complexity = 0,
            enable_segmentation = True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        with open(f"{base_path}/deep_learning/classes.txt", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        
        self.people_per_frame_limit = 3
        self.intrinsics = None
        self.database = []
        self.tf_listener = tf.TransformListener()
        self.person_pub = rospy.Publisher('New_person', PointStamped, queue_size=10)
        
        self.minimum_time_detection_limit = 1
        self.maximum_time_detection_limit = 10
        self.maximum_distance_limit = 1.5
        self.avg_limit = 10
    
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
        idxs = []
        if len(self.database) != 0:
            for i, per in enumerate(self.database):
                avg_dist = np.average([np.average(np.abs(np.array(per.right_thigh) - np.array(person.right_thigh))),
                                       np.average(np.abs(np.array(per.left_thigh) - np.array(person.left_thigh))),
                                       np.average(np.abs(np.array(per.right_calve) - np.array(person.right_calve))),
                                       np.average(np.abs(np.array(per.left_calve) - np.array(person.left_calve))),
                                       np.average(np.abs(np.array(per.torso) - np.array(person.torso))),
                                       np.average(np.abs(np.array(per.right_back) - np.array(person.right_back))),
                                       np.average(np.abs(np.array(per.left_back) - np.array(person.left_back))),
                                       np.average(np.abs(np.array(per.shoulder) - np.array(person.shoulder))),
                                       np.average(np.abs(np.array(per.right_arm_upper) - np.array(person.right_arm_upper))),
                                       np.average(np.abs(np.array(per.right_arm_lower) - np.array(person.right_arm_lower))),
                                       np.average(np.abs(np.array(per.left_arm_upper) - np.array(person.left_arm_upper))),
                                       np.average(np.abs(np.array(per.left_arm_lower) - np.array(person.left_arm_lower)))])
                # print(f'{avg_dist=}')

                dists.append(avg_dist)
                idxs.append(i)
            
        if len(dists) != 0:
            idx = np.argmin(dists)
            dist = dists[idx]
            if dist < self.avg_limit:
                self.exists_index = idx

        if self.exists_index != -1:

            if euclidean_dist(person, self.database[self.exists_index]) > self.maximum_distance_limit:
                self.database[self.exists_index].x = person.x
                self.database[self.exists_index].y = person.y
                self.database[self.exists_index].z = person.z
                self.database[self.exists_index].tm = person.tm
            if (time.time() - person.tm)*1000 > self.minimum_time_detection_limit and euclidean_dist(person, self.database[self.exists_index]) < self.maximum_distance_limit:
                self.publish_to_ros(person)
            #Update everything except for the location and time since we want to check if they are moving or not.
            self.database[self.exists_index].right_thigh = person.right_thigh
            self.database[self.exists_index].left_thigh = person.left_thigh
            self.database[self.exists_index].right_calve = person.right_calve
            self.database[self.exists_index].left_calve = person.left_calve
            self.database[self.exists_index].torso = person.torso
            self.database[self.exists_index].right_back = person.right_back
            self.database[self.exists_index].left_back = person.left_back
            self.database[self.exists_index].shoulder = person.shoulder
            self.database[self.exists_index].right_arm_upper = person.right_arm_upper
            self.database[self.exists_index].right_arm_lower = person.left_arm_upper
            self.database[self.exists_index].left_arm_upper = person.left_arm_upper
            self.database[self.exists_index].left_arm_lower = person.left_arm_lower 
            self.updated_index = self.exists_index+1
        else:
            self.database.append(person)
        return
        
    def segment_and_get_stats(self, image, person_image, distance, box, relevant_landmarks):
        def line_thing(x0, y0, x1, y1, image):
            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x1)
            y1 = int(y1)
            "Bresenham's line algorithm"
            points_in_line = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = -1 if x0 > x1 else 1
            sy = -1 if y0 > y1 else 1
            if dx > dy:
                err = dx / 2.0
                while x != x1:
                    points_in_line.append(image[y][x])
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != y1:
                    points_in_line.append(image[y][x])
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            points_in_line.append(image[y][x])
            return points_in_line
        def color_stats(p):
            h = []
            s = []
            v = []
            for x in p:
                h.append(x[0])
                s.append(x[1])
                v.append(x[2])
            return [np.average(h), np.average(s), np.average(v)]
        x,y,w,h = box
        image_hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        height_bottom = self.convert_from_uvd(x+w/2, y+h, distance)
        
        #Translate local camera point to world points for tracking between frames.
        msg = PointStamped()
        msg.header.stamp = self.image_time_stamp
        msg.header.frame_id = "camera_link"
        msg.point.x = height_bottom[0]
        msg.point.y = height_bottom[1]
        msg.point.z = height_bottom[2]
        
        p1 = self.tf_listener.transformPoint('/map', msg)
        
        # print(relevant_landmarks[24].x, relevant_landmarks[24].y, relevant_landmarks[26].x, relevant_landmarks[26].y)
        right_thigh = color_stats(line_thing(relevant_landmarks[24].x, relevant_landmarks[24].y, relevant_landmarks[26].x, relevant_landmarks[26].y, image_hsv))
        left_thigh = color_stats(line_thing(relevant_landmarks[23].x, relevant_landmarks[23].y, relevant_landmarks[25].x, relevant_landmarks[25].y, image_hsv))
        right_calve = color_stats(line_thing(relevant_landmarks[26].x, relevant_landmarks[26].y, relevant_landmarks[28].x, relevant_landmarks[28].y, image_hsv))
        left_calve = color_stats(line_thing(relevant_landmarks[25].x, relevant_landmarks[25].y, relevant_landmarks[27].x, relevant_landmarks[27].y, image_hsv))
        torso = color_stats(line_thing(relevant_landmarks[24].x, relevant_landmarks[24].y, relevant_landmarks[23].x, relevant_landmarks[23].y, image_hsv))
        right_back = color_stats(line_thing(relevant_landmarks[24].x, relevant_landmarks[24].y, relevant_landmarks[12].x, relevant_landmarks[12].y, image_hsv))
        left_back = color_stats(line_thing(relevant_landmarks[23].x, relevant_landmarks[23].y, relevant_landmarks[11].x, relevant_landmarks[11].y, image_hsv))
        shoulder = color_stats(line_thing(relevant_landmarks[12].x, relevant_landmarks[12].y, relevant_landmarks[11].x, relevant_landmarks[11].y, image_hsv))
        right_arm_upper = color_stats(line_thing(relevant_landmarks[12].x, relevant_landmarks[12].y, relevant_landmarks[14].x, relevant_landmarks[14].y, image_hsv))
        right_arm_lower = color_stats(line_thing(relevant_landmarks[14].x, relevant_landmarks[14].y, relevant_landmarks[16].x, relevant_landmarks[16].y, image_hsv))
        left_arm_upper = color_stats(line_thing(relevant_landmarks[11].x, relevant_landmarks[11].y, relevant_landmarks[13].x, relevant_landmarks[13].y, image_hsv))
        left_arm_lower = color_stats(line_thing(relevant_landmarks[13].x, relevant_landmarks[13].y, relevant_landmarks[15].x, relevant_landmarks[15].y, image_hsv)) 
        
        new_person = Person(right_thigh, left_thigh, right_calve, left_calve, 
                            torso, right_back, left_back, shoulder, right_arm_upper, 
                            right_arm_lower, left_arm_upper, left_arm_lower, 
                            p1.point.x, p1.point.y, p1.point.z, time.time(), person_image)
        
        self.compare_stats_to_database(new_person)
    
    def get_relevant_markers(self, landmarks, relevant_landmarks, box):
        x,y,w,h = box
        for i in range(0, 32):
            if landmarks.landmark[i] != None:
                landmarks.landmark[i].x = landmarks.landmark[i].x * w + x
                landmarks.landmark[i].y = landmarks.landmark[i].y * h + y
                relevant_landmarks.append(landmarks.landmark[i])
            else:
                relevant_landmarks.append(None)
        return relevant_landmarks
            
    def check_frame(self, classes, scores, boxes, image, depth):
        detected_bboxes = []
        n_people_frame = len(classes[classes == 0])
        for (classid, score, box) in zip(classes, scores, boxes):
            if self.class_names[classid] == 'person':
                label = f"{self.class_names[classid]} : {score:.2f}"
                detected_bboxes.append([box, label])
                box[2] += 10
                box[3] += 10
                x,y,w,h = box
                image_person = image[y:y+h, x:x+w]

                results = self.pose.process(cv2.cvtColor(image_person, cv2.COLOR_BGR2RGB))
                annotated_image = image_person.copy()
                relevant_landmarks = []
                if results.pose_landmarks != None:
                    self.updated_index = -2
                    
                    # mask of person
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(image_person.shape, dtype=np.uint8)
                    bg_image[:] = (0,0,0)
                    annotated_image = np.where(condition, annotated_image, bg_image)
                    
                    # draw pose landmarks and connections
                    self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    # remove the landmarks we are not interested in (anything above torso) and use correct x and y positions.
                    self.get_relevant_markers(results.pose_landmarks, relevant_landmarks, box)
                    distance = depth[int(y+h/2)][int(x+w/2)]

                    self.segment_and_get_stats(image, image_person, distance, box, relevant_landmarks)
                
        return detected_bboxes
        
        
        

    def model_inference(self, image, depth):
        
        self.image_shape = image.shape
        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
              
        if len(boxes) != 0:
            detected_bboxes = self.check_frame(classes, scores, boxes, image, depth)
            return detected_bboxes
    
    def new_frame(self, image, depth, image_time_stamp):
        self.image_time_stamp = image_time_stamp
        detected_bboxes = self.model_inference(image, depth)
        self.remove_long_time_members_from_database()
        
