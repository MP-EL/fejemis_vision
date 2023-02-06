#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import ndimage
import tf
from geometry_msgs.msg import PointStamped
import rospy
import pyrealsense2 as rs2
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt

class NetDetector:
    def __init__(self) -> None:
        self.minimum_boxes = 5
        self.min_box_size = 4
        self.min_box_ratio = 0.2
        self.max_box_size = 1000
        self.patch_height = 0.4
        self.patch_nr = 10
        self.spacing = 5
        self.standard_deviation = 15
        self.tf_listener = tf.TransformListener()
        self.markerarray_pub = rospy.Publisher('marker_array', MarkerArray, queue_size=20)
        
        self.intrinsics = None
        self.old_depth = 1
    
    def approximate_dist(self, max_avg):
        #Approximate the distance to the net.
        focal_length = 566
        width = 60 #60 mm
        return (width*focal_length)/max_avg
        
    def convert_from_uvd(self, u, v, d):
        #converts pixels and depth to 3D camera coordinates.
        if d == 0:
            d = self.old_depth
        else:
            d = d
            
        u = u * (self.intrinsics.width / self.image_shape[1])
        v = v * (self.intrinsics.height / self.image_shape[0]) 
        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [u,v], d)
        X,Y,Z = result[0]*0.001, result[1]*0.001, result[2]*0.001
        
        self.old_depth = d
        return X,Y,Z
        
    def set_intrinsics(self, intrinsics):
        # Camera intrinsics are used to calculate the 3D coordinates from the depth.
        self.intrinsics = intrinsics
        
    def euclidean_dist(self, a1, a2):
        return np.sqrt((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2)
        
    def publish_net_to_ros_markerarray(self, points):
        #Publishes the net as a marker array.. Might need a custom solution to to Pointarray.
        #Polygon will not work since it will indicate net between 2 patches which could be seperated by an opening in the net.
        def clear_previous_markers():
            #Clears the previous markers from rviz.. Only usefull for visualization.
            marker_array_msg = MarkerArray()
            marker = Marker()
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array_msg.markers.append(marker)
            self.markerarray_pub.publish(marker_array_msg)
        
        clear_previous_markers()
        
        markerArray = MarkerArray()
        
        for point in points:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1] 
            marker.pose.position.z = point[2]
            
            markerArray.markers.append(marker)
        
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1
            
        self.markerarray_pub.publish(markerArray)
            
    def label_boxes(self, threshold): 

        #Morphological operations on the image. Also inverts the image, which is needed for labeling.
        threshold = cv2.dilate(threshold, None, iterations=1)
        threshold_invert = cv2.bitwise_not(threshold)
        im = cv2.erode(threshold_invert, None,iterations=1)
        
        boxes = []
        box_sizes = []
        largest_sizes = []
        
        im_label, num = ndimage.label(im)
        #Takes all the labels and turns them into minAreaRects and gathers sizes and such of each box.
        for label in range(1, num+1):
            points = np.array(np.where(im_label==label)[::-1]).T.reshape(-1,1,2).copy()
            rect = cv2.minAreaRect(points)
            box_points = cv2.boxPoints(rect)
            
            dst1 = self.euclidean_dist(box_points[0], box_points[1])
            dst2 = self.euclidean_dist(box_points[1], box_points[2])
            larg_siz = max(dst1, dst2)
            if larg_siz > 5:
                largest_sizes.append(larg_siz)
            boxes.append(np.array(box_points).astype(int))
            box_sizes.append(dst1*dst2)
        
        return boxes, box_sizes, largest_sizes
    
    def convert_points_to_world_coordinates(self, dists):
        tmp_pts = []
        for i, dist in enumerate(dists):
            if dist != 0:
                p1 = self.convert_from_uvd(self.center_points[i][0],self.center_points[i][1],dists[i])
                #Translate local camera point to world points for tracking between frames.
                msg = PointStamped()
                msg.header.stamp = self.image_time_stamp
                msg.header.frame_id = "camera_link"
                msg.point.x = p1[0]
                msg.point.y = p1[1]
                msg.point.z = p1[2]
                
                p2 = self.tf_listener.transformPoint('/map', msg)
                tmp_pts.append([p2.point.x, p2.point.y, 0])
                
        return np.array(tmp_pts)
    
    def detect_net(self, image, depth, image_time_stamp):
        self.image_time_stamp = image_time_stamp
        def filter_boxes(boxes, box_sizes, largest_sizes):
            #Removes very small boxes.
            temp_boxes = []
            temp_box_sizes = []
            temp_largest_sizes = []
            if len(boxes) > self.minimum_boxes:
                for j,(box,bs,sz) in enumerate(zip(boxes, box_sizes, largest_sizes)):
                    if bs > self.min_box_size:
                        temp_boxes.append(box)
                        temp_box_sizes.append(bs)
                        temp_largest_sizes.append(sz)
                return temp_boxes, temp_box_sizes, temp_largest_sizes
            else:
                return [], [], []
                        
        test_arr  =cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        self.image_shape = test_arr.shape
        
        #Make the patches based on the settings from init.
        bins = [[(i*test_arr.shape[1]/self.patch_nr)+self.spacing, ((i+1)*test_arr.shape[1]/self.patch_nr)-self.spacing] for i in range(self.patch_nr)]
        image_patches = [] 
        for i in range(self.patch_nr):
            image_patches.append([0, int(test_arr.shape[0]*self.patch_height), int(bins[i][0]),int(bins[i][1])])
        
        # Visuzliation things:
        test_arr_copy = test_arr.copy()
        test_arr_copy = cv2.cvtColor(test_arr_copy, cv2.COLOR_GRAY2BGR)
        
        #Center points for each patch used later to determine its location in 3D space.
        center_points = []
        for patch in image_patches:
            center_points.append([int((patch[2]+patch[3])/2), int(patch[1]/2)])
        self.center_points = center_points
        
        patch_true = []
        approx_dists = []
        
        for i, patch in enumerate(image_patches):
            # Visuzliation things:
            box = ((patch[2], patch[0]), (patch[3], patch[0]), (patch[3], patch[1]), (patch[2], patch[1]))
            test_arr_copy = cv2.polylines(test_arr_copy, np.int32([box]), True, (255,0,0), 2)
            
            # Extract patch image and depth from full image and depth
            img = test_arr[patch[0]:patch[1], patch[2]:patch[3]]
            dp = depth[patch[0]:patch[1], patch[2]:patch[3]]
            self.img_shape = img.shape
            
            # Visuzliation things:
            img_copy = img.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            
            #Obtain the threshold and label it.
            threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12) #21, 10
            boxes_og, box_sizes, largest_sizes = self.label_boxes(threshold)
            
            #Filter out small boxes.
            boxes, box_sizes, largest_sizes = filter_boxes(boxes_og, box_sizes, largest_sizes)
            
            #Histogram used for voting on the best net hole size
            largest_sizes = np.array(largest_sizes)
            bins = np.arange(0,100,5)
            hist_f, bin_edges = np.histogram(largest_sizes, range=(0,100), bins=bins)
            max_hist_idx = np.argmax(hist_f)
            n_range = [int(max_hist_idx*5), int((max_hist_idx+1)*5)]
            max_avg = np.average(largest_sizes[(largest_sizes>= n_range[0]) & (largest_sizes <= n_range[1])])
            
            exists = False
            
            # The standard deviation is chosen to best work in ASTA.
            if np.std(largest_sizes) < self.standard_deviation:
                #The only thing from these that isnt visualization is "approx_dists"
                for box in boxes_og:
                    addi = [patch[2],0]
                    tmp_box = [box[0]+ addi, box[1]+ addi, box[2]+ addi, box[3]+ addi]
                    cv2.polylines(img_copy, np.int32([box]), True, (0,255,0), 2)
                    cv2.polylines(test_arr_copy, np.int32([tmp_box]), True, [0,255,0], 2)
                dp = np.array(dp)
                filtered_depth = dp[(dp != 0) & (dp > 500)]
                depth_min = np.min(filtered_depth) if filtered_depth != [] else 0.
                calc_depth = self.approximate_dist(max_avg)
                if depth_min < calc_depth:
                    approx_dists.append(depth_min)
                    text1 = f'{depth_min/1000:.2f}m'
                else:
                    approx_dists.append(calc_depth)
                    text1 = f'{calc_depth/1000:.2f}m'
                exists = True
            else:
                for box in boxes_og:
                    addi = [patch[2],0]
                    tmp_box = [box[0]+ addi, box[1]+ addi, box[2]+ addi, box[3]+ addi]
                    cv2.polylines(img_copy, np.int32([box]), True, (255,0,0), 2)
                    cv2.polylines(test_arr_copy, np.int32([tmp_box]), True, [0,0,255], 2)
                approx_dists.append(0)
                text1 = f'NaN m'
            
            # Visuzliation things:
            patch_true.append(exists)
            if exists == True:
                cv2.putText(test_arr_copy, text1, (patch[2]+20, int(self.patch_height*test_arr_copy.shape[0]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(test_arr_copy, text1, (patch[2]+20, int(self.patch_height*test_arr_copy.shape[0]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow('test_arr_copy', test_arr_copy)
            
        pts = self.convert_points_to_world_coordinates(approx_dists)
        if pts.size != 0:
            self.publish_net_to_ros_markerarray(pts)
        return patch_true, test_arr_copy
    