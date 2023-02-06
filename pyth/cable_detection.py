#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import pyrealsense2 as rs2
from geometry_msgs.msg import PolygonStamped, Point, PointStamped
import rospy
from std_msgs.msg import Header
import tf
from skimage import feature
# To use the PolygonArray the package jsk_recognition is required. It is recommended to use this.
from jsk_recognition_msgs.msg import PolygonArray
import rospkg

class CablesDetector:
    def __init__(self) -> None:        
        #level og recursion in angle checking.
        self.depth_recursion = 3
        #size of the patches
        self.box_sizeX, self.box_sizeY = 18,18
        #largest angle allowed from patch to patch in the recursion.
        self.angle_limit = 30
        #the area of the frame we are looking at so if its [0.3, 1.0] we are looking from 0.3*height_of_frame to 1.0*height_of_frame
        self.offset = [0.5, 1.0]
        #The amount of overlap between each patch
        self.overlap = 0.5
        #minimum parallel contours in each patch (can either be 1 or 2)
        self.minimum_contours_in_cable = 2
        #The angle limit used to determine if each contour in each patch is parallel
        self.parrallel_contours_angle_limit = 5
        
        #ROS topic where the polygons containing cables are published
        self.polygon_array_pub = rospy.Publisher("Cable_array", PolygonArray, queue_size=10)

        # self.polygon_pub = rospy.Publisher('Cable_polygon', PolygonStamped, queue_size=10)
        
        #TF transform listener to make all the cable detections z=0.
        self.tf_listener = tf.TransformListener()
        
        #Setup rospkg for find folders within the ros package.
        rospack = rospkg.RosPack()
        base_path = rospack.get_path('fejemis_vision')
        # Import the mask of the robot.
        self.robot_blackout = cv2.cvtColor(cv2.imread(f'{base_path}/pyth/other_imgs/robot_section.png'), cv2.COLOR_BGR2GRAY)
        self.robot_blackout = ~self.robot_blackout

        # The image size used when running detection. Also resizes the mask.
        self.desired_image_shape = (640,480)
        self.robot_blackout = cv2.resize(self.robot_blackout, self.desired_image_shape)
        
        #Dont mind these
        self.candidates = []
        self.checked = []
        self.intrinsics = None
        self.old_dist = 1
        

    def convert_point_to_offset(self, point, mode='image'):
        #This is used to account for the offset between the image and the 
        # cables since when using self.offset some of the edges array will be removed.
        if mode == 'image':
            point = (point[0], point[1] + self.original_edges_shape[0]*self.offset[0])
            point = (int(point[0]), int(point[1]))
            return point
        if mode == 'edge':
            point = (int(point[0]), int(point[1]))
            return point
    
    def set_intrinsics(self, intrinsics):
        # Camera intrinsics are used to calculate the 3D coordinates from the depth.
        self.intrinsics = intrinsics
        
    def convert_from_uvd(self, u, v, d):
        #converts pixels and depth to 3D camera coordinates.
        if d == 0:
            d = self.old_dist
            
        u = u * (self.intrinsics.width / self.image_shape[1])
        v = v * (self.intrinsics.height / self.image_shape[0]) 
        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [u,v], d)
        X,Y,Z = result[0]*0.001, result[1]*0.001, result[2]*0.001
        
        self.old_dist = d
        return X,Y,Z

    def publish_cables_to_ros(self, cable_rects, depth):
        """publishes the cable segments individually.. If possible use polygon_array_publish instead."""
        rect_for_ros = []
        # points_for_ros = []
        for box in cable_rects:
            temp = []
            #Quick fix for having polygon points outside the image.
            for x,y in box:
                if x < 0: 
                    x = 0
                if x >= int(self.image_shape[1]): 
                    x = int(self.image_shape[1]-1)
                if y < 0: 
                    y = 0
                if y >= int(self.image_shape[0]): 
                    y = int(self.image_shape[0]-1)
                
                x1,y1,z1 = self.convert_from_uvd(x,y,depth[int(y)][int(x)])
                
                temp.append(Point(x1,y1,z1))
                
            rect_for_ros.append(temp)
        rect_for_ros = np.array(rect_for_ros)
        
        for i, rect in enumerate(rect_for_ros):
            msg = PolygonStamped()
            for point in rect:
               msg.polygon.points.append(point) 
            
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera_link"
            msg.header.seq = i
            self.polygon_pub.publish(msg)
        
    def polygon_array_publish(self, cable_rects, depth):
        rect_for_ros = []
        for box in cable_rects:
            temp = []
            #Quick fix for having polygon points outside the image.
            for x,y in box:
                if x < 0: 
                    x = 0
                if x >= int(self.image_shape[1]): 
                    x = int(self.image_shape[1]-1)
                if y < 0: 
                    y = 0
                if y >= int(self.image_shape[0]): 
                    y = int(self.image_shape[0]-1)
                
                #First convert the pixel coordinates to camera coordinates:
                x1,y1,z1 = self.convert_from_uvd(x,y,depth[int(y)][int(x)])
                
                msg = PointStamped()
                msg.header.stamp = self.image_time_stamp
                msg.header.frame_id = "camera_link"
                msg.point.x = x1
                msg.point.y = y1
                msg.point.z = z1
                
                #Secondly convert them to world coordinates.
                p1 = self.tf_listener.transformPoint('/map', msg)
                
                #Set the z axis to 0 so that the cables lie flat on the ground.
                temp.append(Point(p1.point.x,p1.point.y,0))
                
            rect_for_ros.append(temp)
            
        #make the array of polygons and publish it:    
        rect_for_ros = np.array(rect_for_ros)
        header = Header()
        msg = PolygonArray()
        polygons = []
        labels = []
        likelihoods = []
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        msg.header = header
        for i, rect in enumerate(rect_for_ros):
            msg_temp = PolygonStamped()
            msg_temp.header = header
            for point in rect:
                msg_temp.polygon.points.append(point)
            labels.append(i)
            likelihoods.append(np.random.ranf())
            polygons.append(msg_temp)
        msg.polygons = polygons
        msg.labels = labels
        msg.likelihood = likelihoods
        self.polygon_array_pub.publish(msg)
        
    def check_angle(self, angle0, angle1, angle2):
        """Returns (True, the patches that have been checked) if the the recursion reaches the desired 
        depth without failing otherwise (False,the patches that have been checked) """
        # print(f'angle0 = {angle0}, angle1 = {angle1}, angle2 = {angle2}')
        #angle of -1 means that the patch does not contain any contours.
        if angle0 != -1 and angle1 != -1 and angle2 != -1:
            #checks if the patches in either direction are between the limits or not.
            if ((angle1 < angle0+self.angle_limit) and (angle1 > angle0-self.angle_limit) and (angle2 < angle0+self.angle_limit) and (angle2 > angle0-self.angle_limit)):
                # #print('True')
                return True
            else:
                # #print('False')
                return False
        else:
            return False

    def angle_logic(self, index, depth):
        #If we reach depth 0 we return true and the cells that were checked.
        if depth == 0:
            return True, self.checked
        #If the index has already been checked.
        if index in self.checked:
            return True, self.checked
        self.checked.append(index)

        res1 = []
        res2 = []
        
        #The different permutations that will be checked based on the angle of the current patch. Each entry represents the 2 directions that will be explored.
        horizontal_angle = [[index+1, index-1],
                            [index + 1 - self.nCols, index - 1 + self.nCols],
                            [index - 1 - self.nCols, index + 1 + self.nCols],
                            [index+1, index - 1 + self.nCols],
                            [index+1, index - 1 - self.nCols],
                            [index-1, index + 1 - self.nCols],
                            [index-1, index + 1 + self.nCols],
                            [index + 1 - self.nCols, index - 1 - self.nCols],
                            [index + 1 + self.nCols, index - 1 + self.nCols]]

        forty_five_up_angle = [[index + 1-self.nCols, index - 1 + self.nCols],
                               [index+1, index-1],
                               [index - self.nCols, index + self.nCols],
                               [index + 1-self.nCols, index + self.nCols],
                               [index + 1-self.nCols, index-1],
                               [index - 1 + self.nCols, index+1],
                               [index - 1 + self.nCols, index - self.nCols],
                               [index+1, index + self.nCols],
                               [index-1, index - self.nCols]]

        vertical_angle = [[index - self.nCols, index + self.nCols], 
                          [index + 1-self.nCols, index - 1 + self.nCols], 
                          [index - 1 - self.nCols, index + 1 + self.nCols],
                          [index - self.nCols, index + 1 + self.nCols],
                          [index - self.nCols, index - 1 + self.nCols],
                          [index + self.nCols, index - 1 - self.nCols],
                          [index + self.nCols, index + 1 - self.nCols],
                          [index - 1 - self.nCols, index - 1 + self.nCols],
                          [index + 1 - self.nCols, index + 1 + self.nCols]]

        forty_five_down_angle = [[index - 1 - self.nCols, index + 1 + self.nCols], 
                                 [index+1, index-1], 
                                 [index - self.nCols, index + self.nCols],
                                 [index - 1 - self.nCols, index+1],
                                 [index - 1 - self.nCols, index + self.nCols],
                                 [index + 1 + self.nCols, index-1],
                                 [index + 1 + self.nCols, index - self.nCols],
                                 [index + 1, index - self.nCols],
                                 [index - 1, index + self.nCols]]
        #Try except here because it will fail if it checks a patch outside the image which doesnt exists.
        try:
            if self.candidates[index][0] != -1:
                #Then we just check which slot the angle fits into and check it recursively.
                if ((self.candidates[index][0] < 22.5) or (self.candidates[index][0] > 157.5)):
                    for an in vertical_angle:
                        if self.check_angle(self.candidates[index][0], self.candidates[an[0]][0], self.candidates[an[1]][0]) == True:
                            res1.append(self.angle_logic(an[0], depth-1))
                            res2.append(self.angle_logic(an[1], depth-1))
                elif (self.candidates[index][0] > 22.5) and (self.candidates[index][0] < 67.5):
                    for an in forty_five_up_angle:
                        if self.check_angle(self.candidates[index][0], self.candidates[an[0]][0], self.candidates[an[1]][0]) == True:
                            res1.append(self.angle_logic(an[0], depth-1))
                            res2.append(self.angle_logic(an[1], depth-1))
                elif ((self.candidates[index][0] > 67.5) and (self.candidates[index][0] < 112.5)):
                    for an in horizontal_angle:
                        if self.check_angle(self.candidates[index][0], self.candidates[an[0]][0], self.candidates[an[1]][0]) == True:
                            res1.append(self.angle_logic(an[0], depth-1))
                            res2.append(self.angle_logic(an[1], depth-1))
                elif (self.candidates[index][0] > 112.5) and (self.candidates[index][0] < 157.5):
                    for an in forty_five_down_angle:
                        if self.check_angle(self.candidates[index][0], self.candidates[an[0]][0], self.candidates[an[1]][0]) == True:
                            res1.append(self.angle_logic(an[0], depth-1))
                            res2.append(self.angle_logic(an[1], depth-1))
                else:
                    raise Exception("Something is wrong with the angle of the proposed cable")
                
                if len(res1) != 0 or len(res2) != 0:
                    #Only return true if every element in both res1 and res2 is true
                    return all(p==True for p in [i[0] for i in res1]) and all(p==True for p in [i[0] for i in res2]), self.checked
        except:
            return False, self.checked
        return False, self.checked
    
    def convert_frame_to_cells(self, image,slice_height,slice_width,overlap_height_ratio,overlap_width_ratio):
        """Returns a list of cells with their bounding box coordinates from the original image.

        Args:
            image (Image): The image to be sliced
            slice_height (Int): height in pixels of the cells
            slice_width (Int): width in pixels of the cells
            overlap_height_ratio (Float): the amount of overlap between each cell up and down
            overlap_width_ratio (Float): the amount of overlap between each cell left and right

        Returns:
            [Cell, bounding_box]: The desired cells of the image + the bounding box of the cell in the original image
        """
        image_height = image.shape[0]
        image_width = image.shape[1]
        cells = []
        ncols = 0
        nrows = 0
        y_max = y_min = 0
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
        while y_max < image_height:
            nrows = nrows + 1
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                ncols = ncols + 1
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    cells.append([image[ymin:ymax,xmin:xmax],[xmin,ymin,xmax,ymax]])
                else:
                    cells.append([image[y_min:y_max,x_min:x_max],[x_min,y_min,x_max,y_max]])
                    
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        ncols = int(ncols / nrows)
        return cells, nrows, ncols
    
    def parallel_contours(self, contours, cell):
        def check_parallel(data, limit):
            """Returns True if there is no outliers larger than the set limit else returns false"""
            if len(data) >= self.minimum_contours_in_cable:
                d = np.abs(data - np.median(data))
                if np.max(d) > limit:
                    return False
                else:
                    return True
            else:
                return False

        def correct_points_max_distance(points):
            #Generates 2 points that are the furthest from eachother, 
            # this is done because it gives more stable calculations when calculating angles for the contours to check if they are parallel.
            dists = cdist(points, points)
            i,j = np.unravel_index(dists.argmax(), dists.shape)
            return [points[i], points[j]]
        
        angles = []
        for contour in contours:
            if len(contour)>5:
                contour = np.unique(contour.squeeze(),axis=0)
                [p1, p2] = correct_points_max_distance(contour)
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                calc = np.arctan2(dy,dx)
                
                angles.append(calc)
                
        angles = np.degrees(angles)
        if check_parallel(angles, self.parrallel_contours_angle_limit) == True: 
            res = int(np.average(angles)+90) #+90 to make it fit in with the assumptions made otherwise.
            return res
        else:
            return -1
        
        
    def pre_filters(self, image):
        image = cv2.GaussianBlur(image, (3,3), 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def edges_get(self, img):
        return np.uint8(feature.canny(img, sigma=1.3) * 255)

    def black_out_robot(self, img):
        #masks the robot.
        thresh = cv2.threshold(self.robot_blackout, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        img = cv2.bitwise_and(img, img, mask = thresh)
        return img

    def narrow_to_include_depth(self, image, depth, amounts = [0, 0.98, 0.05, 0.95]):
        image = image[int(depth.shape[0]*amounts[0]):int(image.shape[0]*amounts[1]), int(image.shape[1]*amounts[2]):int(image.shape[1]*amounts[3])]
        depth = depth[int(depth.shape[0]*amounts[0]):int(depth.shape[0]*amounts[1]), int(depth.shape[1]*amounts[2]):int(depth.shape[1]*amounts[3])]
        return image, depth
    
    def resize_and_cut(self, image, depth):
        image, depth = self.narrow_to_include_depth(image, depth)
        depth = cv2.resize(depth, self.desired_image_shape)
        image = cv2.resize(image, self.desired_image_shape)
        return image, depth
    
    def find_cable(self, image, depth_map, image_time_stamp):
        #Time stamp used for the transform lookup.
        self.image_time_stamp = image_time_stamp 
        
        image, depth_map = self.resize_and_cut(image, depth_map)# If this is not done the polygons returned to ros will have 0 in them sometimes because the depth sensor has 0 around the edges.
        
        im = self.pre_filters(image)
        
        edges = self.edges_get(im)
        edges = self.black_out_robot(edges)
        self.image_shape = image.shape
            
        self.original_edges_shape = edges.shape
        #Cut out the part of the edges that we actually look at based on self.offset
        edges = edges[int(edges.shape[0]*self.offset[0]):int(edges.shape[0]*self.offset[1]), 0:int(edges.shape[1])]
        
        #Generate patches
        self.cells, self.nRows, self.nCols = self.convert_frame_to_cells(edges.copy(), self.box_sizeX, self.box_sizeY, self.overlap, self.overlap)
        
        self.candidates = []
        cables = []
        cable_rects = []
        #First we get the contours from each patchand check if the patch 
        # contains 2 parallel contours  if yes save the angle if no set the angle to -1
        for cell in self.cells:
            contours, _ = cv2.findContours(cell[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                angle = self.parallel_contours(contours, cell[0])
                if angle != -1:
                    self.candidates.append([angle, cell])
                else:
                    self.candidates.append([-1,cell])
            else:
                self.candidates.append([-1,cell])
                
        cable_outline = np.zeros(edges.shape, dtype='uint8')
        
        for index, candidate in enumerate(self.candidates): 
            if candidate[0] != -1:
                self.checked = []
                #Do the recursive checking of each angle candidate from above.
                res, self.checked = self.angle_logic(index, self.depth_recursion)
                
                #if the angle passes the angle check the box is put into the cable_outline array which is essentially 
                # a black and white image where each patch is colored white if it contains a cable and black if it doesnt.
                if res == True:
                    cables.append(self.checked)
                    for part in self.checked: 
                        box = self.candidates[part][1][1]
                        cable_outline = cv2.rectangle(cable_outline, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

        cable_outline_copy = cv2.cvtColor(cable_outline.copy(), cv2.COLOR_GRAY2BGR)
        im3 = image.copy()
        #Then the contours of the cable_outline image/array is found and used to generate the polygons which are published to ROS.
        contours2, _ = cv2.findContours(cable_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour2 in contours2:
            cable_rects.append(cv2.boxPoints(cv2.minAreaRect(contour2)))
        for cable_rect in cable_rects:
            cable_outline_copy = cv2.polylines(cable_outline_copy, np.int32([cable_rect]), True, (36,255,12), 1)
            for index, cab in enumerate(cable_rect):
                cable_rect[index] = self.convert_point_to_offset((cab[0], cab[1]))

        for cable_rect in cable_rects:
            im3 = cv2.polylines(im3, np.int32([cable_rect]), True, (36,255,12), 1)
        # cv2.imshow('cable_outline', cable_outline_copy)
        # cv2.imshow('im3', im3)

        #Lastly if there are any cable polygons they are published to ROS
        if len(cable_rects) > 0:
            self.polygon_array_publish(cable_rects, depth_map)
        # if len(cable_rects) > 0:
        #     self.publish_cables_to_ros(cable_rects, depth_map)
        #And we return the cables since they can be used for visualization.
        return cables