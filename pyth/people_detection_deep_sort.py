#!/usr/bin/env python3
import cv2
import os
import tf
import rospkg
from dataclasses import dataclass

from deepsort.application_util import preprocessing
from deepsort.deep_sort import nn_matching
from deepsort.deep_sort.detection import Detection
from deepsort.deep_sort.tracker import Tracker
from deepsort.tools import generate_detections as gdet

@dataclass
class Person:
    right_thigh: float
    left_thigh: float
    right_calve: float
    left_calve: float
    torso: float
    x: float
    y: float
    z: float
    image_person: list

class HumanDetectorDL:
    def __init__(self):
        rospack = rospkg.RosPack()
        base_path = rospack.get_path('fejemis_vision')
        
        self.net = cv2.dnn.readNet(f"{base_path}/deep_learning/yolov4-tiny.weights", f"{base_path}/deep_learning/yolov4-tiny.cfg")
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.5

        # DeepSORT -> Initializing tracker.
        max_cosine_distance = 0.1
        nn_budget = None
        model_filename = f"{base_path}/deep_learning/mars-small128.pb"
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        
        with open(f"{base_path}/deep_learning/classes.txt", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        
        self.people_per_frame_limit = 3
        self.intrinsics = None
        self.database = []
        self.tf_listener = tf.TransformListener()
    
    def set_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics
            
    def model_inference(self, image, depth):
        
        self.image_shape = image.shape
        # start = time.time()
        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        # end = time.time()
        
        detected_bboxes = []
        
        # DeepSORT -> Getting appearance features of the object.
        features = self.encoder(image, boxes)
        # DeepSORT -> Storing all the required info in a list.
        detections = []
        for bbox, score, feature, cla in zip(boxes, scores, features, classes):
            if cla == 0:
                detections.append(Detection(bbox, score, feature))
        # detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]

        # DeepSORT -> Predicting Tracks.
        self.tracker.predict()
        self.tracker.update(detections)
        #track_time = time.time() - prev_time

        # DeepSORT -> Plotting the tracks.
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
            bbox = list(track.to_tlbr())
            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            txt = 'id: ' + str(track.track_id)
            (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_SIMPLEX,1,1)
            top_left = tuple(map(int,[int(bbox[0]),int(bbox[1])-(label_height+baseline)]))
            top_right = tuple(map(int,[int(bbox[0])+label_width,int(bbox[1])]))
            org = tuple(map(int,[int(bbox[0]),int(bbox[3])-baseline]))

            image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 3)
            # image = cv2.rectangle(image, top_left, top_right, (255,0,0), -1)
            image = cv2.putText(image, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 6)
            # cv2.imshow('image', image)
            # k = cv2.waitKey(0)
            # if k==27:    # Esc key to stop
            #     cv2.destroyAllWindows()
            #     quit()        
        
        return detected_bboxes
    
    def new_frame(self, image, depth, image_time_stamp):
        self.image_time_stamp = image_time_stamp
        detected_bboxes = self.model_inference(image, depth)
        
