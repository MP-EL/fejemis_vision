#!/usr/bin/env python3
from people_detection_depth_segmentation import HumanDetectorDL as HMDET_DEPTH
from people_detection_simple_segmentation import HumanDetectorDL as HMDET_SIMPLE
from people_detection_pose_segmentation import HumanDetectorDL as HMDET_POSE
from main import MasterNode
import cv2
import glob
import rospy
import pyrealsense2 as rs2
import rospkg

def main():
    VISUAL = True
    rospy.init_node('fejemisVisionTest')
    node = MasterNode()
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('fejemis_vision')
    
    list_of_detectors = [HMDET_POSE(), HMDET_DEPTH(), HMDET_SIMPLE()]
    for detector in list_of_detectors:
        cv2.destroyAllWindows()
        HMdet = detector
        
        imgs1 = sorted(glob.glob(f'{base_path}/pyth/people_test_images/*.jpg'))
        imgs2 = sorted(glob.glob(f'{base_path}/pyth/people_test_images/*.PNG'))
        
        intrinsics = rs2.intrinsics()
        intrinsics.width = 1280
        intrinsics.height = 720
        intrinsics.ppx = 629.3971557617188
        intrinsics.ppy = 358.9218444824219
        intrinsics.fx = 639.0593872070312
        intrinsics.fy = 638.0255126953125
        intrinsics.model = rs2.distortion.brown_conrady
        intrinsics.coeffs = [-0.05425437167286873, 0.06392815709114075, -0.0004175958165433258, 7.225961599033326e-05, -0.020541394129395485]
        HMdet.set_intrinsics(intrinsics)
        
        for i, (im, dep) in enumerate(zip(imgs1, imgs2)):
            print(f'run: {i+1} / {len(imgs1)}')
            
            img = cv2.imread(im)
            depth = cv2.imread(dep, cv2.CV_16UC1)
            
            
            image_time_stamp = rospy.Time()
            # head,tail = os.path.split(im)
            # fake_time = float(tail.replace('_image.jpg', ''))
            HMdet.new_frame(img, depth, image_time_stamp)
            
            if VISUAL == True:
                cv2.imshow('img', img)
                k = cv2.waitKey(0)
                if k == 27:  # close on ESC key
                    cv2.destroyAllWindows()  
                    quit()
    
main()