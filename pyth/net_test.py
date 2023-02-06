#!/usr/bin/env python3
from net_detection import NetDetector
# from main import MasterNode
import cv2
import numpy as np
import glob
import rospy
import pyrealsense2 as rs2
import json
import time
import rospkg

def main():
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('fejemis_vision')
    stds = [25]#[5, 10, 15, 20, 25, 30, 35, 40]
    for standard_deviation in stds:
        
        VISUAL = True
        rospy.init_node('fejemisVisionTest')
        NTdet = NetDetector()
        # node = MasterNode()
        imgs1 = sorted(glob.glob(f'{base_path}/pyth/net_test_images/*.jpg'))
        imgs2 = sorted(glob.glob(f'{base_path}/pyth/net_test_images/*.PNG'))
        labels = sorted(glob.glob(f'{base_path}/pyth/net_test_images/*.txt'))
        
        intrinsics = rs2.intrinsics()
        intrinsics.width = 1280
        intrinsics.height = 720
        intrinsics.ppx = 629.3971557617188
        intrinsics.ppy = 358.9218444824219
        intrinsics.fx = 639.0593872070312
        intrinsics.fy = 638.0255126953125
        intrinsics.model = rs2.distortion.brown_conrady
        intrinsics.coeffs = [-0.05425437167286873, 0.06392815709114075, -0.0004175958165433258, 7.225961599033326e-05, -0.020541394129395485]
        NTdet.set_intrinsics(intrinsics)
        NTdet.standard_deviation = standard_deviation
        tp = 1
        fn = 1
        tn = 1
        fp = 1
        timings = []
        
        for i, (im, dep, lab) in enumerate(zip(imgs1, imgs2, labels)):
            # print(f'run: {i+1} / {len(imgs1)}')
            with open(lab, 'r') as f:
                label = json.loads(f.read())
            label = [bool(i) for i in label]
            
            img = cv2.imread(im)
            depth = cv2.imread(dep, cv2.CV_16UC1)
            depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            
            
            st = time.time()
            image_time_stamp = rospy.Time()
            output, img2 = NTdet.detect_net(img, depth, image_time_stamp)
            et = time.time()
            timings.append((et-st)*1000)
            
            for (y, x) in zip(output, label):
                if y == True and x == True:
                    tp += 1
                if y != True and x == True:
                    fn += 1
                if y == True and x != True:
                    fp += 1
                if y == False and x == False:
                    tn += 1
            
            if VISUAL == True:
                cv2.imshow('depth', depth_map)
                cv2.imshow('img', img2)
                k = cv2.waitKey(0)
                if k == 27:  # close on ESC key
                    cv2.destroyAllWindows()  
                    quit()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        # MCC_score = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        
        print(f'{standard_deviation=:.2f}, {precision=:.2f}, {recall=:.2f}, {accuracy=:.2f}, runtime avg={np.average(timings):.2f} ms, (min : max) runtimes: ({np.min(timings):.2f} : {np.max(timings):.2f})')

def main2():
    VISUAL = True
    rospy.init_node('fejemisVisionTest')
    NTdet = NetDetector()
    # node = MasterNode()
    # imgs1 = sorted(glob.glob('net_test_images/*.jpg'))
    # imgs2 = sorted(glob.glob('net_test_images/*.PNG'))
    # imgs1 = sorted(glob.glob('/home/mikkel/ros_pics/net_with_robot/test/*.jpg'))
    # imgs2 = sorted(glob.glob('/home/mikkel/ros_pics/net_with_robot/test/*.PNG'))
    # labels = sorted(glob.glob('net_test_images/*.txt'))
    imgs1 = sorted(glob.glob('/home/mikkel/ros_pics/net_test_7/*.jpg'))
    imgs2 = sorted(glob.glob('/home/mikkel/ros_pics/net_test_7/*.PNG'))
    print(imgs1)

    intrinsics = rs2.intrinsics()
    intrinsics.width = 1280
    intrinsics.height = 720
    intrinsics.ppx = 629.3971557617188
    intrinsics.ppy = 358.9218444824219
    intrinsics.fx = 639.0593872070312
    intrinsics.fy = 638.0255126953125
    intrinsics.model = rs2.distortion.brown_conrady
    intrinsics.coeffs = [-0.05425437167286873, 0.06392815709114075, -0.0004175958165433258, 7.225961599033326e-05, -0.020541394129395485]
    NTdet.set_intrinsics(intrinsics)
    
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    timings = []
    
    for i, (im, dep) in enumerate(zip(imgs1, imgs2)):
        print(f'run: {i+1} / {len(imgs1)}')
        
        img = cv2.imread(im)
        depth = cv2.imread(dep, cv2.CV_16UC1)
        # depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        
        st = time.time()
        image_time_stamp = rospy.Time()
        output, img2 = NTdet.detect_net(img, depth, image_time_stamp)
        et = time.time()
        timings.append((et-st)*1000)

        
        if VISUAL == True:
            # cv2.imshow('depth', depth_map)
            cv2.imshow('img', img2)
            k = cv2.waitKey(0)
            if k == 27:  # close on ESC key
                cv2.destroyAllWindows()  
                quit()
    print(f'runtime avg={np.average(timings):.2f} ms, (min : max) runtimes: ({np.min(timings):.2f} : {np.max(timings):.2f})')


main2()