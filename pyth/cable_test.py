from cable_detection import CablesDetector
import glob
import cv2
import numpy as np
import time
import pyrealsense2 as rs2
import rospkg
import rospy

def main():
    VISUAL = True
    rospy.init_node('fejemisVisionTest')
    CBdet = CablesDetector()
    
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('fejemis_vision')
    
    imgs = sorted(glob.glob(f'{base_path}/pyth/cable_test_images/*_image.jpg'))
    depths = sorted(glob.glob(f'{base_path}/pyth/cable_test_images/*_depth.PNG'))
    labels = sorted(glob.glob(f'{base_path}/pyth/cable_test_images/*_label.jpg'))
    
    offsets_screen = [[0.0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
    avg_precision1 = []
    avg_recall1 = []
    avg_accuracy1 = []
    avg_precision2 = []
    avg_recall2 = []
    avg_accuracy2 = []
    avg_precision3 = []
    avg_recall3 = []
    avg_accuracy3 = []
    avg_precision4 = []
    avg_recall4 = []
    avg_accuracy4 = []
    avg_overall_precision = []
    avg_overall_recall = []
    avg_overall_accuracy = []
    
    intrinsics = rs2.intrinsics()
    intrinsics.width = 1280
    intrinsics.height = 720
    intrinsics.ppx = 629.3971557617188
    intrinsics.ppy = 358.9218444824219
    intrinsics.fx = 639.0593872070312
    intrinsics.fy = 638.0255126953125
    intrinsics.model = rs2.distortion.brown_conrady
    intrinsics.coeffs = [-0.05425437167286873, 0.06392815709114075, -0.0004175958165433258, 7.225961599033326e-05, -0.020541394129395485]
    CBdet.set_intrinsics(intrinsics)
    
    avg_time = []
    for i, (img, depth_frame, label) in enumerate(zip(imgs, depths, labels)):
        print(f'{i+1}/{len(imgs)}')
        # print(i, img, label)
        im = cv2.imread(img)
        lab = cv2.imread(label)
        depth = cv2.imread(depth_frame, cv2.CV_16UC1)
        st = time.time()
        image_time_stamp = rospy.Time()
        cables = CBdet.find_cable(im, depth, image_time_stamp)
        et = time.time()
        lab = ~lab
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (640,480))
        depth = cv2.resize(depth, (640,480))
        im_og = im.copy()
        im_og, depth = CBdet.resize_and_cut(im_og, depth)
        im3 = im_og.copy()
        lab, _ = CBdet.resize_and_cut(lab, depth)
        
        # edges2 = edges[int(edges.shape[0]*offset[0]):int(edges.shape[0]*offset[1]), 0:int(edges.shape[1])]
        lab = lab[int(lab.shape[0]*CBdet.offset[0]):int(lab.shape[0]*CBdet.offset[1]), 0:int(lab.shape[1])]
        # im = im[int(im.shape[0]*offset[0]):int(im.shape[0]*offset[1]), 0:int(im.shape[1])]
        
        lab_cells, nrows, ncols = CBdet.convert_frame_to_cells(lab, CBdet.box_sizeX, CBdet.box_sizeY, CBdet.overlap, CBdet.overlap)
        
        lab_checked = []
        for i, cell in enumerate(lab_cells):
            if np.sum(cell[0]) > 35*255:
               lab_checked.append(i)
        cab = []
        for outer in cables:
            for cable in outer:
                cab.append(cable)
        cab = np.unique(cab)
        per_1_precision = []
        per_1_recall = []
        per_1_accuracy = []
        
        per_2_precision = []
        per_2_recall = []
        per_2_accuracy = []
        
        per_3_precision = []
        per_3_recall = []
        per_3_accuracy = []
        
        per_4_precision = []
        per_4_recall = []
        per_4_accuracy = []
        
        colors = [(0,0,255), (0,255,255), (255,255,125), (0,255,0)]
        # colors = [(0,0,255), (0,0,255), (0,0,255), (0,0,255)]
        
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        # edges2 = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)
        im2 = im_og.copy()
        t1 = False
        t2 = False
        t3 = False
        t4 = False
        for i, offs in enumerate(offsets_screen):
            lower_cell = int(nrows*ncols*offs[0])
            upper_cell = int(nrows*ncols*offs[1])
            temp_ca = []
            temp_la = []
            for ca in cab:
                if int(ca) <= upper_cell and int(ca) > lower_cell:
                    temp_ca.append(ca)
                    box = CBdet.candidates[int(ca)][1][1]
                    im2 = cv2.rectangle(im2, CBdet.convert_point_to_offset((box[0],box[1])), CBdet.convert_point_to_offset((box[2],box[3])), colors[i], 1) 
                    # edges2 = cv2.rectangle(edges2, CBdet.convert_point_to_offset((box[0],box[1]),'edge'), CBdet.convert_point_to_offset((box[2],box[3]),'edge'), colors[i], 1) 
            for lab_c in lab_checked:
                if int(lab_c) <= upper_cell and int(lab_c) > lower_cell:
                    temp_la.append(lab_c)
                    box = CBdet.candidates[lab_c][1][1]
                    im_og = cv2.rectangle(im_og, CBdet.convert_point_to_offset((box[0],box[1])), CBdet.convert_point_to_offset((box[2],box[3])), colors[i], 1)
            
            tp = 1
            tn = 1
            fn = 1
            fp = 1
            
            for la in temp_la:
                if la in temp_ca:
                    tp += 1
                if la not in temp_ca:
                    fn += 1
            for c in temp_ca:
                if c not in lab_checked:
                    fp += 1
                    
            if tp != 1 or fp != 1 or fn != 1:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                accuracy = (tp+tn)/(tp+tn+fp+fn)#(2*tp)/(2*tp+fp+fn)
                
                # print(f'precision: {precision:.2f}, recall: {recall:.2f}, {i}')
                
                if i == 0:
                    t1 = True
                    per_1_precision.append(precision)
                    per_1_recall.append(recall)
                    per_1_accuracy.append(accuracy)
                if i == 1:
                    t2 = True
                    per_2_precision.append(precision) 
                    per_2_recall.append(recall)
                    per_2_accuracy.append(accuracy)
                if i == 2:
                    t3 = True
                    per_3_precision.append(precision)   
                    per_3_recall.append(recall)
                    per_3_accuracy.append(accuracy)
                if i == 3:
                    t4 = True
                    per_4_precision.append(precision)
                    per_4_recall.append(recall)
                    per_4_accuracy.append(accuracy)
            else:
                # print('catagory unrepresented by data', i)
                pass
        if t1 == True:
            avg_precision1.append(np.average(per_1_precision))
            avg_recall1.append(np.average(per_1_recall))
            avg_overall_precision.append(np.average(per_1_precision))
            avg_overall_recall.append(np.average(per_1_recall))
            avg_accuracy1.append(np.average(per_1_accuracy))
            avg_overall_accuracy.append(np.average(per_1_accuracy))
        if t2 == True:
            avg_precision2.append(np.average(per_2_precision))
            avg_recall2.append(np.average(per_2_recall))
            avg_overall_precision.append(np.average(per_2_precision))
            avg_overall_recall.append(np.average(per_2_recall))
            avg_accuracy2.append(np.average(per_2_accuracy))
            avg_overall_accuracy.append(np.average(per_2_accuracy))
        if t3 == True:
            avg_precision3.append(np.average(per_3_precision))
            avg_recall3.append(np.average(per_3_recall))
            avg_overall_precision.append(np.average(per_3_precision))
            avg_overall_recall.append(np.average(per_3_recall))
            avg_accuracy3.append(np.average(per_3_accuracy))
            avg_overall_accuracy.append(np.average(per_3_accuracy))
        if t4 == True:
            avg_precision4.append(np.average(per_4_precision))
            avg_recall4.append(np.average(per_4_recall))
            avg_overall_precision.append(np.average(per_4_precision))
            avg_overall_recall.append(np.average(per_4_recall))
            avg_accuracy4.append(np.average(per_4_accuracy))
            avg_overall_accuracy.append(np.average(per_4_accuracy))

        
        timed = (et-st)*1000
        avg_time.append(timed)

        if VISUAL == True:
            image_shape = im_og.shape
            start_offset = CBdet.offset[0]*image_shape[0]-1
            new_offset_screen = offsets_screen.copy()
            new_offset_screen.append([0.0, 0.0])
            for i, point in new_offset_screen:
                print(point)
                pixel = (int((image_shape[1]/6)*5), int((point * (image_shape[0]-start_offset))+start_offset)-1)
                print(pixel , depth[pixel[1]][pixel[0]])
                cv2.circle(im2, pixel, 3, (255,0,255), 1)
                text = f'{depth[pixel[1]][pixel[0]]*0.001:.2f}'
                cv2.putText(im2, text, pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            stacked = np.hstack([im_og, im2])
            cv2.imshow('labeled image', im_og)
            cv2.imshow('detected image', im2)
            cv2.imshow('<-- labeled image || cable detected image -->', stacked)
            # cv2.imshow('labeled image', im_og)
            # cv2.imshow('cable detected image', im2)
            # cv2.imshow('edges', edges2)
            
            k = cv2.waitKey(0)
            if k == 27:  # close on ESC key
                cv2.destroyAllWindows()  
                quit()
    print(f'for settings: depth: {CBdet.depth_recursion}, angle: {CBdet.angle_limit}')
    print(f'for offset: {offsets_screen[0]}')
    print(f'avg_precision: {np.average(avg_precision1):.2f} ({np.max(avg_precision1):.2f} : {np.min(avg_precision1):.2f}), avg_recall: {np.average(avg_recall1):.2f} ({np.max(avg_recall1):.2f} : {np.min(avg_recall1):.2f}), avg_accuracy: {np.average(avg_accuracy1):.2f} ({np.max(avg_accuracy1):.2f} : {np.min(avg_accuracy1):.2f})')
    print(f'for offset: {offsets_screen[1]}')
    print(f'avg_precision: {np.average(avg_precision2):.2f} ({np.max(avg_precision2):.2f} : {np.min(avg_precision2):.2f}), avg_recall: {np.average(avg_recall2):.2f} ({np.max(avg_recall2):.2f} : {np.min(avg_recall2):.2f}), avg_accuracy: {np.average(avg_accuracy2):.2f} ({np.max(avg_accuracy2):.2f} : {np.min(avg_accuracy2):.2f})')
    print(f'for offset: {offsets_screen[2]}')
    print(f'avg_precision: {np.average(avg_precision3):.2f} ({np.max(avg_precision3):.2f} : {np.min(avg_precision3):.2f}), avg_recall: {np.average(avg_recall3):.2f} ({np.max(avg_recall3):.2f} : {np.min(avg_recall3):.2f}), avg_accuracy: {np.average(avg_accuracy3):.2f} ({np.max(avg_accuracy3):.2f} : {np.min(avg_accuracy3):.2f})')
    print(f'for offset: {offsets_screen[3]}')
    print(f'avg_precision: {np.average(avg_precision4):.2f} ({np.max(avg_precision4):.2f} : {np.min(avg_precision4):.2f}), avg_recall: {np.average(avg_recall4):.2f} ({np.max(avg_recall4):.2f} : {np.min(avg_recall4):.2f}), avg_accuracy: {np.average(avg_accuracy4):.2f} ({np.max(avg_accuracy4):.2f} : {np.min(avg_accuracy4):.2f})')
    print('==============================================================')
    print(f'for all offsets')
    print(f'avg_precision: {np.average(avg_overall_precision):.2f} ({np.max(avg_overall_precision):.2f} : {np.min(avg_overall_precision):.2f}), avg_recall: {np.average(avg_overall_recall):.2f} ({np.max(avg_overall_recall):.2f} : {np.min(avg_overall_recall):.2f}), avg_accuracy: {np.average(avg_overall_accuracy):.2f} ({np.max(avg_overall_accuracy):.2f} : {np.min(avg_overall_accuracy):.2f})')
    print('==============================================================')
    print(f'timings: avg_time: {np.average(avg_time):.0f} ({np.max(avg_time):.0f} : {np.min(avg_time):.0f})')
    
        
if __name__ == '__main__':
    main()
    
    
    
