'''
Script that contains various detection functions for the KITTI dataset
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from kitti_utils import *
from PIL import Image


def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    ''' Obtains detected object centers projected to uvz camera coordinates. 
        Starts by associating LiDAR uvz coordinates to detected object centers,
        once a match is found, the coordiantes are transformed to the uvz
        camera reference and added to the bboxes array.

        NOTE: The image is modified in place so there is no need to return it.

        Inputs:
          image - input image for detection 
          velo_uvz - LiDAR coordinates projected to camera reference
          bboxes - xyxy bounding boxes form detections from yolov5 model output
          draw - (_Bool) draw measured depths on image
        Outputs:
          bboxes_out - modified array containing the object centers projected
                       to uvz image coordinates
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = np.round(bbox[0:2]).astype(int)
        pt2 = np.round(bbox[2:4]).astype(int)

        # get center location of the object on the image
        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u)) 
                              - np.array([[obj_x_center, obj_y_center]]).T)
        
        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])
        
        # add velo projections (u, v, z) to bboxes_out
        bboxes_out[i, -3:] = uvz_location

        # draw depth on image at center of each bounding box
        # This is depth as perceived by the camera
        if draw:
            object_center = (np.round(obj_y_center).astype(int), 
                             np.round(obj_x_center).astype(int))
            cv2.putText(image, 
                        '{0:.2f} m'.format(velo_depth), 
                        object_center, # top left
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font scale
                        (255, 0, 0), 2, cv2.LINE_AA)    
            
    return bboxes_out

  
  

def get_detection_coordinates(image, bin_path, model,T_velo_cam2, draw_boxes=True, draw_depth=True):
    """
    Obtains detections for the input image, along with the coordinates of 
    the detected object centers in:
    - Camera with depth (uvz)
    - LiDAR/velo (xyz)
    - GPS/IMU  (xyz)
    """
    # Perform detection
    detections = model(image)
    
    # Filter detections based on confidence and class indices
    desired_classes = [0, 1, 2, 3, 5, 7]  # Only person, bicycle, car, motorcycle, bus, truck
    confidence_threshold = 0.5
    filtered_boxes = []
    for box in detections[0].boxes.data.cpu().numpy():  # [x1, y1, x2, y2, confidence, class]
        confidence, cls = box[4], int(box[5])
        if confidence >= confidence_threshold and cls in desired_classes:
            filtered_boxes.append(box)
    
    filtered_boxes = np.array(filtered_boxes)
    
    # Draw boxes on the image
    if draw_boxes:
        if len(filtered_boxes) > 0:
            for box in filtered_boxes:
                x1, y1, x2, y2, conf, cls = box
                label = f"{model.names[int(cls)]} {conf:.2f}"
                # Draw rectangle and label on the image
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                image = cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image_2dbox = Image.fromarray(image)
            image_2dbox.show()  # Opens the image in the default viewer
            # plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()
        else:
            print("No detections met the criteria.")
    
    # Project LiDAR points to camera space
    velo_uvz = project_velobin2uvz(bin_path, T_velo_cam2, image, remove_plane=False)

    # Map bounding boxes to uvz centers
    if len(filtered_boxes) > 0:
        bboxes = get_uvz_centers(image, velo_uvz, filtered_boxes, draw=draw_depth)
    else:
        bboxes = []

    return bboxes, velo_uvz
  
  
def get_imu_xyz(image, bin_path, model, T_velo_cam, T_cam_imu):
     ''' Obtains (x,y,z) location referenced to IMU 
         Camera reference specified by image and Transformation Matrices, 
         Can use any camera as the reference
         '''
     # get detections and object centers in uvz
     bboxes, velo_uvz = get_detection_coordinates(image, 
                                                  bin_path, 
                                                  model, 
                                                  T_velo_cam,
                                                  draw_boxes=False,
                                                  draw_depth=False)

     # get transformed coordinates of object centers
     uvz = bboxes[:, -3:]

     # transform to IMU (u,v,z)
     imu_xyz = transform_uvz(uvz, T_cam_imu)
      
     return imu_xyz
