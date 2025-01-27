import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymap3d as pm
from PIL import Image
import matplotlib.image as mpimg

plt.rcParams["figure.figsize"] = (20, 10)
from ultralytics import YOLO
from sklearn import linear_model

import folium
from sklearn import linear_model
from sklearn.cluster import DBSCAN


from pkg.kitti_utils import *
from pkg.kitti_detection_utils import *


###model
model = YOLO("yolov8n.pt")
model.conf = 0.5  # confidence threshold
model.iou = 0.5 # NMS IoU threshold
model.classes = [0, 1, 2, 3, 5, 7]  # only person, bicycle, car, motorcycle, bus, truck


# DATA_PATH = r'./../visionNav/fusion/dataset/2011_10_03_drive_0047_sync'
DATA_PATH = r'./dataset/2011_10_03_drive_0047_sync'

# get RGB camera data
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))

# get GPS/IMU data
oxts_paths = sorted(glob(os.path.join(DATA_PATH, r'oxts/data**/*.txt')))

print(f"Number of left images: {len(left_image_paths)}")
print(f"Number of right images: {len(right_image_paths)}")
print(f"Number of LiDAR point clouds: {len(bin_paths)}")
print(f"Number of GPS/IMU frames: {len(oxts_paths)}")


###########################################
with open('./dataset/2011_10_03_calib/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# get projection matrices (rectified left camera --> left camera (u,v,z))
P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))

# get rectified rotation matrices (left camera --> rectified left camera)
R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))

# add (0,0,0) translation and convert to homogeneous coordinates
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)


# get rigid transformation from Camera 0 (ref) to Camera 2
R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))

# get cam0 to cam2 rigid body transformation in homogeneous coordinates
T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)

T_velo_ref0 = get_rigid_transformation(r'./dataset/2011_10_03_calib/calib_velo_to_cam.txt')
T_imu_velo = get_rigid_transformation(r'./dataset/2011_10_03_calib/calib_imu_to_velo.txt')

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 

# transform from IMU to left color camera (shape 3x4)
T_imu_cam2 = T_velo_cam2 @ T_imu_velo

# homogeneous transform from left color camera to IMU (shape: 4x4)
T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0,0,0,1], axis=0)) 


# get projection matrices
P_left = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))
P_right = np.array([float(x) for x in calib[33].strip().split(' ')[1:]]).reshape((3,4))

# get rectified rotation matrices
R_left_rect = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))
R_right_rect = np.array([float(x) for x in calib[32].strip().split(' ')[1:]]).reshape((3, 3,))

R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0], axis=0)
R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0,1], axis=1)

K_left, R_left, T_left = decompose_projection_matrix(P_left)
K_right, R_right, T_right = decompose_projection_matrix(P_right)

T_mat = P_left @ R_left_rect @ T_velo_ref0

with open(r'./dataset/2011_10_03_calib/calib_velo_to_cam.txt', 'r') as f:
    calib = f.readlines()

R_cam_velo = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
t_cam_velo = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

T_cam_velo = np.vstack((np.hstack((R_cam_velo, t_cam_velo)),
                        np.array([0, 0, 0, 1])))

# matrix to transform from velo (LiDAR) to left color camera
T_mat = P_left @ R_left_rect @ T_cam_velo
##################################################################################


def decompose_projection_matrix(P):    
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T/T[3]

    return K, R, T

K_left, R_left, T_left = decompose_projection_matrix(P_left)
K_right, R_right, T_right = decompose_projection_matrix(P_right)


def imu2geodetic(x, y, z, lat0, lon0, alt0, heading0):
    # convert to RAE
    rng = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
    el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90 
    
    # convert to geodetic
    lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)

    # convert to numpy array
    lla = np.vstack((lla[0], lla[1], lla[2])).T

    return lla

def draw_scenario(canvas, imu_xyz,canvas_height,sf=12):

    # get consistent center for ego vehicle
    ego_center = (250, int(canvas_height*0.95))

    # get rectangle coordiantes for ego vehicle
    ego_x1 = ego_center[0] - 5
    ego_y1 = ego_center[1] - 10
    ego_x2 = ego_center[0] + 5
    ego_y2 = ego_center[1] + 10
    # draw ego vehicle
    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (0, 255, 0), -1)

    # draw detected objects
    for val in imu_xyz:
        obj_center = (ego_center[0] - sf*int(np.round(val[1])),
                      ego_center[1] - sf*int(np.round(val[0])))
        # cv2.circle(canvas, obj_center, 5, (255, 0, 0), -1);

        # get object rectangle coordinates
        obj_x1 = obj_center[0] - 5
        obj_y1 = obj_center[1] - 10
        obj_x2 = obj_center[0] + 5
        obj_y2 = obj_center[1] + 10

        cv2.rectangle(canvas, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 0), -1)
    return canvas


def project_velo2cam(lidar_bin, image, remove_plane=True):
    ''' Projects LiDAR point cloud onto the image coordinate frame '''

    # get homogeneous LiDAR points from binn file
    velo_points = bin2h_velo(lidar_bin, remove_plane)

    # get camera (u, v, z) coordinates
    velo_camera = velo2camera(velo_points, image, remove_outliers=True)
    
    return velo_camera

# 1) Define a class dictionary for YOLO classes of interest.
#    (Add more if you have additional classes.)
CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def draw_bboxes_on_lidar_image(
    lidar_image, 
    bboxes, 
    color=(0, 255, 0),    # (B, G, R) => red bounding boxes
    thickness=2
):
    
    for bbox in bboxes:
        # Unpack the bounding box fields 
        # (Adjust indices if your format is slightly different)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
      

        
        # Draw bounding box
        cv2.rectangle(
            lidar_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color,
            thickness=thickness
        )


    return lidar_image


## 3d box
def velo2camera(velo_points, image=None, remove_outliers=True):
    ''' maps velo points (LiDAR) to camera (u,v,z) space '''
    # convert to (left) camera coordinates
    # P_left @ R_left_rect @ T_cam_velo
    velo_camera =  T_mat @ velo_points

    # delete negative camera points ??
    velo_camera  = np.delete(velo_camera , np.where(velo_camera [2,:] < 0)[0], axis=1) 

    # get camera coordinates u,v,z
    velo_camera[:2] /= velo_camera[2, :]

    # remove outliers (points outside of the image frame)
    if remove_outliers:
        u, v, z = velo_camera
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_camera = np.delete(velo_camera, np.where(outlier), axis=1)

    return velo_camera

def bin2h_velo(lidar_bin, remove_plane=True):
    ''' Reads LiDAR bin file and returns homogeneous (x,y,z,1) LiDAR points'''
    # read in LiDAR data
    scan_data = np.fromfile(lidar_bin, dtype=np.float32).reshape((-1,4))

    # get x,y,z LiDAR points (x, y, z) --> (front, left, up)
    velo_points = scan_data[:, 0:3] 

    # delete negative liDAR points
    velo_points = np.delete(velo_points, np.where(velo_points[3, :] < 0), axis=1)

    # use ransac to remove ground plane
    if remove_plane:
            ransac = linear_model.RANSACRegressor(
                                          linear_model.LinearRegression(),
                                          residual_threshold=0.1,
                                          max_trials=5000
                                          )

            X = velo_points[:, :2]
            y = velo_points[:, -1]
            ransac.fit(X, y)


            # remove outlier points
            mask = ransac.inlier_mask_
            velo_points = velo_points[~mask]
            pass
    # homogeneous LiDAR points
    velo_points = np.insert(velo_points, 3, 1, axis=1).T 

    return velo_points


def project_velo2cam(lidar_bin, image, remove_plane=True):
    ''' Projects LiDAR point cloud onto the image coordinate frame '''

    # get homogeneous LiDAR points from binn file
    velo_points = bin2h_velo(lidar_bin, remove_plane)

    # get camera (u, v, z) coordinates
    velo_camera = velo2camera(velo_points, image, remove_outliers=True)
    
    return velo_camera


def get_distances(image, velo_camera, bboxes, draw=False, draw_boxes=False):
    """
    Obtains distance measurements for each detected object in the image.
    """

    # Unpack LiDAR camera coordinates
    u, v, z = velo_camera

    # Prepare output bounding boxes with additional LiDAR data (u, v, z)
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # Loop through each bounding box
    for i, bbox in enumerate(bboxes):
        # Get bounding box points as integers
        pt1 = np.round(bbox[0:2]).astype(int)  # Top-left corner
        pt2 = np.round(bbox[2:4]).astype(int)  # Bottom-right corner

        # Compute the center of the bounding box
        x_center = (pt1[0] + pt2[0]) // 2
        y_center = (pt1[1] + pt2[1]) // 2

        # Find the closest LiDAR points to the center
        center_delta = np.abs(np.array([v, u]) - np.array([[y_center, x_center]]).T)

        # Choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # Get LiDAR location in camera space
        velo_depth = z[min_loc]  # LiDAR depth in camera space
        velo_location = np.array([v[min_loc], u[min_loc], velo_depth])

        # Add LiDAR coordinates (u, v, z) to bounding boxes
        bboxes_out[i, -3:] = velo_location

        # Draw bounding boxes and depth on image if draw_boxes or draw is True
        if draw_boxes:
            cv2.rectangle(
                image, 
                (pt1[0], pt1[1]), 
                (pt2[0], pt2[1]), 
                (0, 255, 0), 
                2
            )
        
        if draw:
            object_center = (x_center, y_center)
            cv2.putText(
                image,
                '{0:.2f} m'.format(velo_depth),
                object_center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

    return image, bboxes_out


def image_clusters_from_velo(velo_points, labels, image):
    ''' Obtains clusters in image space from velo (LiDAR) points 
        '''
    cam_clusters = {}
    for label in np.unique(labels):

        # convert from velo to camera 
        velo_cam = velo2camera(velo_points[:, labels == label], image)
        
        # append cluster label and cluster to clusters
        if velo_cam.shape[1] > 0:
            cam_clusters.update({label : velo_cam})

    return cam_clusters

def draw_clusters_on_image(image, cluster_dict, draw_centroids=False):
    ''' draws the clusters on an image '''
    pastel = cm.get_cmap('Pastel2', lut=50)
    get_pastel = lambda z : [255*val for val in pastel(int(z.round()))[:3]]

    for label, cluster in cluster_dict.items():
        for (u, v, z) in cluster.T:
            cv2.circle(image, (int(u), int(v)), 1, 
                       get_pastel(label), -1)

        if draw_centroids:
            centroid = np.mean(cluster, axis=1)
            cv2.circle(image, (int(centroid[0]), int(centroid[1])), 5, 
                       get_pastel(label), -1)

    return image

def get_depth_detections(left_image, lidar_bin, draw_boxes=False, draw_depth=False):
    # Compute detections
    detections = model(left_image)  # YOLOv8 inference
    boxes = detections[0].boxes

    # Convert bounding boxes, confidence scores, and class IDs to NumPy arrays
    xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    conf = boxes.conf.cpu().numpy()  # Confidence scores
    cls = boxes.cls.cpu().numpy()  # Class IDs
    bboxes = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])

    ###########Draw boxes on the image
    if draw_boxes:
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.rectangle(left_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(left_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Project LiDAR points to camera space
    velo_camera = project_velo2cam(lidar_bin, left_image)

    # Get distances
    left_image, bboxes_out = get_distances(left_image, velo_camera, bboxes, draw=draw_depth)

    return left_image, bboxes_out, velo_camera


def get_likely_clusters(cluster_dict, object_centers, min_thresh=50, thresh=75):
    ''' Obtains more likely clusters '''

    new_clusters = {}
    noise_cluster = cluster_dict[-1]

    labels = sorted(cluster_dict)

    for label in labels:
        cluster = cluster_dict[label]

        if cluster.shape[1] > min_thresh \
           and cluster.shape[1] < noise_cluster.shape[1] //2: 

            # ensure centroid has correct order
            centroid = np.mean(cluster, axis=1)[[1, 0, 2]]

            # check if any objects are detected
            if len(object_centers) > 0:
                delta = np.linalg.norm(centroid - object_centers, axis=1)
                min_loc = np.argmin(delta)

                if delta[min_loc] < thresh:
                    new_clusters.update({label : cluster})

            else:
                new_clusters.update({label : cluster})

    return new_clusters

def get_3d_bboxes(cluster_dict, labels, velo_points):
    camera_box_point_list = []

    for c_label, cluster in cluster_dict.items():

        velo_cluster = velo_points[:3, labels == c_label]

        (x_min, y_min, z_min) = velo_cluster.min(axis=1)
        (x_max, y_max, z_max) = velo_cluster.max(axis=1)

        # now get points to define 3d bounding box
        # box_points = np.array([[x_max, y_max, z_max]]) * arr
        # just manually do this for now
        box_points = np.array([
            [x_max, y_max, z_max, 1],
            [x_max, y_max, z_min, 1],
            [x_max, y_min, z_max, 1],
            [x_max, y_min, z_min, 1],
            [x_min, y_max, z_max, 1],
            [x_min, y_max, z_min, 1],
            [x_min, y_min, z_max, 1],
            [x_min, y_min, z_min, 1]
        ])

        # convert these box points to image space
        camera_box_points = T_mat @ box_points.T
        camera_box_points[:2] /= camera_box_points[2, :]

        camera_box_points = camera_box_points.round().T.astype(int)

        # append to list
        camera_box_point_list.append(camera_box_points)

    return camera_box_point_list

pastel = cm.get_cmap('Pastel2', lut=50)
get_pastel = lambda z : [255*val for val in pastel(z)[:3]]

def draw_3d_boxes(image, camera_box_points):
    for i, box_pts in enumerate(camera_box_points):
        [A, B, C, D, E, F, G, H] = box_pts[:, :2]
        color = get_pastel(i)

        # draw rear box (X-axis is forward for LiDAR)
        cv2.line(image, A, B, color, 2)
        cv2.line(image, B, D, color, 2)
        cv2.line(image, A, C, color, 2)
        cv2.line(image, D, C, color, 2)
        
        # draw front box
        cv2.line(image, G, E, color, 2)
        cv2.line(image, H, F, color, 2)
        cv2.line(image, G, H, color, 2)
        cv2.line(image, E, F, color, 2)

        # draw sides
        cv2.line(image, E, A, color, 2)
        cv2.line(image, G, C, color, 2)
        cv2.line(image, F, B, color, 2)
        cv2.line(image, H, D, color, 2)

    return image


def get_clusters(velo_points):
    dbscan = DBSCAN(eps=0.55, min_samples=30)
    dbscan.fit(velo_points[:3, :].T)

    labels = dbscan.labels_

    return dbscan, labels
#####################################################
def main():
    index =  9

    left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    bin_path = bin_paths[index]
    oxts_frame = get_oxts(oxts_paths[index])

    left_image_for_3D = left_image.copy()
    # get detections and object centers in uvz
    bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path,model,T_velo_cam2)


 # 1) Draw LiDAR points on a blank image or a copy of left_image
    lidar_proj_image = np.zeros_like(left_image)  # black background
    lidar_proj_image = draw_velo_on_image(velo_uvz, lidar_proj_image)
    
    # 2) Draw bounding boxes from YOLO onto the LiDAR-projected image
    lidar_proj_image_with_bboxes = draw_bboxes_on_lidar_image(lidar_proj_image.copy(), bboxes)

    # -- Display or show the new image (depending on your environment) --


    # Option B: Using PIL to show the image in a popup:
    lidar_proj_image_pil = Image.fromarray(lidar_proj_image_with_bboxes)
    lidar_proj_image_pil.show(title="LiDAR with Detected Bounding Boxes")

    # get transformed coordinates of object centers
    uvz = bboxes[:, -3:]

    # transform to (u,v,z)
    # velo_xyz = transform_uvz(uvz, T_cam2_velo) # we can also get LiDAR coordiantes
    imu_xyz = transform_uvz(uvz, T_cam2_imu)

    # get Lat/Lon on each detected object
    lat0 = oxts_frame[0]
    lon0 = oxts_frame[1]
    alt0 = oxts_frame[2]
    heading0 = oxts_frame[5]

    lla = imu2geodetic(imu_xyz[:, 0], imu_xyz[:, 1], imu_xyz[:, 2], lat0, lon0, alt0, heading0)
    velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))

        # stack image with LiDAR point cloud
    stacked = np.vstack((left_image, velo_image))
    stacked_image = Image.fromarray(stacked)
    stacked_image.show(title= "rgb and point cloud")

    left_image_2 = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    velo_image_2 = draw_velo_on_image(velo_uvz, left_image_2)
    velo_image_2 = Image.fromarray(velo_image_2)
    velo_image_2.show()

    canvas_height = stacked.shape[0]
    canvas_width = 500


    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas = draw_scenario(canvas, imu_xyz,canvas_height=canvas_height, sf=12)
    canvas=Image.fromarray(canvas)
    canvas.show()

    frame = np.hstack((stacked, 
                  255*np.ones((canvas_height, 1, 3), dtype=np.uint8),
                  canvas))
    
    frame=Image.fromarray(frame)
    frame.show()

    drive_map = folium.Map(
        location=(lat0, lon0), # starting location
        zoom_start=18
    )

    # add Lat/Lon points to map
    folium.CircleMarker(location=(lat0, lon0),
                        radius=2,
                        weight=5,
                        color='red').add_to(drive_map)
    
        # place the position of each detection on the map
    for pos in lla:
        folium.CircleMarker(location=(pos[0], pos[1]),
                            radius=2,
                            weight=5,
                            color='green').add_to(drive_map)

    # add Lat/Lon points to map
    folium.CircleMarker(location=(lat0, lon0),
                        radius=2,
                        weight=5,
                        color='red').add_to(drive_map)
   
    # drive_map.save('drive.html')


    ### 3d box
    (u, v, z) = project_velo2cam(bin_path, left_image_for_3D)
    velo_points = bin2h_velo(bin_path, remove_plane=True)



    # get object detections
    left_image_for_3D, bboxes_out, velo_camera = get_depth_detections(left_image_for_3D, 
                                                           bin_path, 
                                                           draw_boxes=False, 
                                                           draw_depth=False)
    # perform clustering in LiDAR space
    dbscan, labels = get_clusters(velo_points)

     # get clusters in image space
    cam_clusters = image_clusters_from_velo(velo_points, labels, left_image_for_3D)

    # get object centers in camera (u, v, z) coordinates
    object_centers = bboxes_out[:, 6:]

    # remove small and large clusters also remove cluster far away from detected objects
    cam_clusters = get_likely_clusters(cam_clusters, object_centers, min_thresh=25, thresh=70)

    # get 3D bbox points in camera space from clusters
    camera_box_points = get_3d_bboxes(cam_clusters, labels, velo_points)

    # draw 3D bounding boxes on the image
    left_image_for_3D = draw_3d_boxes(left_image_for_3D, camera_box_points)

        # draw projected clusters on new image
    cluster_image = np.zeros_like(left_image_for_3D)
    cluster_image = draw_clusters_on_image(cluster_image, cam_clusters, draw_centroids=True)
    stacked = np.vstack((left_image_for_3D, cluster_image))

    stacked = Image.fromarray(stacked)
    stacked.show()


if __name__ == "__main__":
    main()
