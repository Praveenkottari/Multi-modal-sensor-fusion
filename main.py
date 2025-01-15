import os
import cv2
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import pymap3d as pm
from PIL import Image
import folium
from glob import glob
from sklearn.cluster import DBSCAN
from sklearn import linear_model
from ultralytics import YOLO
import torch
import matplotlib.cm as cm

matplotlib.use('Agg')  # no GUI

# local import from your package (assuming it's accessible)
from pkg.kitti_utils import *


################################################################################
#                              HELPER FUNCTIONS                                #
################################################################################

def velo2camera(velo_points, image=None, remove_outliers=True):
    """Maps LiDAR (velo) points to camera (u,v,z) space."""
    velo_camera = T_mat @ velo_points  # Convert to (left) camera coordinates
    # Delete negative camera points
    velo_camera = np.delete(velo_camera, np.where(velo_camera[2, :] < 0)[0], axis=1)
    # Normalize camera coords
    velo_camera[:2] /= velo_camera[2, :]

    # Remove outliers (points outside of the image frame)
    if remove_outliers and image is not None:
        u, v, z = velo_camera
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_camera = np.delete(velo_camera, np.where(outlier), axis=1)
    return velo_camera

def bin2h_velo(lidar_bin, remove_plane=True):
    """Reads LiDAR bin file and returns homogeneous (x,y,z,1) LiDAR points."""
    scan_data = np.fromfile(lidar_bin, dtype=np.float32).reshape((-1, 4))
    velo_points = scan_data[:, 0:3]  # (x, y, z)

    # Use RANSAC to remove ground plane
    if remove_plane:
        ransac = linear_model.RANSACRegressor(
            linear_model.LinearRegression(),
            residual_threshold=0.1,
            max_trials=5000
        )
        X = velo_points[:, :2]
        y = velo_points[:, -1]
        ransac.fit(X, y)
        mask = ransac.inlier_mask_
        velo_points = velo_points[~mask]

    # Homogeneous LiDAR points
    velo_points = np.insert(velo_points, 3, 1, axis=1).T
    return velo_points

def image_clusters_from_velo(velo_points, labels, image):
    """Obtains clusters in image space from velo (LiDAR) points."""
    cam_clusters = {}
    for label in np.unique(labels):
        velo_cam = velo2camera(velo_points[:, labels == label], image)
        if velo_cam.shape[1] > 0:
            cam_clusters.update({label: velo_cam})
    return cam_clusters

def project_velo2cam(lidar_bin, image, remove_plane=True):
    """Projects LiDAR point cloud onto the image coordinate frame."""
    velo_points = bin2h_velo(lidar_bin, remove_plane)
    velo_camera = velo2camera(velo_points, image, remove_outliers=True)
    return velo_camera

def get_distances(image, velo_camera, bboxes, draw=False):
    """Obtains distance measurements for each detected object in the image."""
    u, v, z = velo_camera
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    for i, bbox in enumerate(bboxes):
        pt1 = torch.round(torch.tensor(bbox[0:2])).to(torch.int).numpy()
        pt2 = torch.round(torch.tensor(bbox[2:4])).to(torch.int).numpy()
        x_center = (pt1[1] + pt2[1]) / 2
        y_center = (pt1[0] + pt2[0]) / 2
        center_delta = np.abs(np.array((v, u)) - np.array([[x_center, y_center]]).T)
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # Depth in camera space
        velo_depth = z[min_loc]
        velo_location = np.array([v[min_loc], u[min_loc], velo_depth])
        bboxes_out[i, -3:] = velo_location

        if draw:
            object_center = (int(round(y_center)), int(round(x_center)))
            cv2.putText(image,
                        '{0:.2f} m'.format(velo_depth),
                        object_center,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0), 2, cv2.LINE_AA)
    return image, bboxes_out

def get_depth_detections(left_image, lidar_bin, draw_boxes=True, draw_depth=True):
    """Obtains detection depth estimates for all objects in the input image."""
    detections = model(left_image)

    # Draw boxes on image (using YOLOv8's plot if desired)
    if draw_boxes:
        for r in detections:
            plt.imshow(r.plot())  # show the first result
            plt.axis('off')
            plt.show()

    # bboxes: Nx6 -> [x1, y1, x2, y2, confidence, class]
    bboxes = detections[0].boxes.data.cpu().numpy()
    velo_camera = project_velo2cam(lidar_bin, left_image)
    left_image, bboxes_out = get_distances(left_image, velo_camera, bboxes, draw=draw_depth)
    return left_image, bboxes_out, velo_camera

def draw_velo_on_image(velo_camera, velo_image, color_map=lambda x: (255, 255, 255)):
    """Draws LiDAR point cloud on an image."""
    u, v, z = velo_camera
    for i in range(len(u)):
        cv2.circle(velo_image, (int(u[i]), int(v[i])), 1, color_map(z[i]), -1)
    return velo_image

def draw_clusters_on_image(image, cluster_dict, draw_centroids=False):
    """Draws the clusters on an image."""
    pastel = cm.get_cmap('Pastel2', lut=50)
    get_pastel = lambda z: [255 * val for val in pastel(int(z.round()))[:3]]

    for label, cluster in cluster_dict.items():
        for (u, v, _) in cluster.T:
            cv2.circle(image, (int(u), int(v)), 1, get_pastel(label), -1)

        if draw_centroids:
            centroid = np.mean(cluster, axis=1)
            cv2.circle(image, (int(centroid[0]), int(centroid[1])), 5, get_pastel(label), -1)
    return image

def get_clusters(velo_points):
    """DBSCAN clustering of velo (LiDAR) points."""
    dbscan = DBSCAN(eps=0.5, min_samples=30)
    dbscan.fit(velo_points[:3, :].T)
    labels = dbscan.labels_
    return dbscan, labels

def get_likely_clusters(cluster_dict, object_centers, min_thresh=50, thresh=75):
    """Filters clusters based on size and distance to object centers."""
    new_clusters = {}
    noise_cluster = cluster_dict[-1] if -1 in cluster_dict else None
    labels = sorted(cluster_dict)

    for label in labels:
        if label == -1:
            continue
        cluster = cluster_dict[label]

        if noise_cluster is not None:
            # use noise cluster size for reference
            if (cluster.shape[1] > min_thresh) and (cluster.shape[1] < noise_cluster.shape[1] // 2):
                centroid = np.mean(cluster, axis=1)[[1, 0, 2]]
                if len(object_centers) > 0:
                    delta = np.linalg.norm(centroid - object_centers, axis=1)
                    min_loc = np.argmin(delta)
                    if delta[min_loc] < thresh:
                        new_clusters.update({label: cluster})
                else:
                    new_clusters.update({label: cluster})
        else:
            # if there's no noise cluster to reference
            if cluster.shape[1] > min_thresh:
                centroid = np.mean(cluster, axis=1)[[1, 0, 2]]
                if len(object_centers) > 0:
                    delta = np.linalg.norm(centroid - object_centers, axis=1)
                    min_loc = np.argmin(delta)
                    if delta[min_loc] < thresh:
                        new_clusters.update({label: cluster})
                else:
                    new_clusters.update({label: cluster})
    return new_clusters

def get_3d_bboxes(cluster_dict, labels, velo_points):
    """Constructs 3D bounding boxes in camera space from cluster dict."""
    camera_box_point_list = []
    for c_label, cluster in cluster_dict.items():
        velo_cluster = velo_points[:3, labels == c_label]
        (x_min, y_min, z_min) = velo_cluster.min(axis=1)
        (x_max, y_max, z_max) = velo_cluster.max(axis=1)

        # Points defining a 3D bounding box
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

        camera_box_points = T_mat @ box_points.T
        camera_box_points[:2] /= camera_box_points[2, :]
        camera_box_points = camera_box_points.round().T.astype(int)
        camera_box_point_list.append(camera_box_points)
    return camera_box_point_list

def draw_3d_boxes(image, camera_box_points):
    """Draws 3D bounding boxes on the image."""
    pastel = cm.get_cmap('Pastel2', lut=50)
    get_pastel = lambda z: [255 * val for val in pastel(z)[:3]]

    for i, box_pts in enumerate(camera_box_points):
        [A, B, C, D, E, F, G, H] = box_pts[:, :2]
        color = get_pastel(i)

        # Rear box
        cv2.line(image, A, B, color, 2)
        cv2.line(image, B, D, color, 2)
        cv2.line(image, A, C, color, 2)
        cv2.line(image, D, C, color, 2)

        # Front box
        cv2.line(image, G, E, color, 2)
        cv2.line(image, H, F, color, 2)
        cv2.line(image, G, H, color, 2)
        cv2.line(image, E, F, color, 2)

        # Sides
        cv2.line(image, E, A, color, 2)
        cv2.line(image, G, C, color, 2)
        cv2.line(image, F, B, color, 2)
        cv2.line(image, H, D, color, 2)
    return image

def main_pipeline(left_image, lidar_bin, velo_points):
    """Main pipeline for depth detections and DBSCAN clustering."""
    left_image, bboxes_out, velo_camera = get_depth_detections(
        left_image, 
        lidar_bin, 
        draw_boxes=False,
        draw_depth=False
    )
    dbscan, labels = get_clusters(velo_points)
    cam_clusters = image_clusters_from_velo(velo_points, labels, left_image)
    object_centers = bboxes_out[:, 6:]  # (u, v, z) coordinates
    cam_clusters = get_likely_clusters(cam_clusters, object_centers, min_thresh=50, thresh=75)
    camera_box_points = get_3d_bboxes(cam_clusters, labels, velo_points)
    left_image = draw_3d_boxes(left_image, camera_box_points)
    return left_image, cam_clusters

def imu2geodetic(x, y, z, lat0, lon0, alt0, heading0):
    """
    Converts cartesian IMU coordinates to Geodetic based on current location.
    - Correct orientation is provided by heading
    - Elevation must be corrected for pymap3d
    """
    rng = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
    el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90
    lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)
    lla = np.vstack((lla[0], lla[1], lla[2])).T
    return lla

def draw_scenario(canvas, imu_xyz, sf, canvas_height, canvas_width):
    """Draws ego vehicle and detected object rectangles on a canvas."""
    ego_center = (250, int(canvas_height * 0.95))
    ego_x1 = ego_center[0] - 5
    ego_y1 = ego_center[1] - 10
    ego_x2 = ego_center[0] + 5
    ego_y2 = ego_center[1] + 10
    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (0, 255, 0), -1)

    for val in imu_xyz:
        obj_center = (ego_center[0] - sf*int(np.round(val[1])),
                      ego_center[1] - sf*int(np.round(val[0])))
        obj_x1 = obj_center[0] - 5
        obj_y1 = obj_center[1] - 10
        obj_x2 = obj_center[0] + 5
        obj_y2 = obj_center[1] + 10
        cv2.rectangle(canvas, (obj_x1, obj_y1), (obj_x2, obj_y2), (255, 0, 0), -1)
    return canvas

def get_detection_coordinates(image, bin_path, draw_boxes=True, draw_depth=True):
    """
    Obtains detections for the input image, along with the coordinates of 
    the detected object centers in:
    - Camera with depth (uvz)
    - LiDAR/velo (xyz)
    - GPS/IMU  (xyz)
    """
    detections = model(image)
    # Draw boxes on image
    if draw_boxes:
        for r in detections:
            plt.imshow(r.plot())
            plt.axis('off')
            plt.show()
            
    ###########Nx6 -> [x1, y1, x2, y2, confidence, class]
    bboxes = detections[0].boxes.data.cpu().numpy()
    velo_uvz = project_velobin2uvz(bin_path, T_velo_cam2, image, remove_plane=True)
    bboxes = get_uvz_centers(image, velo_uvz, bboxes, draw=draw_depth)
    return bboxes, velo_uvz

# 1. Create a class map (for the COCO indices we are using)
class_map = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    """
    Obtains detected object centers projected to uvz camera coordinates.
    Associates LiDAR uvz coordinates to detected object centers.
    """
    (u, v, z) = velo_uvz
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    for i, bbox in enumerate(bboxes):
        pt1 = np.round(bbox[0:2]).astype(int)
        pt2 = np.round(bbox[2:4]).astype(int)

        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        center_delta = np.abs(np.array((v, u)) - np.array([[obj_x_center, obj_y_center]]).T)
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))
        velo_depth = z[min_loc]
        uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])
        bboxes_out[i, -3:] = uvz_location

        if draw:
            object_center = (int(round(obj_y_center)), int(round(obj_x_center)))

            # 2. Extract class index from bbox and get readable name
            class_id = int(bbox[5])  # assuming bbox = [x1, y1, x2, y2, conf, cls]
            class_name = class_map.get(class_id, "unknown")

            # 3. Combine class name + distance in one text string
            label_text = f"{class_name} {velo_depth:.2f} m"

            # 4. Draw on the image
            cv2.putText(image,
                        label_text,
                        object_center,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,        # font scale
                        (0, 255, 0),# blue color
                        1,          # thickness
                        cv2.LINE_AA)

    return bboxes_out


################################################################################
#                           YOLO MODEL INITIALIZATION                           #
################################################################################

# Replace YOLOv5 with YOLOv8 and filter classes = [person(0), bicycle(1), car(2),
# motorcycle(3), bus(5), truck(7)] (COCO indices)
model = YOLO("yolov8s.pt")
model.conf = 0.45  # confidence threshold
model.iou = 0.5 # NMS IoU threshold
model.classes = [0, 1, 2, 3, 5, 7]  # only person, bicycle, car, motorcycle, bus, truck


################################################################################
#                               GLOBALS / CALIBS                               #
################################################################################

DATA_PATH = r'./../visionNav/fusion/dataset/2011_10_03_drive_0047_sync'
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
oxts_paths = sorted(glob(os.path.join(DATA_PATH, r'oxts/data**/*.txt')))

print(f"Number of left images: {len(left_image_paths)}")
print(f"Number of right images: {len(right_image_paths)}")
print(f"Number of LiDAR point clouds: {len(bin_paths)}")
print(f"Number of GPS/IMU frames: {len(oxts_paths)}")

with open('./../visionNav/fusion/dataset/2011_10_03_calib/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# get projection matrices (rectified left camera --> left camera (u,v,z))
P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))

# get rectified rotation matrices (left camera --> rectified left camera)
R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)

# get rigid transformation from Camera 0 (ref) to Camera 2
R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))
T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)

T_velo_ref0 = get_rigid_transformation(r'./../visionNav/fusion/dataset/2011_10_03_calib/calib_velo_to_cam.txt')
T_imu_velo = get_rigid_transformation(r'./../visionNav/fusion/dataset/2011_10_03_calib/calib_imu_to_velo.txt')

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 
# homogeneous transform from left color camera to velo (LiDAR) (4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 
T_imu_cam2 = T_velo_cam2 @ T_imu_velo
T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0,0,0,1], axis=0)) 

# get left and right camera projection matrices
P_left = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))
P_right = np.array([float(x) for x in calib[33].strip().split(' ')[1:]]).reshape((3,4))

# get rectified rotation matrices for left and right
R_left_rect = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))
R_right_rect = np.array([float(x) for x in calib[32].strip().split(' ')[1:]]).reshape((3, 3,))
R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0], axis=0)
R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0,1], axis=1)

# Decompose to get K, R, T for left/right
def decompose_projection_matrix(P):
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T / T[3]
    return K, R, T

K_left, R_left, T_left = decompose_projection_matrix(P_left)
K_right, R_right, T_right = decompose_projection_matrix(P_right)

with open(r'./../visionNav/fusion/dataset/2011_10_03_calib/calib_velo_to_cam.txt', 'r') as f:
    calib = f.readlines()

R_cam_velo = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
t_cam_velo = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
T_cam_velo = np.vstack((np.hstack((R_cam_velo, t_cam_velo)), np.array([0, 0, 0, 1])))
T_mat = P_left @ R_left_rect @ T_cam_velo


################################################################################
#                                MAIN FUNCTION                                 #
################################################################################

def main():
    index = 500
    left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    bin_path = bin_paths[index]
    oxts_frame = get_oxts(oxts_paths[index])

    lidar_bin = bin_paths[index]
    (u, v, z) = project_velo2cam(lidar_bin, left_image)
    velo_points = bin2h_velo(lidar_bin, remove_plane=True)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=30)
    dbscan.fit(velo_points[:3, :].T)
    labels = dbscan.labels_
    cam_clusters = image_clusters_from_velo(velo_points, labels, left_image)

    # Draw clusters on a blank image
    new_image = np.zeros_like(left_image)
    new_image = draw_clusters_on_image(new_image, cam_clusters, draw_centroids=True)

    # Possibly RANSAC the walls too
    velo_points = bin2h_velo(lidar_bin, remove_plane=True)

    # Run the pipeline
    left_image, cam_clusters = main_pipeline(left_image, lidar_bin, velo_points)

    # Draw projected clusters on new image
    # cluster_image = np.zeros_like(left_image)
    # cluster_image = draw_clusters_on_image(cluster_image, cam_clusters, draw_centroids=True)
    # stacked = np.vstack((left_image, cluster_image))
    # image_3dbox = Image.fromarray(stacked)
    # image_3dbox.show()

    ############################################################################
    # Get detections and object centers in uvz
    bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path)
    uvz = bboxes[:, -3:]
    imu_xyz = transform_uvz(uvz, T_cam2_imu)

    # Get Lat/Lon on each detected object
    lat0 = oxts_frame[0]
    lon0 = oxts_frame[1]
    alt0 = oxts_frame[2]
    heading0 = oxts_frame[5]
    lla = imu2geodetic(imu_xyz[:, 0], imu_xyz[:, 1], imu_xyz[:, 2], lat0, lon0, alt0, heading0)

    # Draw LiDAR on image
    velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))
    # plt.rcParams["figure.figsize"] = (20, 10)
    stacked = np.vstack((left_image, velo_image))
    # image_disp = Image.fromarray(stacked)
    # image_disp.show()

    # left_image_2 = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    # velo_image_2 = draw_velo_on_image(velo_uvz, left_image_2)
    # image_velo = Image.fromarray(velo_image_2)
    # image_velo.show()

    ############################################################################
    # Draw scenario on canvas
    canvas_height = stacked.shape[0]
    canvas_width = 500
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    draw_scenario(canvas, imu_xyz, sf=12, canvas_height=stacked.shape[0], canvas_width=500)
    frame = np.hstack((stacked, 255 * np.ones((canvas_height, 1, 3), dtype=np.uint8), canvas))
    image_frame = Image.fromarray(frame)
    image_frame.show()

    # (Optional) Folium mapping example commented out:
    # drive_map = folium.Map(location=(lat0, lon0), zoom_start=18)
    # folium.CircleMarker(location=(lat0, lon0), radius=2, weight=5, color='red').add_to(drive_map)
    # for pos in lla:
    #     folium.CircleMarker(location=(pos[0], pos[1]), radius=2, weight=5, color='green').add_to(drive_map)
    # folium.CircleMarker(location=(lat0, lon0), radius=2, weight=5, color='red').add_to(drive_map)
    # drive_map.save('drive.html')


if __name__ == "__main__":
    main()
