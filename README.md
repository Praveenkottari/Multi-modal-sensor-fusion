
# Multi Modal Sensor Fusion for Visual Perception

In modern Robotics and Autonomous driving, Robust perception is crucial for safe navigation, obstacle avoidance, and situational awareness. This study presents a Multi-Modal Fusion Pipeline that integrates Lidar, Camera, and IMU sensors for end-to end 3D object detection, mapping and global localization, particularly suited for real world driving  environments. The approach is validated on the KITTI dataset, but i can be generalized for any input source. In Point cloud mapping applied  ground plane removal to furthur reduce the computational overhead, while extracting meaningful 3D object clusters. The resulting 3D bounding boxes and depth measurements are then fused with IMU data for real time geospatial mapping, enabling reliable global localization.  This pipeline has broad applicability in self driving vehicles and mobile robotics, providing a unified framework for efficient environment perception. Experimental results confirm that the pipeline maintains reliable object localization while meeting real time constraints, demonstrating its effectiveness in delivering an end-to-end system that is computationally efficient.


## Detection + Depth + BEV mapping
<div align="center">
<img src="https://github.com/Praveenkottari/Multi-modal-sensor-fusion/blob/a264e2e5ec3818c5cec5ab0dd777a6e4cc2a9e48/output/out.gif" width="600" alt="animated hello">
</div>

    
## Project pipeline
<div align="center">
<img src="https://github.com/Praveenkottari/Multi-modal-sensor-fusion/blob/a6ea5feb5954416a6b29ae5996f2e0e4e2ee7627/pipline.png" width="600" alt="animated hello">
</div>

The projection pipeline aligns Lidar points with the camera view of detection and depth estimation. The systemproduces 2D and 3D bounding boxes on the image space with depth, while mapping the objects to a global coordinate frame


<div align="center">
<img src="https://github.com/Praveenkottari/Multi-modal-sensor-fusion/blob/eaa03c1bdbd092d6183c6729131db1af474c101c/output/steps.gif" width="600" alt="animated hello">
</div>

## Run Locally

Clone the project

```bash
   git clone https://github.com/Praveenkottari/Multi-modal-sensor-fusion.git
```

Go to the project directory

```bash
   cd Multi-modal-sensor-fusion
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run Python file

```bash
  python main.py
```

Note: In this repo small dataset is given the complete dataset can be downloded from [here](https://www.cvlibs.net/datasets/kitti/raw_data.php) 


## Acknowledgements

 - [PixelOverflow](https://youtube.com/@pixeloverflow?si=GoiB8ai2mv4GR1x_)
 - [Ultralytics/yolov8](https://docs.ultralytics.com/)
 - [KITTI dataset](https://www.cvlibs.net/datasets/kitti/)
