
# Multi Modal Sensor Fusion for Visual Perception

In modern Robotics and Autonomous driving, Robust perception is crucial for safe navigation, obstacle avoidance, and situational awareness. This study presents a Multi-Modal Fusion Pipeline that integrates Lidar, Camera, and IMU sensors for end-to end 3D object detection, mapping and global localization, particularly suited for real world driving  environments. The approach is validated on the KITTI dataset, but i can be generalized for any input source. In Point cloud mapping applied  ground plane removal to furthur reduce the computational overhead, while extracting meaningful 3D object clusters. The resulting 3D bounding boxes and depth measurements are then fused with IMU data for real time geospatial mapping, enabling reliable global localization.  This pipeline has broad applicability in self driving vehicles and mobile robotics, providing a unified framework for efficient environment perception. Experimental results confirm that the pipeline maintains reliable object localization while meeting real time constraints, demonstrating its effectiveness in delivering an end-to-end system that is computationally efficient.


## Demo

[Insert gif or link to demo](https://github.com/Praveenkottari/Multi-modal-sensor-fusion/blob/a264e2e5ec3818c5cec5ab0dd777a6e4cc2a9e48/output/out.gif)


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## Roadmap

- Additional browser support

- Add more integrations


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```



## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
