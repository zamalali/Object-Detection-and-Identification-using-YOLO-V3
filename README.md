# Object Detection and Identification using YOLO V3 🔎

This project utilizes the YOLO V3 algorithm, a powerful convolutional neural network for real-time object detection. YOLO V3 is capable of identifying multiple objects in images and videos with high accuracy and speed.

## Introduction

YOLO (You Only Look Once) V3 is an improvement over previous versions, offering better detection accuracy with relatively high speed, making it ideal for real-time applications. This project is designed to be easy to use, requiring minimal setup to start detecting objects in images and video streams.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following:
- Python 3.x installed
- Necessary Python libraries installed (e.g., OpenCV, NumPy). You can install these using `pip install -r requirements.txt` (make sure to include a `requirements.txt` file with all the necessary libraries).

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/zamalali/Object-Detection-and-Identification-using-YOLO-V3.git
   cd Object-Detection-and-Identification-using-YOLO-V3
   python run Object.py
## Download Required Files

To run the object detection model, you need to download the following files and place them in the appropriate directory within your project:

### COCO Names File

- **Description**: The `coco.names` file contains the list of object names that YOLO V3 can detect.
- **Download**: Ensure this file is placed in your project directory. If it's not included in the repository, you can find it as part of the YOLOv3 GitHub repository or through a quick web search for "coco.names for YOLOv3".

### YOLO V3 Configuration File

- **Description**: The YOLO V3 config file (`yolov3.cfg`) contains the network architecture details used by YOLO V3.
- **Download**: This file can usually be found in the official YOLO website or GitHub repositories dedicated to YOLO. Ensure you download the correct configuration file corresponding to YOLO V3.

### YOLO V3 Weights

- **Description**: The weights file contains the pre-trained model weights for YOLO V3. This is crucial for detecting objects accurately.
- **Download**: [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
- **Instructions**: Click the link and the download should start automatically. Once downloaded, place the weights file in your project directory.

### Directory Structure
You can also use Yolo-tiny if there are hardware limitations.

Ensure the downloaded files are placed in the correct directory within your project. A suggested structure is as follows:


