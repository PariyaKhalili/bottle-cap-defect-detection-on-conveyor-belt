# Bottle Cap Defect Detection on Conveyor Belt
This repository contains a computer vision project for detecting and classifying bottle cap defects on a conveyor belt using YOLOv8. The system identifies bottles in a video stream, tracks them, classifies the cap status (e.g., good, loose, missing, defective, or ring-missing), counts the total bottles and properly closed caps, and overlays results on the output video.
The project includes two Jupyter notebooks:

- One for fine-tuning a YOLO model on a custom bottle cap dataset.
- One for applying the fine-tuned model to a video, performing detection, tracking, classification, and counting.

This is useful for quality control in manufacturing lines, such as beverage production, to ensure caps are correctly sealed.

## Features

- **Fine-Tuning YOLOv8**: Train a custom model on a dataset of bottle caps with classes like 'good', 'loose-cap', 'no-cap', 'defect', and 'ring-missing'.
- **Object Detection and Tracking**: Detect bottles using pre-trained YOLOv8, track them across frames with ByteTrack.
- **Cap Classification**: Crop the cap region, upscale it, and classify defects using the fine-tuned model.
- **Counting and Speed Estimation**: Count total bottles and closed caps; estimate conveyor speed in cm/s.
- **Video Processing**: Process input videos and generate annotated output videos with overlays (e.g., bounding boxes, labels, counts).
- **Visualization**: Display sample detections and save results to Google Drive (Colab-compatible).

Demo

Watch a sample output video: output.mp4 (replace with your actual file path).
Example Output:

Total Bottles: X
Closed Caps: Y
Cap Classes: Color-coded (Green: Good, Yellow: Loose, Red: No Cap/Defect, etc.)

Requirements

Python 3.9+
Google Colab (for GPU acceleration and Drive integration)
Libraries:
ultralytics==8.2.103 (YOLOv8)
opencv-python
numpy
matplotlib
supervision (for annotations, install via pip install supervision)


Full dependencies are installed in the notebooks.
Installation

Clone the repository:textgit clone https://github.com/yourusername/bottle-cap-detection.git
cd bottle-cap-detection
Install dependencies:textpip install -r requirements.txt(Create a requirements.txt file with the above libraries if needed.)
Mount Google Drive in Colab (as shown in notebooks) to access datasets and models.
Download the dataset: Use the Roboflow-exported YOLOv8 bottle cap dataset (unzipped in the fine-tuning notebook).
Pre-trained Models:
Base YOLOv8: yolov8s.pt (auto-downloaded by Ultralytics).
Fine-tuned Cap Model: Place your best.pt in /path/to/models/bottle-cap-v8/weights/best.pt.


Usage
1. Fine-Tuning the Model
Use finetune_bottle_cap_model.ipynb to train the cap classification model.

Steps:
Mount Google Drive.
Unzip the dataset (e.g., from Roboflow).
Install Ultralytics and other libraries.
Load the dataset and visualize samples.
Run training (adjust epochs, batch size, etc. in the notebook).

Example Training Command (embedded in notebook):Pythonfrom ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Start with pre-trained model
model.train(data='path/to/data.yaml', epochs=50, imgsz=640)
Output: Trained weights saved as best.pt. Move to your models folder.

2. Running Detection on Video
Use bottle_cap_detection_and_counting.ipynb to process videos.

Steps:
Mount Google Drive.
Install libraries (cvzone, Ultralytics).
Load models: Base YOLO for bottle detection, fine-tuned for cap classification.
Set input video path (e.g., /path/to/bottle1.mp4).
Run the processing loop: Detect, track, classify caps, count, and estimate speed.
Output video saved to /path/to/output/output.mp4.

Key Configurations:
Confidence threshold: 0.6 for detection, 0.8 for classification.
Cap crop: Top 20% of bottle bounding box.
Line for counting: Vertical line at 50% of frame width.
Speed calculation: Based on pixel-to-cm conversion (adjust PIXELS_PER_CM as needed).

Run in Notebook:
Execute cells sequentially. The output video will include:
Bounding boxes around bottles.
ID, speed, and cap class labels.
Total bottles and closed caps overlay.


Example
Python# In bottle_cap_detection_and_counting.ipynb
model = YOLO("yolov8s.pt")  # Bottle detection
correct_cap_model = YOLO('/path/to/best.pt')  # Cap classification

results = model.track(source='input_video.mp4', stream=True, persist=True, tracker="bytetrack.yaml")
# Process results as in the notebook...
Dataset

Bottle Cap Dataset: Exported from Roboflow (YOLOv8 format).
Classes: defect (0), good (1), loose-cap (2), no-cap (3), ring-missing (4).
Video: Sample conveyor belt video (bottle1.mp4) from your Drive.

Results

Accuracy: Depends on fine-tuning (e.g., mAP@0.5 from training logs).
Performance: ~12ms inference per frame on GPU.
Sample Metrics: Total bottles counted, closed caps, defect types.

Limitations

Assumes vertical conveyor movement; adjust line position for other setups.
Speed estimation requires calibration (pixel-to-cm ratio).
Tested on Colab with T4 GPU; may need adjustments for local environments.
