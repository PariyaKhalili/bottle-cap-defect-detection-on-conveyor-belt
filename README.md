# Bottle Cap Defect Detection & Counting on Conveyor Belt  
**YOLOv8-powered Industrial Quality Control System**

A complete end-to-end computer vision solution that detects bottles on a conveyor belt, tracks them using ByteTrack, classifies cap condition (good, loose, no-cap, defect, ring-missing), estimates bottle speed, and counts properly sealed bottles — all in real time.

![Example](https://github.com/PariyaKhalili/bottle-cap-defect-detection-on-conveyor-belt/blob/main/samples/closed_cap_detection.jpg)


Ideal for beverage manufacturing quality inspection lines.

## Features
- **Two-stage YOLOv8 pipeline**:  
  1. Bottle detection & tracking (`yolov8s.pt` + ByteTrack)  
  2. Fine-tuned cap defect classifier (5 classes)
- Real-time cap inspection with color-coded labels
- Automatic counting of total bottles and correctly closed caps
- Conveyor speed estimation (cm/s)
- Clean annotated output video with overlays
- Fully reproducible in Google Colab (GPU accelerated)

## Technologies Used
- Ultralytics YOLOv8 (`8.2.103`)
- OpenCV + NumPy
- ByteTrack (via Ultralytics tracker)
- Google Colab + Google Drive integration
- Supervision (for visualization in training)

## Project Structure

    ├── samples/                                  
    │   ├── closed_cap_detection.jpg
    │   ├── full_defect_detection_sample.jpg
    ├── src/
    │   ├── 01_finetune_cap_model.ipynb          # Fine-tune cap classifier on custom dataset
    │   └── 02_bottle_detection_and_counting.ipynb # Full inference: tracking + cap classification + counting
    │
    │
    ├── requirements-colab.txt                    # Colab-ready dependencies
    └── README.md

  
## Prerequisites
- Google Colab (recommended) with GPU runtime  
  or local machine with NVIDIA GPU + CUDA
- Google Drive account (for storing dataset/video/models)

## Installation (Colab - Recommended)
1. Place the video your Google Drive
2. Open Colab  
3. Uplode the files
4. Run the cells

##  How It Works
**Stage 1**: Fine-Tuning (01_finetune_cap_model.ipynb)

- Uses Roboflow-exported YOLOv8 dataset
- Trains a 5-class cap classifier:
- good, defect, loos-cap, no-cap, ring-missing
- Exports best.pt

**Stage 2**: Inference & Counting (02_bottle_detection_and_counting.ipynb)

- Detect & track bottles using yolov8s.pt + ByteTrack
- Crop top 20% of each bottle → upscale 3×
- Run fine-tuned cap model
- Count when bottle crosses center line
- Overlay total count + closed caps
- Save annotated video

**This program could be extended to all types of bottle defects, such as checking whether a bottle is full or properly labeled.**

![Example](https://github.com/PariyaKhalili/bottle-cap-defect-detection-on-conveyor-belt/blob/main/samples/full_defect_detection_sample.jpg)

