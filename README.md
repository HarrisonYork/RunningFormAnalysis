# Running Form Analysis

This project takes a video of runners and analyzes their form, identifying four different errors and providing feedback to correct them.

# What It Does

This semester, I decided to challenge myself and run the Tarheel 10, a ten-mile race in Chapel Hill. I've never trained for long-distance running, and one problem I encountered was learning proper running form to avoid injury and get faster. I found it difficult to analyze my own running form from a first-person view, so I designed this project to accept side-view running videos and analyze the runner's form. There are two main steps in this process. First, an open-source YOLO object detection model is used to estimate the runner's pose in each frame of an uploaded video. These pose estimates are then fed into a custom form analysis model, which is a 1-dimensional convolutional neural network, that outputs the probability of the runner making four major form errors: heel striking, leaning too far forward, bending arms too tightly, or letting arms swing too loosely. This process is abstracted with a basic frontend and backend api allowing users to upload videos and receive feedback on their form quickly.

# Quick Start

`git clone https://github.com/HarrisonYork/RunningFormAnalysis.git`
`cd RunningFormAnalysis`

`python3 -m venv .venv`
`source .venv/bin/activate`

`pip install -r requirements.txt`

## Backend:
First create a `.env` file. 
This file specifies which model version to use, for example:

`YOLO_MODEL_PATH = "../models/pose_estimation/3_model_n_epoch5_freeze10.pt"`
`CNN_MODEL_PATH = "../models/form_analysis/form_analyzer_model_7kernel.pt"`

more model options are in the models/ directory

`cd src`
`flask --app api.py run`

## Frontend
In a new terminal window, navigate to RunningFormAnalysis/ and run the following:
`cd frontend`
`npm run dev`

If there is a command not found response, run this: 
`npm install`
`npm run dev`

The frontend should now be running at http://localhost:5173 (or the port specified by Vite).

# Video Links



# Evaluation

## Pose Estimation

The pose estimation model in the final implementation is a fine-tuned version of the YOLO11 small variant. Here is the comparison of the fine-tuned nano vs fine-tuned small models on the AthletePose validation set (with training time). Due to large training time for the small model, as well as memory limitations on my machine I did not consider larger YOLO variants.

| Metric | Fine-tuned nano (freeze backbone) | Fine-tuned small (freeze backbone) |
| ---- | -------- | -------- |
| Bounding Box Accuracy   | 0.9836 | 0.9874 |
| Pose Keypoint Accuracy  | 0.7534 | 0.7743 |
| Parameters | 2,866,468 | 9,902,940 |
| Training time (mins) | 48 | 401 |
| GPU Memory (GB) | 4.35 | 5.51 |

## Form Analysis

After evaluating model performance with different kernel size and patience, I selected the form analysis model with kernel size 7 and a patience of 5 for the webapp implementation.

| Metric | Kernel Size 3, Patience 3   | Kernel Size 5, Patience 3 | Kernel Size 7, Patience 3 | Kernel Size 3, Patience 5 | Kernel Size 5, Patience 5 | Kernel Size 7, Patience 5 |
| ---- | -------- | -------- | ------ | -------- | -------- | ------ |
| Test Loss   | 0.2595 | 0.1180 | 0.5007 | 0.0718  | 0.2007 | 0.0632 |
| Test Accuracy  | 72.27% | 85.71% | 47.06% | 86.55%  | 81.51% | 99.16% |


<br>
Evaluation details for the pose estimation model are in src/pose_estimator_training.py

Evaluation details for the form analysis model are in src/form_analyzer_training.py
