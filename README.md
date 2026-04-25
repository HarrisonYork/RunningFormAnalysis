# What It Does

This semester, I decided to challenge myself and run the Tarheel 10, a ten-mile race in Chapel Hill. I've never trained for long-distance running, and one problem I encountered was learning proper running form to avoid injury and get faster. I found it difficult to analyze my own running form from a first-person view, so I designed this project to accept side-view running videos and analyze the runner's form. There are two main steps in this process. First, an open-source YOLO object detection model is used to estimate the runner's pose in each frame of an uploaded video. These pose estimates are then fed into a custom form analysis model, which is a 1-dimensional convolutional neural network, that outputs the probability of the runner making four major form errors: heel striking, leaning too far forward, bending arms too tightly, or letting arms swing too loosely. This process is abstracted with a basic frontend and backend api allowing users to upload videos and receive feedback on their form quickly.

# Quick Start

Install dependencies
[TODO]

Set up .env variables:


To run the frontend:
`cd frontend`
`npm run dev`

To run the backend api:
`flask --app api.py run`

# Video Links

# Evaluation
