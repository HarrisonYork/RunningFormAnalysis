# Setup

`git clone https://github.com/HarrisonYork/RunningFormAnalysis.git`

`cd RunningFormAnalysis`

`python3 -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

## Backend:
First, create a `.env` file. This specifies which model version to use, for example:

`YOLO_MODEL_PATH = "../models/pose_estimation/3_model_s_epoch5_freeze10.pt"`

`CNN_MODEL_PATH = "../models/form_analysis/form_analyzer_model_7kernel.pt"`

more model options are in the models/ directory

`cd src`

`flask --app api.py run`

The backend should now be running at http://127.0.0.1:5000


## Frontend

In a new terminal window, navigate to RunningFormAnalysis/ and run the following:

`cd frontend`

`npm run dev`

If there is a "command not found response", do this: 

`npm install`

`npm run dev`

The frontend should now be running at http://localhost:5173 (or the port specified by Vite).


## Running the Application Locally
Open your web browser and navigate to the frontend URL (e.g., http://localhost:5173).

The React app is pre-configured to send API requests to http://127.0.0.1:5000 (the local Flask server).

Upload a side-profile MP4 or MOV video of a runner.

Wait a few seconds for the form analysis results, and check your backend terminal to read logs. 