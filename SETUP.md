# Setup

git clone https://github.com/HarrisonYork/RunningFormAnalysis.git

cd RunningFormAnalysis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

## Frontend
`cd frontend`
`npm run dev`
The frontend should now be running at http://localhost:5173 (or the port specified by Vite).

## Backend:
`cd src`
`flask --app api.py run`
The backend should now be running at http://127.0.0.1:5000


## Running the Application Locally
Open your web browser and navigate to the frontend URL (e.g., http://localhost:5173).

The React app is pre-configured to send API requests to http://127.0.0.1:5000 (the local Flask server).

Upload a side-profile MP4 or MOV video of a runner.

Wait a few seconds for the form analysis results, and check your terminals to read logs. 