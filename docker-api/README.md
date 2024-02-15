# Building Image
### Enter docker folder
`cd docker-api`
### Build the image
`docker build -t emotion_prediction_fastapi:latest .`
### Run the container
`docker run --name emotion_prediction_container -p 8000:8000 emotion_prediction_fastapi:latest`
### Open the API Front-end
`http://127.0.0.1:8000`

# Running API directly
### Create a local pythong venv
`python3 -m venv venv`
### Activate the virtual environment
`source venv/bin/activate`
### Install the required modules
`pip3 install -r requirements.txt`
### Run the API 
`uvicorn main:app --reload`
### Open the API Front-end
`http://127.0.0.1:8000`