# Enter docker folder
cd docker-api

# Build the image
docker build -t emotion_prediction_fastapi:latest .

# Run the container
docker run --name emotion_prediction_container -p 8000:8000 emotion_prediction_fastapi:latest

# Open the API Front-end
http://127.0.0.1:8000