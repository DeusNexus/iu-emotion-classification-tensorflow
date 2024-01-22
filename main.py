import os
# Setting logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or '3' to additionally suppress all warnings

from io import BytesIO
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load your pre-trained Keras model
# Due to file upload limitation the ckpt file is now loaded with json_file model architecture and weights.
# Load architecture from JSON file
with open("./models/best_model/best_model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

keras_best_model = model_from_json(loaded_model_json)

# Load weights
keras_best_model.load_weights("./models/best_model/best_weights.h5")

html_file_path = Path(__file__).parent / "html" / "index.html"
css_file_path = Path(__file__).parent / "html" / "index.css"
favicon_file_path = Path(__file__).parent / "html" / "favicon.ico"

app = FastAPI()

# Enable CORS for all origins (useful for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label encodings
labels = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

# Utility function to preprocess the image
def preprocess_image(file):
    # Read the content of the file into a BytesIO object
    file_content = BytesIO(file.read())

    # Load and preprocess the image using the same approach as in your testing code
    img = image.load_img(file_content, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    return img_array

@app.get("/")
def read_root():
    return FileResponse(html_file_path, media_type="text/html")

@app.get("/index.css")
def get_css():
    return FileResponse(css_file_path, media_type="stylesheet")

@app.get("/favicon.ico")
def get_favicon():
    return FileResponse(favicon_file_path, media_type="image/x-icon")

# Corrected route without @app.head('/')
@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Preprocess the image
        img_array = preprocess_image(file.file)
        
        # Make predictions
        predictions = keras_best_model.predict(img_array)
        print(predictions)

        # Get the predicted emotion (assuming your model has a softmax output layer)
        predicted_emotion = int(np.argmax(predictions))

        return JSONResponse(content = {
            "predicted_emotion": predicted_emotion,
            "class_probabilities": str(predictions),
            "label_encodings":labels
            }, status_code=200)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")