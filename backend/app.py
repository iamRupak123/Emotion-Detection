from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from tensorflow import keras
import json

app = Flask(
    __name__,
    template_folder="../frontend"
)

IMG_SIZE = 128

model = keras.models.load_model(
    "Emotion_detection_model2.keras"
)

class_names =['angry','disgusted','fearful','happy','neutral','sad','surprised']  


# ==========================================
# HTML IMAGE UPLOAD
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:

        return jsonify({
            "error": "No Image Upload"
        }), 400

    file = request.files["image"]

    try:

        img = Image.open(file)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        img_array = np.array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        print("RAW PREDICTION:", prediction[0])
        print("CLASS NAMES:", class_names)

        predicted_class = np.argmax(
            prediction[0]
        )

        confidence = float(
            prediction[0][predicted_class] * 100
        )

        emotion = class_names[predicted_class]

        print("EMOTION:", emotion)
        print("CONFIDENCE:", confidence)

        return render_template(
            "result.html",
            emotion=emotion,
            confidence=round(confidence, 2)
        )

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# OPENCV LIVE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        img = Image.open(file)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        img_array = np.array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction[0]
        )

        confidence = float(
            prediction[0][predicted_class] * 100
        )

        emotion = class_names[predicted_class]

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True
    )