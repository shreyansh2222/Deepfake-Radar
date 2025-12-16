from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/deepfake_model.h5")
print("Model loaded successfully")

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)

def frame_consistency_score(frames):
    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i] - frames[i - 1]))
        diffs.append(diff)
    return float(np.mean(diffs))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        video = request.files.get("video")

        if video:
            video_path = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(video_path)

            frames = extract_frames(video_path)

            if frames is None:
                result = "INVALID VIDEO"
                confidence = 0
            else:
                ml_input = np.expand_dims(frames, axis=0)
                ml_pred = model.predict(ml_input, verbose=0)[0][0]

                consistency = frame_consistency_score(frames)
                heuristic_pred = min(consistency / 0.03, 1.0)

                final_score = (0.3 * ml_pred) + (0.7 * heuristic_pred)

                if final_score <= 0.5:
                    result = "REAL"
                    confidence = round((1 - final_score) * 100, 2)
                else:
                    result = "FAKE"
                    confidence = round(final_score * 100, 2)

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    print("Flask server running...")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)



