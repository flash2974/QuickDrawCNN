from flask import Flask, jsonify, request, render_template
from quickdraw_cnn.prediction import prediction
from quickdraw_cnn import IMG_DIR, PROJECT_ROOT
import base64
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = info, 2 = warning, 3 = error


app = Flask(__name__, template_folder=str(PROJECT_ROOT))


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["screenshot"]
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    file.save(IMG_DIR / "capture.jpg")

    save = True  # save preprocessed image
    predictions = prediction(name=str(IMG_DIR / "capture.jpg"), save=save)

    image_data = None
    if save and (IMG_DIR / "preprocessed.jpg").exists():
        with open(IMG_DIR / "preprocessed.jpg", "rb") as f:
            b = f.read()
            image_data = "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")

    return jsonify({"predictions": predictions, "image": image_data})


@app.route("/")
def home():
    return render_template("index.html", label=None)


app.run()
