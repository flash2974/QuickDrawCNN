import os
from keras.models import load_model
import numpy as np
from .utils import preprocess_image, train_save_models
from . import MODELS_DIR


MODELS_DIR.mkdir(parents=True, exist_ok=True)

classes = train_save_models()
models = []
for model_name in os.listdir(MODELS_DIR):
    model = load_model(MODELS_DIR / model_name)
    models.append(model)


def prediction(name, save):
    arr = preprocess_image(name, save)

    predictions = []
    for model in models:
        print(f"model {model}")
        res = model.predict(arr)[0]
        print("\tprobs:", res)
        pred_idx = int(np.argmax(res))
        print("\tpredicted class:", classes[pred_idx])

        predictions.append(
            {
                "model": model.name,
                "label": classes[pred_idx].capitalize(),
                "prec": f"{res[pred_idx]:.2%}",
            }
        )

    return predictions
