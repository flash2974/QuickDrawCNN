import json
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import DATASET_DIR, IMG_DIR, MODELS_DIR
from .models import model_list


def train_save_models(max_size=10_000):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    classes_path = MODELS_DIR / "classes.json"

    if classes_path.exists():
        saved_classes = json.loads(classes_path.read_text(encoding="utf-8"))
        expected_models = model_list(len(saved_classes))
        if all((MODELS_DIR / f"{m.name}.keras").exists() for m in expected_models):
            return saved_classes

    # Load dataset
    assert DATASET_DIR.exists(), "Pas de dataset pour train"

    files = sorted(name for name in os.listdir(DATASET_DIR) if name.endswith(".npy"))
    assert files, "Aucun fichier .npy trouvé dans le dataset"

    draw_class = []
    X_train = np.zeros((max_size, 28, 28))
    y_train = np.zeros((max_size,))

    max_size_per_cl = int(max_size / len(files))

    it = 0
    t = 0
    for name in tqdm(files, desc="Loading dataset"):
        draw_class.append(name.replace("full_numpy_bitmap_", "").replace(".npy", ""))
        draws = np.load(DATASET_DIR / name, mmap_mode="r")
        draws = draws[:max_size_per_cl]

        X_train[it : it + draws.shape[0]] = np.invert(draws.reshape(-1, 28, 28))
        y_train[it : it + draws.shape[0]] = t
        # Iter
        it += draws.shape[0]
        t += 1

    X_train = X_train.astype(np.float32)

    X_train = X_train[:it]
    y_train = y_train[:it]

    # on mélange le dataset
    indexes = np.arange(it)
    np.random.shuffle(indexes)
    X_train = X_train[indexes]
    y_train = y_train[indexes]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    X_train = X_train.reshape(-1, 28, 28, 1)

    # models
    num_classes = len(draw_class)
    for model in model_list(num_classes):
        model_path = MODELS_DIR / f"{model.name}.keras"
        if not model_path.exists():
            print(f"Training {model.name}")
            model.compile(
                loss="sparse_categorical_crossentropy",  # sparse : label = int au lieu de hot encoding (3 au lieu de 0000000011 )
                optimizer="adam",
                metrics=[
                    "accuracy"
                ],  # Connaître la performance du modèle, cb de prédictions correctes
            )

            model.fit(
                X_train, y_train, epochs=10, validation_data=(X_test, y_test)
            )  # on explore 10 fois le dataset
            model.save(model_path)

    classes_path.write_text(json.dumps(draw_class, indent=2), encoding="utf-8")
    return draw_class


def preprocess_image(name, save=False):
    img = Image.open(name).convert("L")
    arr = np.array(img)

    draw_positions = np.argwhere(
        arr < 255
    )  # postiions ou la valeur est < 255 (ou c pas du blanc)
    y_min, x_min = draw_positions.min(axis=0)  # sur les rows
    y_max, x_max = draw_positions.max(axis=0)  # sur les rows
    h = y_max - y_min
    w = x_max - x_min
    size = max(h, w)

    # centrer
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    y_min = max(cy - size // 2, 0)
    y_max = min(cy + size // 2, arr.shape[0])
    x_min = max(cx - size // 2, 0)
    x_max = min(cx + size // 2, arr.shape[1])

    # rogner
    arr = arr[y_min:y_max, x_min:x_max]

    # redim
    img = Image.fromarray(arr, mode="L").resize((28, 28))
    if save:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        img.save(IMG_DIR / "preprocessed.jpg")
    arr = np.array(img).reshape(1, 28, 28, 1).astype(np.float32)
    return arr


def plot_image(arr):
    plt.imshow(arr.reshape(28, 28, 1), cmap="gray")
    plt.show()
