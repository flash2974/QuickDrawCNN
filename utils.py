from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from train_model import train_save_models
import os
from keras.models import load_model


MODEL_PATH = 'models'
if not os.path.exists(MODEL_PATH) :
    os.makedirs(MODEL_PATH)
    
train_save_models()
models = []
for model_name in os.listdir(MODEL_PATH):
    model = load_model(os.path.join(MODEL_PATH, model_name))
    models.append(model)

def preprocess_image(name, save = False):
    img = Image.open(name).convert('L')
    arr = np.array(img)

    draw_positions = np.argwhere(arr < 255) # postiions ou la valeur est < 255 (ou c pas du blanc)
    y_min, x_min = draw_positions.min(axis=0) # sur les rows
    y_max, x_max = draw_positions.max(axis=0) # sur les rows
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
    img = Image.fromarray(arr, mode='L').resize((28, 28))
    if save : 
        img.save('img/preprocessed.jpg')
    arr = np.array(img).reshape(1, 28, 28, 1).astype(np.float32)
    return arr



def plot_image(arr) : 
    plt.imshow(arr.reshape(28, 28, 1), cmap='gray')
    plt.show()

def prediction(name='img/capture.jpg', save = False) :
    classes = ['airplane', 'apple', 'book', 'brain', 'car', 'chair', 'dog', 'eye', 'face', 'The Eiffel Tower']
    arr = preprocess_image(name, save)
    
    l = []
    for model in models :
        print(f'model {model}')
        res = model.predict(arr)[0]
        print('\tprobs:', res)
        pred_idx = int(np.argmax(res))
        print('\tpredicted class:', classes[pred_idx])
        
        l.append({'model' : model.name, 
                  'label' : classes[pred_idx].capitalize(),
                  'prec' : f'{res[pred_idx]:.2%}'})
    
    return l