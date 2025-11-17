import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

MAX_SIZE = 10_000

def train_save_models() :
    # Load dataset
    dataset_dir = "quickdraw_dataset"
    
    assert os.path.exists(dataset_dir), "Pas de dataset pour train"
    
    files = [name for name in os.listdir(dataset_dir) if ".npy" in name]
    draw_class = []
    X_train = np.zeros((MAX_SIZE, 28, 28))
    y_train = np.zeros((MAX_SIZE,))
    
    max_size_per_cl = int(MAX_SIZE / len(files))

    it = 0
    t = 0
    for name in tqdm(files, desc = 'Loading dataset'):
        # Open each dataset and add the new class
        draw_class.append(name.replace("full_numpy_bitmap_", "").replace(".npy", ""))
        draws = np.load(os.path.join(dataset_dir, name))
        draws = draws[:max_size_per_cl] # Take only 10 000 draw
        # Add X_train to the buffer
        X_train[it:it+draws.shape[0]] = np.invert(draws.reshape(-1, 28, 28))
        y_train[it:it+draws.shape[0]] = t
        # Iter
        it += draws.shape[0]
        t += 1

    X_train = X_train.astype(np.float32)

    # Keep only filled entries (avoid zeros from preallocated buffer)
    X_train = X_train[:it]
    y_train = y_train[:it]

    # Shuffle dataset (only the real entries)
    indexes = np.arange(it)
    np.random.shuffle(indexes)
    X_train = X_train[indexes]
    y_train = y_train[indexes]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    X_train = X_train.reshape(-1, 28, 28, 1)

    # Model
    num_classes = len(draw_class)
    for model in model_bench(num_classes) : 
        print(f'Training {model.name}')
        model.compile(
            loss = "sparse_categorical_crossentropy", # sparse : label = int au lieu de hot encoding (3 au lieu de 0000000011 )
            optimizer = "adam",
            metrics = ["accuracy"] # Connaître la performance du modèle, cb de prédictions correctes
        )

        model.fit(X_train, y_train, epochs=10, validation_data = (X_test, y_test)) # on explore 10 fois le dataset
        model.save(f'models/{model.name}.keras')
    
    
    
def model_bench(num_classes = 10, input_shape=(28,28,1)) :
    BasicCNN = Sequential(name = 'BasicCNN', layers = [
        Conv2D(32, 4, activation = 'relu', input_shape=input_shape),
        Conv2D(64, 3, activation = 'relu'),
        Conv2D(128, 3, activation = 'relu'),
        
        Flatten(),
        Dense(128, activation = 'relu'),
        Dense(num_classes, activation = 'softmax')
    ])
    
    MaxPooling = Sequential(name = 'MaxPooling', layers = [
        Conv2D(32, 4, activation = 'relu', padding='same', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(64, 3, activation = 'relu', padding='same'),
        MaxPooling2D(),
        Conv2D(128, 3, activation = 'relu', padding='same'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation = 'relu'),
        Dense(num_classes, activation = 'softmax')
    ])
    
    AVGPooling_Dropout = Sequential(name = 'AVGPooling_Dropout', layers = [
        Conv2D(32, 3, activation = 'relu', padding='same', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(64, 3, activation = 'relu', padding='same'),
        MaxPooling2D(),
        Conv2D(128, 3, activation = 'relu', padding='same'),
        MaxPooling2D(),
        
        GlobalAveragePooling2D(),
        Dense(128, activation = 'relu'),
        Dropout(0.3),
        Dense(num_classes, activation = 'softmax')
    ])
    
    return [BasicCNN, MaxPooling, AVGPooling_Dropout]