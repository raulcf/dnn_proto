import numpy as np
import config as C

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

np.random.seed(C.random_seed)


def build_sequential_model():
    model = Sequential() 
    model.add(Dense(output_dim=64, input_dim=30, init="uniform", activation="relu"))
    model.add(Dense(output_dim=32, input_dim=30, init="uniform", activation="relu"))
    model.add(Dense(output_dim=16, input_dim=30, init="uniform", activation="relu"))
    model.add(Dense(output_dim=8, input_dim=30, init="uniform", activation="relu"))
    model.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


def fit_model_batch(model, x, y, num_epoch=None):
    if num_epoch is None:
        num_epoc = 50
    model.fit(x, y, nb_epoch=num_epoch, batch_size=1)
    return model


def predict_with_model(x, model):
    c = model.predict_classes(x)
    return c

if __name__ == "__main__":
    print("todo")
