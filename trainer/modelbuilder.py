import numpy as np
import config as C

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

np.random.seed(C.random_seed)


def build_sequential_model():
    model = Sequential() 
    model.add(Dense(output_dim=64, input_dim=30, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=64, input_dim=30, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=64, input_dim=30, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=1, init="glorot_uniform"))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=0.01,
                                      momentum=0.9, 
                                      nesterov=True))
    return model


def fit_model_batch(model, x, y):
    model.fit(x, y, nb_epoch=500, batch_size=1)
    return model


def predict_with_model(x, model):
    c = model.predict_classes(x)
    return c

if __name__ == "__main__":
    print("todo")
