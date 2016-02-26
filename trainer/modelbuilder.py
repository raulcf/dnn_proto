from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

def build_sequential_model():
    model = Sequential() 
    model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10, init="glorot_uniform"))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=0.01,
                                      momentum=0.9, 
                                      nesterov=True))
    return model

def fit_model_batch(model, x, y):
    model.fit(x, y, nb_epoch=5, batch_size=1)
    return model

def predict_with_model(x, model):
    c = model.predict_class(x)
    return c

if __name__ == "__main__":
    print("todo")
