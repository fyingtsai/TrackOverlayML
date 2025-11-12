from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.activations import sigmoid, relu

def getTrkPredictor(features, layers, activation=relu, name=None):

    assert len(layers)

    finputs = Input(shape=(features,), name="input_layer")

    hidden = Dense(units=layers[0], activation=activation, name="hidden_0")(finputs)

    for i,node in enumerate(layers[1:],1):
        hidden = Dense(units=node, activation=activation, name=f"hidden_{i}")(
            hidden
        )

    output = Dense(1, activation=sigmoid, name=f"hidden_{len(layers)}")(hidden)

    return Model(inputs=finputs, outputs=output, name=name)
