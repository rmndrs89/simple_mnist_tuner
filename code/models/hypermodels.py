from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from kerastuner import HyperModel

class DenseHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        # Define the layers
        input_layer = Input(shape=(28, 28), name='input_layer')
        flatten_layer = Flatten(name='flatten_layer')(input_layer)
        dense_layer = Dense(units=2**hp.Int('units', min_value=3, max_value=5, step=1), activation='relu', name='dense_layer')(flatten_layer)
        output_layer= Dense(units=10, activation='softmax', name='output_layer')(dense_layer)

        # Define the model
        dense_model = Model(inputs=input_layer, outputs=output_layer, name='dense_model')

        # Compile the model
        dense_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return dense_model
