import numpy as np
from tensorflow import keras
import keras_tuner as kt
from utils import _load_data
from models.hypermodels import DenseHyperModel
import matplotlib.pyplot as plt

INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10

def main():

    # Get the data
    (x, y), (x_test, y_test) = _load_data()

    # Split data in training and validation data
    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]

    # Normalize
    x_train = np.expand_dims(x_train ,-1).astype('float32') / 255.0
    x_val = np.expand_dims(x_val, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

    # Convert class vector to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Get a hypermodel
    dense_hypermodel = DenseHyperModel(INPUT_SHAPE, NUM_CLASSES)

    # Define tuner
    tuner = kt.RandomSearch(
        hypermodel=dense_hypermodel,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=1,
        overwrite=True,
        directory="./hpoptim",
        project_name="simple_mnist",
    )

    # Seach for best hyperparameter configuration
    tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

    # Query the results
    best_hyperparams = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    # Build best model
    best_model = tuner.hypermodel.build(best_hyperparams)
    
    # Retrain the model
    history = best_model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=10,
        verbose=2,
        validation_data=(x_val, y_val),
        shuffle=True)
    
    # Save model
    best_model.save('best_dense_model') 

    # Plot learning curves
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(history.history['loss'], 'ro-')
    axs[0].plot(history.history['val_loss'], 'bo-')
    axs[1].plot(history.history['accuracy'], 'ro-')
    axs[1].plot(history.history['val_accuracy'], 'bo-')
    plt.savefig('learning_curves.png', dpi='figure', format='png')
    plt.close(fig)
    
    print("*** ANALYSIS COMPLETED ***")
    return

if __name__ == "__main__":
    main()
