def _load_data():
    """Loads the MNIST dataset.

    Returns
    -------
    (x_train, y_train), (x_test, y_test) : tuple of array_like
    """
    import os
    import numpy as np

    base_dir = '/gxfs_work1/cau/sukne964/Downloads'

    with np.load(os.path.join(base_dir, 'mnist.npz'), allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
