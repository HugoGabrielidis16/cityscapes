import tensorflow as tf

dict = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
}

import tensorflow as tf


class Config:
    learning_rate = 3e-4
    size = (224, 224)
    pretrained = True
    epochs = 3
    n_classes = 13

    train_batch_size = 32
    test_batch_size = 64
    seed = True

    # Look into it
    n_split = 5
    split = 0.9  # Need to see the len of the two sets
    # scaler = GradScaler()
    max_gnorm = 1000

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3, min_lr=0.001
        ),
    ]

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.saving_path = f"models/saved/{model_name}.h5"
