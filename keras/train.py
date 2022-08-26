from model import create_model
from data import to_dataset
from config import Config
import tensorflow as tf
from vgg16 import segmentation_model


if __name__ == "__main__":
    config = Config("RESUNET")
    train_ds, val_ds, test_ds = to_dataset()

    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.log_device_placement = True
    session = tf.compat.v1.Session(config=conf)

    model = segmentation_model()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
    )
    model.save(config.saving_path)
    evaluation = model.evaluate(test_ds)
