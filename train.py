from distutils.command.config import config
import tensorflow as tf
from model import create_model
from data import load_dataset
from config import Config

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    config = Config("RESUNET")
    train_ds, val_ds, test_ds = load_dataset()
    print("Number of GPUs used : {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model()
        history = model.fit(
            train_ds,
            batch_size=config.train_batch_size,
            epochs=config.epochs,
            validation_data=val_ds,
        )
        model.save(config.saving_path)
        evaluation = model.evaluate(test_ds)
