from distutils.command.config import config
import segmentation_models as sm
import tensorflow as tf
from config import Config


sm.set_framework("tf.keras")
sm.framework()


def create_model():
    config = Config()
    model = sm.Unet("resnet34", classes=3, activation="sigmoid")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", sm.metrics.iou_score],
    )
    return model


def processing_input():
    BACKBONE = "resnet34"
    preprocess_input = sm.get_preprocessing(BACKBONE)
    return preprocess_input
