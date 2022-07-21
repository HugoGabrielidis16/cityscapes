import segmentation_models as sm
import tensorflow as tf


def create_model():
    model = sm.Unet("resnet34", classes=3, activation="sigmoid")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", sm.metrics.iou_score],
    )
    return model


def processing_input():
    BACKBONE = "resnet34"
    preprocess_input = sm.get_preprocessing(BACKBONE)
    return preprocess_input
