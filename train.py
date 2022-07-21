from model import create_model
from data import load_dataset

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_dataset()
    model = create_model()
    history = model.fit(
        train_ds,
        batch_size=16,
        epochs=10,
        validation_data=val_ds,
    )
    model.save("UNET_RESNET_backbone.h5")
    evaluation = model.evaluate(test_ds)
