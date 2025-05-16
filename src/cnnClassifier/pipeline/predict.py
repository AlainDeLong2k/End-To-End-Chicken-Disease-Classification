import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnnClassifier.constants import CONFIG_FILE_PATH
from cnnClassifier.utils.common import read_yaml


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.config = read_yaml(CONFIG_FILE_PATH)

    def predict(self):
        # Load model
        model = tf.keras.models.load_model(
            Path(self.config.training.trained_model_path)
        )

        image_name = self.filename
        test_image = tf.keras.preprocessing.image.load_img(
            image_name, target_size=(224, 224), interpolation="bilinear"
        )
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Healthy"
        else:  # result[0] == 0
            prediction = "Coccidiosis"

        return [{"image": prediction}]
