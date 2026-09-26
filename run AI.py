
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import keras
import numpy as np
# model list coin_classifier.keras coin_detector 10.6m.keras
model = keras.models.load_model("coin_classifier_int4.keras")
def main():
    test1 = input("write the name of your coin image here (encluding the extension)> ")
    image = keras.utils.load_img(
        test1,
        target_size=(530, 530)
    )

    image = keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("int8")

    prediction = model(image, training=False).detach().cpu().numpy()[0][0]

    print("Prediction:", prediction)

    if prediction <= 0.5:
        print("HEADS")
    else:
        print("TAILS")

    test2 = input("press enter to continue or type end to stop> ")

    if test2 == "end":
        return

    main()

if __name__ == "__main__":
    main()
