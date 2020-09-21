from keras.models import load_model
import cv2
import numpy as np
import sys
import time
from prediction import BinPrediction

filepath = sys.argv[1]
choice = sys.argv[2]

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rock-paper-scissors-model.h5")


# prepare the image
img = cv2.imread(filepath + choice + ".jpg")
t = time.time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
print(time.time()-t)

filepath = sys.argv[1]

back = cv2.imread(filepath+"back.jpg")
img = cv2.imread(filepath + choice + ".jpg")

mod = BinPrediction()
mod.setBackground(back)

t = time.time()
pred = mod.predict(img)

print("Predicted: {}".format(pred))
print(time.time()-t)