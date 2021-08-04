import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


# from tensorflow import keras
# from tqdm import tqdm
# import pickle
# from tensorflow.keras import models
#
# X_test = pickle.load(open("X_predict.pickle", "rb"))
# # X_train = numpy.array(X_train)
#
# y_test = pickle.load(open("y_test.pickle", "rb"))
#
#
# model = keras.models.load_model("Model/")
# print(model.predict_classes(np.array(X_test)))

src = cv2.imread(r"test\test.jpg")

src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src = cv2.blur(src, (2, 2))
ret, src = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
src = cv2.resize(src, (45, 45))

binary_map = (src > 0).astype(np.uint8)

connectivity = 4

output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)

(numLabels, labels, stats, centroids) = output

for i in range(0, numLabels):
    # if this is the first component then we examine the
    # *background* (typically we would just ignore this
    # component in our loop)
    if i == 0:
        text = "examining component {}/{} (background)".format(
            i + 1, numLabels)
    # otherwise, we are examining an actual connected component
    else:
        text = "examining component {}/{}".format(i + 1, numLabels)
    # print a status message update for the current connected
    # component
    print("[INFO] {}".format(text))
    # extract the connected component statistics and centroid for
    # the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

output = np.copy(src)
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)