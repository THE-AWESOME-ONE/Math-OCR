import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle

DATADIR = "Data/Train"

CATEGORIES = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'alpha', 'ascii_124', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X_test', 'y_test', 'z', '[', ']', '{', '}']


training_data = []
X = []
y = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                #print(path,", ", img)
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                blurImg = cv2.blur(gray, (2, 2))
                ret, thresh1 = cv2.threshold(blurImg, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                final_img = cv2.resize(thresh1, (45, 45))  # resize to normalize data size
                training_data.append([final_img, class_num])  # add this to our training_data
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except Exception as e:
               print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))
print(len(X))
print(len(y))


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, 45, 45, 1))

X = np.array(X).reshape(-1, 45, 45, 1)

pickle_out = open("X_train.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()