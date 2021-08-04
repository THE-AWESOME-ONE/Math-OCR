import pandas as pd
import numpy as np
from os import listdir
import shutil
import os.path
from PIL import Image
import cv2

default_image_size = tuple((45, 45))


dataset_dir = r"./dataset"
try:
    directory_list = listdir(dataset_dir)
    # remove '.DS_Store' from list
    if '.DS_Store' in directory_list:
        directory_list.remove('.DS_Store')
        # remove empty directory
    for directory in directory_list:
        if (len(f"{dataset_dir}/{directory}")) < 1:
            directory_list.remove(directory)
        # check for empty dataset folder
    if len(directory_list) < 1:
        print("Train Dataset folder is empty or dataset folder contains no image")

    for directory in directory_list:
        print(directory)
        image_dir = listdir(f"{dataset_dir}/{directory}")
        if '.DS_Store' in image_dir:
            image_dir.remove('.DS_Store')

        split_point = int(0.9 * len(image_dir))
        train_images, test_images = image_dir[:split_point], image_dir[split_point:]
        for images in train_images:
            image = cv2.imread(f"{dataset_dir}/{directory}/{images}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.blur(image, (2, 2))
            ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if os.path.isdir(f"DS/Train/{directory}"):
                #os.chdir(f"DS/Train/{directory}")
                #preprocess(f"{dataset_dir}/{directory}/{images}").save(f"DS/Test/{directory}/{images}")
                cv2.imwrite(f"DS/Test/{directory}/{images}", thresh1)
                #shutil.copy(f"{dataset_dir}/{directory}/{images}", f"Data/Train/{directory}/{images}")
            else:
                os.mkdir(f"DS/Train/{directory}")
                #os.chdir(f"DS/Train/{directory}")
                #preprocess(f"{dataset_dir}/{directory}/{images}").save(f"DS/Test/{directory}/{images}")
                cv2.imwrite(thresh1, preprocess(f"{dataset_dir}/{directory}/{images}"))
                #shutil.copy(f"{dataset_dir}/{directory}/{images}", f"Data/Train/{directory}/{images}")
        for images in test_images:
            image = cv2.imread(f"{dataset_dir}/{directory}/{images}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.blur(image, (2, 2))
            ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if os.path.isdir(f"DS/Test/{directory}"):
                os.chdir(f"DS/Test/{directory}")
                preprocess(f"{dataset_dir}/{directory}/{images}").save(f"DS/Test/{directory}/{images}")

                # shutil.copy(f"{dataset_dir}/{directory}/{images}", f"Data/Test/{directory}/{images}")
        else:
                os.mkdir(f"DS/Test/{directory}")
                os.chdir(f"DS/Test/{directory}")
                preprocess(f"{dataset_dir}/{directory}/{images}").save(f"DS/Test/{directory}/{images}")
                #cv2.imwrite({images}, preprocess(f"{dataset_dir}/{directory}/{images}"))
                # shutil.copy(f"{dataset_dir}/{directory}/{images}", f"Data/Test/{directory}/{images}")
    print("Done")

except Exception as e:
    print(f"Error : {e}")
