import glob
from sys import getsizeof
from sklearn.model_selection import train_test_split
from config import Config
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

class Preprocess:
    def __init__(self):
        config = Config()
        self.num_classes = config.num_attributes
        self.path = config.path
        self.dataset_length = config.dataset_length
        self.labels_dict = config.labels_dict

    def process(self):

        x = np.zeros((self.dataset_length, 224, 224, 3), dtype=np.short)
        y = np.zeros((self.dataset_length, self.num_classes), dtype=np.short)
        print(x.nbytes, y.nbytes)
        count = 0

        for outer in glob.glob(self.path + "*"):
            print(outer)
            if 'CUHK' in outer:
                file_read = str(outer)+'/archive/Label.txt'
                f = open(file_read, "r")
                labels = f.readlines()
                for i in labels:
                    split_label = i.split()
                    img_name = split_label[0]
                    attr = split_label[1:]
                    y_label = [0] * self.num_classes
                    for attributes in attr:
                        if attributes in self.labels_dict:
                            y_label[self.labels_dict[attributes]] = 1

                    images_glob = glob.glob(str(outer) + "/archive/{}*.jpeg".format(img_name))
                    if len(images_glob) < 1:
                        images_glob = glob.glob(str(outer) + "/archive/{}*.png".format(img_name))
                    if len(images_glob) < 1:
                        images_glob = glob.glob(str(outer) + "/archive/{}*.jpg".format(img_name))
                    if len(images_glob) < 1:
                        images_glob = glob.glob(str(outer) + "/archive/{}".format(img_name))

                    for j in images_glob:
                        t = load_img(j)
                        im = img_to_array(t)
                        img = preprocess_input(im)

                        img_resize = np.resize(img, (224,224,3))
                        try:
                            x[count] = img_resize
                            y[count] = y_label
                        except:
                            break
                        count += 1
                        if count % 100 == 0:
                            print('.',end='')
                        break

        x, X_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        print(x.shape)

        return x, X_test, y, y_test

