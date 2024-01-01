from config import Config
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from dafl import SSCAModule, GAMModule, DAFLModel

config = Config()
image_path = '/Users/mayank57/Downloads/PETA/PETA dataset/CUHK/archive/4550.png'
checkpoint_path = config.checkpoint_path + "/cp-{epoch:04d}.ckpt"
print(config.labels_dict.keys())
attr = list(config.labels_dict.keys())

t = load_img(image_path)
im = img_to_array(t)
img = preprocess_input(im)
img_resize = np.resize(img, (224,224,3))

model = DAFLModel(config.num_attributes, config.num_groups, config.momentum, config.group_indices_list, config.num_cascades)
best_epoch = 0
model.load_weights(checkpoint_path.format(epoch=best_epoch))

expanded_array = np.expand_dims(img_resize, axis=0)
pred = model(expanded_array, config.num_attributes, training = False)[0]

print(pred)
for i in range(len(pred[0])):
    if pred[0][i] > 0.5:
        print(attr[i])