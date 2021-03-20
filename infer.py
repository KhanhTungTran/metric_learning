import random
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers


# NOTE: uncomment this if train using GPU
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

queryImageIndex = 21

model = keras.models.load_model('cifar10_metric_model')

def load_raw_data(path):
    features = []
    labels = []
#     cat_input_length = 2000
#     for idx, cat in enumerate(cats):
#     cat_path = os.listdir(path + '/' + cat)
    i = 0
    img_paths = os.listdir(path)
    for img_path in img_paths:
        try:
            img = Image.open(path + '/' + img_path)
            img = img.resize((32, 32))
            # img = img.convert('RGB')
        except Exception as _:
            print(path + '/' + img_path)
            raise(ValueError())
        img = np.array(img, dtype=np.uint8)

        features.append(img)
    print(path + ": done")
    features = np.array(features)
    
    return features

x_test = load_raw_data('./input')
embeddings = model.predict(x_test)

gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix)[:, -(5 + 1) :]

anchor_near_neighbours = list(reversed(near_neighbours[queryImageIndex][:-1]))
imgPaths = os.listdir('./input')
print("Name of the image that we need to query: " + imgPaths[queryImageIndex])
print("Results: ")
for index in anchor_near_neighbours:
    print(imgPaths[index])
# print(anchor_near_neighbours)
# print(near_neighbours)
# num_collage_examples = 1

# examples = np.empty(
#     (
#         num_collage_examples,
#         5 + 1,
#         32,
#         32,
#         3,
#     ),
#     dtype=np.float32,
# )
# for row_idx in range(num_collage_examples):
#     examples[row_idx, 0] = x_test[row_idx]
#     anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
#     for col_idx, nn_idx in enumerate(anchor_near_neighbours):
#         examples[row_idx, col_idx + 1] = x_test[nn_idx]

# def show_collage(examples):
#     box_size = 32 + 2
#     num_rows, num_cols = examples.shape[:2]
#     collage = Image.new(
#         mode="RGB",
#         size=(num_cols * box_size, num_rows * box_size),
#         color=(250, 250, 250),
#     )

#     for row_idx in range(num_rows):
#         for col_idx in range(num_cols):
#             array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
#             collage.paste(
#                 Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
#             )

#     # Double size for visualisation.
#     collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
#     return collage

# plt.imshow(show_collage(examples))
# plt.show()