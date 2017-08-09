import csv
from scipy import ndimage
import numpy as np
import random

def normalize(num):
    return num  / 255.

def vectorify(matrix):
    return [item for sublist in matrix for item in sublist]
    # np.reshape(matrix, (-1, 1))

# images = []
vectors = []
reshaped = []

with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        vectors.append(list(map(int, row)))
        # matrices = np.reshape(list(map(int, row[1:785])), (-1, 28))
        # images.append(matrices)


for vector in vectors:
    number = vector[0]
    image = np.reshape(vector[1:785], (-1, 28))

    rotatedImg = ndimage.interpolation.rotate(image, random.randint(-10, 10))
    shiftedImg = ndimage.interpolation.shift(rotatedImg, [random.randint(-4, 4), random.randint(-4, 4)])
    blurredImg = ndimage.filters.gaussian_filter(shiftedImg, random.randint(0, 1))

    reshaped.append(np.concatenate(([number], vectorify(blurredImg)), axis=0))

with open('reshaped.csv', 'w') as f:
    for vector in reshaped:
        f.write( ','.join([str(x) for x in vector]) )#','.join(map(str, s)))
        f.write("\n")
