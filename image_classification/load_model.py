import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras import datasets, layers, models
import photos as ph

# upload folders path
image_fname = "horse_scaled.jpg"
test_image = os.path.join(r"C:\Users\Michael Harris\Desktop\Masters CS\Clemson MSCS\sideProjects\deep learning\photos", image_fname)


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

############### load model and insert test image for testing ################

# Reducing the size of the data so that my computer doesn't explode
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classifier.model')

img = cv.imread(test_image)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)

index = np.argmax(prediction)
print(f"prediction: {class_names[index]}")