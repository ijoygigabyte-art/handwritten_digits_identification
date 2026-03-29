from sklearn.datasets import fetch_openml
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importing the images from MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

#assigning the images to X and labels to y
X = mnist.data
y = mnist.target

#normalizing the images for easier compute on smaller numbers
X = X / 255.0

#splitting the data into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

#splitting the temporary data into validation and testing sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#saving the images and labels
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

#loading the images and labels
train_images = np.load('X_train.npy',allow_pickle=True)
train_labels = np.load('y_train.npy',allow_pickle=True)

#printing the shape and first image and label
print(train_images.shape)
image = train_images[0].reshape(28,28)
plt.imshow(image, cmap='gray')
plt.show()
print(train_labels.shape)
print(train_labels[:1])

print("Data loaded successfully")

