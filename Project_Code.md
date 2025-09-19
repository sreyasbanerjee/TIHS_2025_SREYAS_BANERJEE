'''
IMAGEDATA AUGMENTATION AND IMAGE CLASSIFICATION


QUESTION 1 :

Why do you think the color images displayed above look different (that is, Method 1 vs Method 2)?
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

# Method 1: Reading a color image an converting to grayscale image

import matplotlib.pyplot as plt
from PIL import Image

original = Image.open('Test_img1.jpg')
grayscale = original.convert('L')

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()

# Method 2: Reading a color image an converting to grayscale image (using opencv)

import cv2
original = cv2.imread('Test_img1.jpg')

# Use the cvtColor() function to grayscale the image
grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap='gray')
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()

'''
ANS :

Method 1 : 
The image is being imported a an image object(PIL)
org=Image.open("Test_img1.jpg")
and all the subsequent transformations are performed on the image object itself 

Method 2 : here the image is opened with OpenCV 
org=cv2.imread("Test_img1.jpg")
the image is stored in org as a tuple with shape (height,weight,channel) and the imported inage is in BGR(blue green red) format during grayscale the BGR is converted to grayscale rather than from RGB to grayscale. Hence, we get two different pictures.
In the above code for method 2 the image is forcefully being converted to RGB.
'''

'''
QUESTION 2 :

Implement the following types of image transformation with OpenCV functions:

1. Image resize
2. Image rotation
'''


'''
Rotation matrix :
R=[cos() -sin() 0]
  [sin() cos()  0]
  [0      0     1]
'''
#Since warpAffine demand matrix of shape 2×3 hence the last row
#in omitted though the result is same
#Here rotation is by 30 degrees
r=np.float32([[np.cos(np.radians(30)),-np.sin(np.radians(30)),0],[np.sin(np.radians(30)),np.cos(np.radians(30)),0]])
rr=cv2.warpAffine(original,r,(1024,536))

#Enlarging the image by 2 times
s=np.float32([[2,0,original.shape[1]//4],[0,2,-original.shape[0]//4]])
sr=cv2.warpAffine(original,s,(original.shape[1],original.shape[0]))

fig,axes=plt.subplots(2,1,figsize=(8,8))
axes[0].imshow(cv2.cvtColor(rr,cv2.COLOR_BGR2RGB))
axes[1].imshow(cv2.cvtColor(sr,cv2.COLOR_BGR2RGB))
plt.show()

#Pre requisite for qns 3

# Downlaod the cat and dog dataset
!wget https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip

# unzip the dataset
!unzip Cat_Dog_data.zip

'''
QUESTION 3 :

Load images from the Cat_Dog_data/train folder, define a few additional transforms, then build the dataloader.

'''
#Transforms

data_dir='Cat_Dog_data/train'
transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                              transforms.RandomRotation(30),
                              transforms.RandomAffine(translate=(0.1,0.1),degrees=0),
                              transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                              transforms.Resize((224, 224)),
                              transforms.ToTensor()
                              ])
dataset=datasets.ImageFolder(data_dir,transform=transform)
data_loader=torch.utils.data.DataLoader(dataset,batch_size=5,shuffle=True)

images,labels=next(iter(data_loader))
img=images[0].numpy().transpose((1,2,0))
plt.imshow(img)
plt.show()

#BUILDING THE DATALOADER

class dataloader:
  def __init__(self,dataset,batch_size=1,shuffle=True):
    self.dataset = dataset
    self.batch_size=int(batch_size)
    self.shuffle=bool(shuffle)

  #creating a list for shuffling
    self.indices=list(range(len(dataset)))

  #to track batch size
    self.curr_index=0

  #for iterative purpose
  def __iter__(self):
    if (self.shuffle) :
      random.shuffle(self.indices)
    self.curr_index=0
    return self

  def __next__(self):
    if (self.curr_index >= len(self.indices)):
      raise StopIteration

   #to not exceed the dataset limit
    end_index=min(self.curr_index+self.batch_size,len(self.indices))
    batch_index=self.indices[self.curr_index:end_index]

  batch=[]
    for i in batch_index:
      batch.append(self.dataset[i])

   #seperating images and labels
    images,label=zip(*batch)
    image_tensor=torch.stack(images)
    label_tensor=torch.tensor(label)

  self.curr_index=end_index
  return image_tensor,label_tensor

data_dir = 'Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
                              transforms.CenterCrop(224),
                               transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)

loader=dataloader(dataset,batch_size=1,shuffle=True)

# Display the images
# Run this to test your data loader
images, labels = next(iter(loader))
img = images[0].numpy().transpose((1, 2, 0))
plt.imshow(img, cmap='gray')
plt.show()

#Pre requisite for question 4

''' Finding mean amd standard deviation of MNIST dataset  '''

# Load MNIST without normalization
transform = transforms.ToTensor()
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False)

images, labels = next(iter(trainloader))  # images shape: [60000, 1, 28, 28]

# Flatten the pixels into a single vector
images = images.view(images.size(0), -1)  # shape: [60000, 784]

mean = images.mean().item()
std = images.std().item()

print(f"Mean: {mean}")
print(f"Std: {std}")


# Transformation to Normalize the data
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean),(std))])
#trainset=datasets.MNIST('-/.pytorch/MNIST_data/',download=True,train=True,transform=transform)
#trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

# Get 64 of images and labels
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Convert images back from normalized tensors to numpy arrays for plotting
images = images.numpy()

n=int(input("Enter the number of images you want to plot : "))

fig, axes = plt.subplots(1,n, figsize=(12, 2))
for i in range(n):
    img = images[i].squeeze() * 0.5 + 0.5   # remove channel dim + unnormalize
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")

plt.show()

'''
QUESTION 5 :

MNIST Classification using OOPS by Tensorflow Keras
'''

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class MNISTClassifier:
    def __init__(self):
        self.model: tf.keras.Model | None = None

  def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
        x_test  = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        return (x_train, y_train), (x_test, y_test)

   def build_model(self):
        self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
                ])
        self.model.compile(optimizer="adam",
                                   loss="sparse_categorical_crossentropy",
                                   metrics=["accuracy"])

  def train(self, x_train, y_train, epochs=5, batch_size=64):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

   def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.4f},Test Loss :{loss:.4f}")
        return acc

  def predict_with_open_cv(self,x_test,y_test,num_samples=3,out_dir="mnist_opencv_samples",display=False):
        os.makedirs(out_dir,exist_ok=True)
        for i in range(num_samples):
            digit=(x_test[i]*255).astype("uint8").squeeze(axis=-1)
            label=int(y_test[i])
            filename=os.path.join(out_dir,f"digit_{i}_label_{label}.png")
            cv2.imwrite(filename,digit)

  img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(28,28))
            img=img.astype("float32")/255.0
            img=np.expand_dims(img,axis=(0,-1))

  probs=self.model.predict(img,verbose=0)
            pred_class=int(np.argmax(probs,axis=-1)[0])
            print(f"Sample{i}:True={label}|Pred={pred_class}")

  if display:
    try:
                      cv2.imshow("digit",(img[0,...,0]*255)).astype("uint8")
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
                except cv2.error:
                    print("GUI not available ")

  def run_step(step_no, title, func, *args, **kwargs):
    print(f"\n[Step {step_no}] {title}")
    result = func(*args, **kwargs)
    print(f"[Step {step_no}] Completed.")
    return result

#Execution

if __name__ == "__main__":
      EPOCHS = 5
      BATCH_SIZE = 64
      clf = run_step(1, "Instantiate classifier", MNISTClassifier)
      (x_train,y_train), (x_test,y_test) = run_step(2, "Load MNIST", clf.load_data)
      run_step(3, "Build model", clf.build_model)
      run_step(4, "Train", clf.train, x_train, y_train, EPOCHS, BATCH_SIZE)
      run_step(5, "Evaluate", clf.evaluate, x_test, y_test)
      run_step(6, "Predict with OpenCV", clf.predict_with_open_cv, x_test, y_test, 3)
