# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
import cv2
import time
directory = r'/Users/deevyanshkhadria/Desktop'
category = ['Cat', 'Dog']
img_size = 100
data = []

for i in category:
    folder = os.path.join(directory, i)
    label = category.index(i)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)

        try:
            img_arr = cv2.imread(img_path)
            if img_arr is not None:
                img_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([img_arr, label])
            else:
                print(f"Error reading image: {img_path}")
        except Exception as e:
            print(f"Error processing image: {img_path}\n{str(e)}")
random.shuffle(data)
x=[]
y=[]
for features,lable in data:
    x.append(features)
    y.append(lable)
X=np.array(x)
Y=np.array(y)
# Continue with the rest of your code
pickle.dump(X,open('X.pkl','wb'))
pickle.dump(Y,open('Y.pkl','wb'))

X=pickle.load(open('X.pkl','rb'))
Y=pickle.load(open('Y.pkl','rb'))
X=X/255
print(X.shape)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard

name=f'cat-vs-dog-prediction-{int(time.time())}'

tenserboard= TensorBoard(log_dir=f'venv1\\{name}\\')

model= Sequential()
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,input_shape=X.shape[1:],activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X,Y,epochs=5, validation_split=0.1,batch_size=32,callbacks=[tenserboard])
