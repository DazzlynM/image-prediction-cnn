from xrayapp.models import FileData
from django.shortcuts import render
from .form import FileDataForm
import tensorflow as tf
import os
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import json
import cv2
# To check the file extension of the image
import imghdr
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
from django.core.files.storage import FileSystemStorage

def imageProcessing():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# remove dodgy images
    data_dir = 'data/train'
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']

    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
            
            
    data_train= tf.keras.utils.image_dataset_from_directory('data/train')
    data_val= tf.keras.utils.image_dataset_from_directory('data/val')
    data_test = tf.keras.utils.image_dataset_from_directory('data/test')

    data_iterator = data_train.as_numpy_iterator()
    # batch = data_iterator.next() 

    data_train = data_train.map(lambda x,y: (x/255, y))
    data_val = data_val.map(lambda x,y: (x/255, y))
    data_test = data_test.map(lambda x,y: (x/255, y))
#   Building the model
    model = Sequential()

    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    logdir= 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(data_train, epochs=2, validation_data=data_val, callbacks=[tensorboard_callback])
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in data_test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
    return model

def index(request):
    # model = imageProcessing()
    new_model = load_model(os.path.join('models', 'pneumoniaClassifier.h5'))
    testResult = ""
    if request.method == "POST":
        form = FileDataForm(
            data=request.POST,
            files=request.FILES
            
        )
        if form.is_valid():
            form.save()
            obj=form.instance
           
            print('obj.image.url', obj.image.url)
            img = cv2.imread('.'+ obj.image.url)
            resize = tf.image.resize(img, (256,256))
            # yhat = model.predict(np.expand_dims(resize/255, 0))
            yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
            if yhatnew > 0.5:
                testResult='Predicted class is Pneumonia'
            else:
                testResult = 'Predicted class is Normal'
            context = {"obj":obj, "testResult": testResult}
            return render(request, "index.html", context)
    else:
        form=FileDataForm()
        image=FileData.objects.all()
    return render(request, "index.html", {"image":image, "form":form})


    