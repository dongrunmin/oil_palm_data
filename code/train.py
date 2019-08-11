#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras.layers.pooling import AveragePooling2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.merge import Add,Multiply, Concatenate
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
#from keras import backend as K
#K.set_image_dim_ordering('th')

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
seed = 7  
np.random.seed(seed)  
  
#data_shape = 360*480  
img_w = 256
img_h = 256  
#有一个为背景  
#n_label = 4+1  
n_label = 4
  
classes = [0. ,  1.,  2.,   3. ]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  

#image_sets = ['1.png','2.png','3.png']
 

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


#filepath ='./plam_train/'  

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    print ('generateData...')
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'train_data/train_src/' + url)
            img = img_to_array(img)  
            train_data.append(img)  
            label = load_img(filepath + 'train_data/train_label/' + url, grayscale=True) 
            label = img_to_array(label).reshape((img_w * img_h,))
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)
                train_label = to_categorical(train_label,num_classes=n_label)
                train_label = train_label.reshape((batch_size,img_w , img_h,n_label))
#                train_label = np.transpose(train_label,(0,3,1,2))
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'val_data/val_src/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'val_data/val_label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label)  
                valid_label = np.array(valid_label).flatten()
                valid_label = labelencoder.transform(valid_label)
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size,img_w , img_h,n_label))
#                valid_label = np.transpose(valid_label,(0,3,1,2))
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  
  
def cab(input1,input2,fn):
    x = concatenate([input1,input2],axis=3)
    x = AveragePooling2D((x.get_shape().as_list()[1],x.get_shape().as_list()[1]))(x)
    x = Conv2D(fn,(1,1),activation="relu", padding="valid")(x)
    x = Conv2D(input2.get_shape().as_list()[-1],(1,1),padding="valid",activation="sigmoid")(x)
    x = Multiply()([input2,x])

    return x

def rec(input1):
    x = Conv2D(int(float(input1.get_shape().as_list()[-1])/float(4)),(1,1),activation="relu", padding="same")(input1)
    x = Conv2D(int(float(input1.get_shape().as_list()[-1])/float(4)),(3,3),activation="relu", padding="same")(x)
    x = Conv2D(input1.get_shape().as_list()[-1],(1,1),activation="relu", padding="same")(x)
    x = Add()([x,input1])

    return x
  
def unet():
#    inputs = Input((3, img_w, img_h))
    inputs = Input((img_w, img_h,3))
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    use6 = cab(up6, conv4,512)
    use6 = concatenate([up6, use6], axis=3)
    use6 = rec(use6)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(use6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    use7 = cab(up7, conv3,512)
    use7 = concatenate([up7, use7], axis=3)
    use7 = rec(use7)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(use7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

#    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)
 
#    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

#    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


  
def train(args): 
    EPOCHS = 50
    BS = 4
    model = unet()
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')  
    callable = [modelcheck]  
#    train_set,val_set = get_train_val()
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'train_data/train_src/'):
      train_set.append(pic)
    for pic in os.listdir(filepath + 'val_data/val_src/'):
      val_set.append(pic)

    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)  

#    model.save('unet_plam_40_epoch20_para2.h5')
    '''
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    '''
  

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    default=True)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    filepath = args['data']
    time_start=time.time()
    train(args)
    time_end=time.time()
    print('time',time_end-time_start)  
    #predict()  
