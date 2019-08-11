import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
#from keras import backend as K
#K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#TEST_SET = ['1.png','2.png','3.png']
#TEST_SET = ['subpalm_01_11.jpg','subpalm_18_02.jpg']
#TEST_SET = ['equal_palm.jpg']
#TEST_SET = ['equal_palm_20180229_1_141.jpg','equal_palm_20180229_1_21.jpg','equal_palm_20180229_1_31.jpg','equal_palm_20180229_1_81.jpg','equal_palm_20180229_2_21.jpg','equal_palm_20180229_3_21.jpg','equal_palm_20180229_4_7.jpg','equal_palm_20180229_4_81.jpg','equal_palm_20180229_5_71.jpg']
#TEST_SET = ['equal_palm_brasil1_0_16.jpg','equal_palm_brasil1_0_17.jpg','equal_palm_brasil1_1_13.jpg','equal_palm_brasil1_1_18.jpg','equal_palm_brasil1_1_19.jpg','equal_palm_ghana2_1_27.jpg','equal_palm_test-20180229_2_141.jpg','equal_palm_test-20180229_2_71.jpg','equal_palm_test-20180229_5_81.jpg','equal_palm_test-20180229_7_151.jpg','equal_palm_test-20180229_7_161.jpg']
TEST_SET = ['equal_test_mala_east1.jpg','equal_test_mala_east2.jpg','equal_test_mala_middle1.jpg','equal_test_mala_middle2.jpg','equal_test_mala_north1.jpg','equal_test_mala_north2.jpg','equal_test_mala_west1.jpg','equal_test_mala_west2.jpg']

image_size = 256

#classes = [1.,  2.,   3.  , 4.]  
classes = [1.,  2.,   3. ] 
 
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('./test_plam/use_test_jpg_others/' + path)
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print ('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
                print(crop.shape)
                ch,cw,_ = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue
                    
                crop = np.expand_dims(crop, axis=0) 
                pred = model.predict(crop,verbose=2)
#                print(np.shape(pred))
#                print(pred)
#                pred = labelencoder.inverse_transform(pred[0])
#                pred = pred[:,1:4,:,:]
                pred = np.argmax(pred,axis=3)
#                print(np.shape(pred))
#                print (np.unique(pred))  
                pred = pred.reshape((256,256)).astype(np.uint8)
                #print 'pred:',pred.shape
                if(i==0):
                  if(j==0):
                    mask_whole[0:int(stride+stride/2),0:int(stride+stride/2)] = pred[0:int(stride+stride/2),0:int(stride+stride/2)]
                  elif(0<j<(padding_w//stride)-2):
                    mask_whole[0:int(stride+stride/2),int(j*stride+stride/2):int(j*stride+image_size-stride/2)] = pred[0:int(stride+stride/2),int(stride/2):int(image_size-stride/2)]
                  else:
                    mask_whole[0:int(stride+stride/2),int(j*stride+stride/2):] = pred[0:int(stride+stride/2),int(stride/2):]
                elif(0<i<(padding_h//stride)-2):
                  if(j==0):
                    mask_whole[int(i*stride+stride/2):int(i*stride+image_size-stride/2),0:int(stride+stride/2)] = pred[int(stride/2):int(image_size-stride/2),0:int(stride+stride/2)] 
                  elif(0<j<(padding_w//stride)-2):
                    mask_whole[int(i*stride+stride/2):int(i*stride+image_size-stride/2),int(j*stride+stride/2):int(j*stride+image_size-stride/2)] = pred[int(stride/2):int(image_size-stride/2),int(stride/2):int(image_size-stride/2)]
                  else:
                    mask_whole[int(i*stride+stride/2):int(i*stride+image_size-stride/2),int(j*stride+stride/2):] = pred[int(stride/2):int(image_size-stride/2),int(stride/2):]
                else:
                  if(j==0):
                    mask_whole[int(i*stride+stride/2):,0:int(stride+stride/2)] = pred[int(stride/2):,0:int(stride+stride/2)]
                  elif(0<j<padding_w//stride-2):
                    mask_whole[int(i*stride+stride/2):,int(j*stride+stride/2):int(j*stride+image_size-stride/2)] = pred[int(stride/2):,int(stride/2):int(image_size-stride/2)]
                  else:
                    mask_whole[int(i*stride+stride/2):,int(j*stride+stride/2):] = pred[int(stride/2):,int(stride/2):]


#                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        
        cv2.imwrite('./predict_plam/test_pre'+str(n+1)+'.png',mask_whole[0:h,0:w])

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)



