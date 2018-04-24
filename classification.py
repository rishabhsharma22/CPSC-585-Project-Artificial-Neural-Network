# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:25:01 2018

@author: Rishabh Sharma
"""

import numpy as np
import scipy
import cv2
from keras.models import load_model
import sys, getopt
from keras.preprocessing import image

def main(argv):
    label = ['water','no_water']
    try:
        file = argv[1:]
        print(file[0])
        inputimage = image.load_img(str(file[0]),target_size=(64,64,3))
        inputimage = np.expand_dims(inputimage,axis=0)
#        print ('image read')
        loaded_model = load_model('Rishabh_model_1.h5')
#        print('Here')
#        loaded_model.summary()
        answer = loaded_model.predict(inputimage)
#        print(answer[0])
        print("\n\n\n")
        print("Given image belongs to class",label[int(answer[0][0])])
    except:
        inputimage = 'invalid_file'
        print('unable to find image')
#    model = load_model('Rishabh_model_1.h5')
    
     
if __name__ == '__main__':
    main(sys.argv)