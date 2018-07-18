from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from keras.layers import Dense, Conv1D, Input, MaxPooling1D, Flatten, Dropout, LSTM
from keras import Sequential
from keras.utils import np_utils
from PIL import Image
import array
import sys
import os
'''
gdal.GetDriverByName('EHdr').Register()
img = gdal.Open("apex17bnads")
b = img.RasterCount
'''
col = 1000
row = 1500
bands = 17
#datatype = gdal.GetDataTypeName(bands)
datatype=np.uint16
#c_c = int(input("Enter Number of Classes"))
c_c = 8
print(datatype)
def train():
    def ReadBilFile(bil,bands,pixels):
        extract_band = 1
        image = np.zeros([pixels, bands], dtype=np.uint16)
        gdal.GetDriverByName('EHdr').Register()
        img = gdal.Open(bil)
        while bands >= extract_band:
            bandx = img.GetRasterBand(extract_band)
            datax = bandx.ReadAsArray()
            temp = datax
            store = temp.reshape(pixels)
            for i in range(pixels):
                image[i][extract_band - 1] = store[i]
            extract_band = extract_band + 1
        return image

    print(os.listdir("17"))
    pixels = row * col
    y_test = np.zeros([row * col], dtype=np.uint16)
    x_test = ReadBilFile("data/apex17bnads", 17, pixels)
    x_test = x_test.reshape(row*col, 17, 1)
    values = []
    path = ["cforest17", "dforest17", "grass17", "river17bands", "art_truf", "roads17", "buildings17"]
    dict1 = {"cforest17":0, "dforest17":0, "grass17":0, "river17bands":0, "art_truf":0, "roads17":0, "buildings17":0}
    c_l = {"cforest17":1, "dforest17":2, "grass17":3, "river17bands":4, "art_truf":5, "roads17":6, "buildings17":7}
    clicks={}
    for address in path:
        with open("17/"+address, "rb") as f:
            k = len(f.read())
            clicks[address] = (k // 2 // bands) if (k // 2 // bands) < 400 else (k // 2 // bands) // 4
            print('{} ==> {}'.format(address, clicks[address]))
             
    for address in path:
        with open("17/"+address, "rb") as f:
            b = array.array("H")
            b.fromfile(f, clicks[address]*bands)
            if sys.byteorder == "little":
                b.byteswap()
            for v in b:
                values.append(v)

    ll = (len(values))
    rex = ll // bands
    print(ll, rex)
    '''from here'''
    f_in = np.zeros([ll], dtype=np.uint16)
    x = 0
    for i in range(ll):
        f_in[x] = values[i]
        x += 1

    sh = int(rex // bands)
    y_train = np.zeros([rex], dtype=np.uint16)
    
    """    print(
        (dict1["Clay"] + dict1["Dalforest"] + dict1["Eufinal"] + dict1["Water"] + dict1["Wheat"] + dict1["Grassland"]) // 2)
    for i in range(dict1["Clay"] // 8):
        y_train[i] = 1
    for i in range(dict1["Dalforest"] // 8):
        y_train[dict1["Clay"] // 8 + i] = 2
    for i in range(dict1["Eufinal"] // 8):
        y_train[(dict1["Clay"] + dict1["Dalforest"]) // 8 + i] = 3
    for i in range(dict1["Water"] // 8):
        y_train[(dict1["Clay"] + dict1["Dalforest"] + dict1["Eufinal"]) // 8 + i] = 4
    for i in range(dict1["Wheat"] // 8):
        y_train[(dict1["Clay"] + dict1["Dalforest"] + dict1["Eufinal"] + dict1["Water"]) // 8 + i] = 5
    for i in range(dict1["Grassland"] // 8):
        y_train[(dict1["Clay"] + dict1["Dalforest"] + dict1["Eufinal"] + dict1["Water"] + dict1["Wheat"]) // 8 + i] = 6"""
   
    '''
    till here
    '''
	#labelling the clicks
    mark = 0
    for add in path:
        for i in range(clicks[add]):
            y_train[mark+i] = c_l[add]
        mark = mark + clicks[add]

        
    x_train = f_in.reshape(rex, bands)

    seed = 7
    np.random.seed(seed)

    x_train = x_train / 2**16-1
    x_test = x_test / 2**16-1
    num_pixels = bands

    for v in y_train:
        print(v, end=" ")

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    n_classes = c_c
    y_test_new = np.zeros([pixels, c_c], dtype=np.uint16)

    print(x_test)
    print(20*'#')
    print(x_train)
    print(20*'#')
    print(y_test)
    print(20*'#')
    print(y_train)

    print(x_test.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(y_test.shape)

    X = x_train.reshape(x_train.shape[0], 17, 1)
    #time_steps=1
    n_units=128
    #n_inputs=4
    n_classes=8
    batch_size=10
    #n_epochs=5
    #data_loaded=false
    #trained=false
    #rnn model
    model = Sequential()
    #model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    #model.add(Conv1D(2 ** 4, 2, activation="relu", padding='same', input_shape=[17, 1]))
    model.add(LSTM(2**7, return_sequences=True,input_shape=[17,1]))
    model.add(MaxPooling1D(2))
    #model.add(Conv1D(2 ** 6, 2, activation="relu", padding='same'))
    model.add(LSTM(2**7, return_sequences=True))
    model.add(MaxPooling1D(2))
    #model.add(Conv1D(2 ** 6, 2, activation="relu", padding='same'))
    model.add(LSTM(2**7, return_sequences=True))
    model.add(MaxPooling1D(2))
    #model.add(Flatten())
    #model.add(Dropout(0.1))
    model.add(LSTM(2**7))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, y_train, batch_size=10, epochs=1500)
    #model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=2)
    y_test_new = model.predict(x_test, batch_size=50)
    print(20*'%')
    #print(y_test_new[:,1])
    print(y_test_new.shape)
    #print(np.squeeze(y_test_new))
    print(20*'%')
    y_test1 = np.argmax(y_test_new, axis=1)
    print(30*'*')
    print("this is predicted output")
    #img = x_test.reshape(row, col, bands)
    #plt.imshow(img)
    #plt.show()
    #result = Image.fromarray((img * 255).astype('uint8'))
    #result.save('image.tiff')


    #img = y_test_new.reshape(row, col)
	#saving the results
    k=y_test1.reshape(row,col)
    plt.imshow(k)
    plt.show()
    result = Image.fromarray((k * (2**16-1)//c_c).astype('uint16'))
    result.save('Classified_images_3/hard_2.tiff')
    '''
    try:
        os.mkdir("Classified_images_12")
    except:
        pass
    '''
    for i in range(1, 8):
        img = y_test_new[:,i].reshape(row, col)
        plt.imshow(img*(2**16-1))
        plt.colorbar()
        plt.show()
        result = Image.fromarray(((img * (2**16-1))).astype('uint16'))
        result.save('Classified_images_3/'+str(i)+'_with_region_2.tiff')


train()

