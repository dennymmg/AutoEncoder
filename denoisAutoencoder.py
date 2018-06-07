from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop

# extract train data
train_data = np.zeros([9900, 96, 96])
for i in range(1,100):
    if i >=10:
        fnamep1 = "res0" + str(i)
    else:
        fnamep1 = "res00" + str(i)
    for j in range(100):
        if j >= 10:
            fnamep2 = "-0" + str(j) + ".png"
        else:
            fnamep2 = "-00" + str(j) + ".png"
        fname = fnamep1 + fnamep2
        print(fname)
        img = cv2.imread(fname)
        # print("Shape of img", img.shape)
        # crop image to 96 x 96: input image is 96(h) x 128(w)
        h = img.shape[0]
        w = img.shape[1]
        im = img[ :, (w-h)//2:h+(w-h)//2, 1] # all the three channels contain the same data
        train_data[(i-1)*100 + j] = im

# extract test data
test_data = np.zeros([100, 96, 96])
for j in range(100):
    if j >= 10:
        fname = "res100" + "-0" + str(j) + ".png"
    else:
        fname = "res100" + "-00" + str(j) + ".png"
    img = cv2.imread(fname)
    # crop image to 96 x 96: input image is 96(h) x 128(w)
    h = img.shape[0]
    w = img.shape[1]
    im = img[ :, (w-h)//2:h+(w-h)//2, 1] # all the three channels contain the same data
    test_data[j] = im
    
train_data = train_data.reshape(-1, 96, 96, 1)
test_data = test_data.reshape(-1, 96, 96, 1)

print("Shape of train_data = ",train_data.shape)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                              train_data, 
                                                              test_size=0.2, 
                                                            random_state=13)
batch_size = 128
epochs = 5
inChannel = 1
x, y = 96, 96
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    # encoder
    # input = 96 x 96 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # 96 x 96 x 16
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 48 x 48 x 16
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1) # 48 x 48 x 32
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 24 x 24 x 32
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2) # 24 x 24 x 64
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 12 x 12 x 64
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3) # 12 x 12 x 128
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 6 x 6 x 128
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4) # 6 x 6 x 256
    
    # decoder
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5) # 6 x 6 x 256
    up1 = UpSampling2D((2,2))(conv6) # 12 x 12 x 256
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1) # 12 x 12 x 128
    up2 = UpSampling2D((2,2))(conv7) # 24 x 24 x 128
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2) # 24 x 24 x 64
    up3 = UpSampling2D((2,2))(conv8) # 48 x 48 x 64
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3) # 24 x 24 x 32
    up4 = UpSampling2D((2,2))(conv9) # 96 x 96 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up4) # 96 x 96 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
