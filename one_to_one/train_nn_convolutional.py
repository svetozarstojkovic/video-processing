from keras.callbacks import TensorBoard, Callback
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, np
from keras.models import Model
from keras import backend as K

temp_input_array = np.load('../output/output.npy')
temp_input_array = temp_input_array.reshape((temp_input_array.shape[0], 28, 28, 3))

test_input_array = np.load('../output/output_test.npy')
test_input_array = test_input_array.reshape((test_input_array.shape[0], 28, 28, 3))

input_array = temp_input_array[:, :, :, 0]
input_array = np.append(input_array, temp_input_array[:, :, :, 1], axis=0)
input_array = np.append(input_array, temp_input_array[:, :, :, 2], axis=0)
input_array = np.reshape(input_array, (input_array.shape[0],
                                       input_array.shape[1],
                                       input_array.shape[2],
                                       1))

test_input_array = test_input_array[:, :, :, 0]
test_input_array = np.reshape(test_input_array, (test_input_array.shape[0],
                                                 test_input_array.shape[1],
                                                 test_input_array.shape[2],
                                                 1))

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# output_array = np.load('../output/output.npy')
# test_output_array = np.load('../output/output_test.npy')


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format


x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(input_array, input_array,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(test_input_array, test_input_array))

autoencoder.save('one_to_one_convolutional.h5')
