from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, np
from keras.models import Model

input_array = []
output_array = []

test_input_array = []
test_output_array = []

factor = 10


def init():
    global input_array, output_array, test_input_array, test_output_array

    input_array = process_input_dataset('input/input.npy')
    output_array = process_output_dataset('output/output.npy')

    test_input_array = process_input_dataset('input/input_test.npy')
    test_output_array = process_output_dataset('output/output_test.npy')

    create_network()


def process_input_dataset(location):
    temp_array = np.load(location)
    temp_array = temp_array.reshape((temp_array.shape[0], factor, factor, 1))

    ret_array = temp_array[:, :, :, 0]
    # ret_array = np.append(ret_array, temp_array[:, :, :, 1], axis=0)
    # ret_array = np.append(ret_array, temp_array[:, :, :, 2], axis=0)
    ret_array = np.reshape(ret_array, (ret_array.shape[0],
                                       ret_array.shape[1],
                                       ret_array.shape[2],
                                       1))
    return ret_array


def process_output_dataset(location):
    temp_array = np.load(location)
    temp_array = temp_array.reshape((temp_array.shape[0], factor, factor, 3))

    # ret_array = temp_array[:, :, :, 0]
    # ret_array = np.append(ret_array, temp_array[:, :, :, 1], axis=0)
    # ret_array = np.append(ret_array, temp_array[:, :, :, 2], axis=0)
    # ret_array = np.reshape(ret_array, (ret_array.shape[0],
    #                                    ret_array.shape[1],
    #                                    ret_array.shape[2],
    #                                    1))

    return temp_array


def create_network():
    global input_array, output_array, test_output_array, test_input_array
    input_img = Input(shape=(factor, factor, 1))  # adapt this if using `channels_first` image data format

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
    x = Conv2D(16, (3, 3), activation='relu')(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    epochs = 100
    batch_size = 32

    autoencoder.fit(input_array, output_array,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_input_array, test_output_array))

    autoencoder.save('gray_to_color_convolutional.h5')


init()
