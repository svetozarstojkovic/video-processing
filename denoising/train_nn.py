from __future__ import print_function

import os

import keras
import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Input
from keras.models import Model
from keras.optimizers import SGD

np.random.seed(1337)  # for reproducibility



# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
#
# # skaliranje na [0,1]
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

input_array = np.load('input/input.npy')
test_input_array = np.load('input/input_test.npy')

output_array = np.load('output/output.npy')
test_output_array = np.load('output/output_test.npy')

# temp_input_array = np.load('../output/output.npy')
# temp_input_array = temp_input_array.reshape((temp_input_array.shape[0], 28, 28, 3))
#
# temp_test_input_array = np.load('../output/output_test.npy')
# temp_test_input_array = temp_test_input_array.reshape((temp_test_input_array.shape[0], 28, 28, 3))

# output_array = np.load('../output/output.npy')
# test_output_array = np.load('../output/output_test.npy')

# input_array = temp_input_array[:, :, :, 0]
# input_array = np.append(input_array, temp_input_array[:, :, :, 1], axis=0)
# input_array = np.append(input_array, temp_input_array[:, :, :, 2], axis=0)
# input_array = np.reshape(input_array, (input_array.shape[0],
#                                        input_array.shape[1],
#                                        input_array.shape[2],
#                                        1))
#
# test_input_array = temp_test_input_array[:, :, :, 0]
# test_input_array = np.reshape(test_input_array, (test_input_array.shape[0],
#                                                  test_input_array.shape[1],
#                                                  test_input_array.shape[2],
#                                                  1))


# input_array = np.round(input_array, 2)
# output_array = np.round(output_array, 2)


# defining autoencoder
# encoder
inpt = Input((300, ))
encoder = Dense(100)(inpt)
encoder = Activation('relu')(encoder)
encoder = Dense(10)(encoder)
encoder = Activation('relu')(encoder)

# decoder
decoder = Dense(10)(encoder)
decoder = Activation('relu')(decoder)
decoder = Dense(100)(decoder)
decoder = Activation('relu')(decoder)
decoder = Dense(300)(decoder)
decoder = Activation('sigmoid')(decoder)

# encoder and decoder
model = Model(input=inpt, output=decoder)

sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9)
opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# uzimamo nasumicnih 10 primera iz testnog skupa za vizualizaciju rezultata
# idx = np.random.randint(test_input_array.shape[0], size=(10, ))
# input_vect = test_input_array[idx]
# input_imgs = input_vect.reshape(input_vect.shape[0], 10, 10, 3)


# output_imgs_dir = '../imgs/mnist_mlp_autoencoder'
# if not os.path.exists(output_imgs_dir):
#     os.mkdir(output_imgs_dir)


# utility funkcija za iscrtavanje rezultata: ulazna slika -> rekonstruisana slika
def generate_images(epoch):
    # output_vect = model.predict(input_vect)
    a = 1
    # output_imgs = output_vect.reshape(output_vect.shape[0], 36, 64, 3)
    # combined_img = combine_images(input_imgs, output_imgs)

    # imsave(output_imgs_dir + '/epoch_{}.png'.format(epoch), combined_img)


def combine_images(inpt_imgs, outpt_imgs):
    num = inpt_imgs.shape[0]
    width = num
    height = 2
    shape = inpt_imgs.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], 3), dtype=inpt_imgs.dtype)
    combined = np.concatenate((inpt_imgs, outpt_imgs))
    for index, img in enumerate(combined):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return combined



    # background = Image.new('RGBA', (inpt_imgs[0].shape[1] * 10, inpt_imgs[0].shape[0] * 2), (255, 255, 255, 255))
    #
    # for i, img in enumerate(inpt_imgs):
    #     input_image = img
    #     output_image = outpt_imgs[i]
    #
    #     # input_image = np.array(input_image) * 255
    #     # output_image = np.array(output_image) * 255
    #
    #     inp_image = Image.fromarray(input_image, mode='RGB')
    #     out_image = Image.fromarray(output_image, mode='RGB')
    #     inp_image.show()
    #     background.paste(inp_image, (i * inpt_imgs[0].shape[1], 0))
    #     background.paste(out_image, (i * inpt_imgs[0].shape[1], inpt_imgs[0].shape[0]))
    #
    # return background


# Callback klasa iscrtavanje rezultata autoencodera
class CombineImagesCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        generate_images(epoch)


batch_size = 32
nb_epoch = 300

# training autoencoder
model.fit(input_array, output_array,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          validation_data=(test_input_array, test_output_array),
          callbacks=[CombineImagesCallback()])

model.save('denoising.h5')
