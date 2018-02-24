import imageio
import numpy as np
from cv2 import cv2

from keras.models import load_model
from tqdm import tqdm

from location import new_girl_location

model = load_model('denoising.h5')
model_convolutional = load_model('denoising_convolutional.h5')

factor = 10

noise_factor = 0.2

cap = cv2.VideoCapture(new_girl_location)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 23.976, (1280 * 3, 720 * 2))

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
i = 0
input_array = []
test_input_array = []

output_frames = []
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        i = i + 1
        print('Testing currently working on ' + str(i))
        if i < 100:
            print('Continue: ' + str(i))
            continue

        # Display the resulting frame
        output_frame = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        output_frame_convolutional = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        noisy_frame = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        array = []
        for j in tqdm(range(0, frame.shape[0] // factor)):
            temp_array = []
            for k in range(0, frame.shape[1] // factor):
                temp = np.array(rgb_frame)
                temp = temp[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :]
                temp = temp.reshape((1, factor * factor * frame.shape[2])).astype('float32')
                temp /= 255
                temp = temp + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=temp.shape)
                temp = np.clip(temp, 0., 1.)

                noisy_temp = temp * 255
                noisy_temp = noisy_temp.reshape((factor, factor, frame.shape[2]))
                noisy_frame[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :] = noisy_temp

                prediction = model.predict([temp])

                prediction *= 255
                prediction = prediction.reshape((factor, factor, frame.shape[2]))
                output_frame[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :] = prediction

                temp_array = temp.reshape((1, factor, factor, 3))

                ret_array = temp_array[:, :, :, 0]
                ret_array = np.append(ret_array, temp_array[:, :, :, 1], axis=0)
                ret_array = np.append(ret_array, temp_array[:, :, :, 2], axis=0)
                ret_array = np.reshape(ret_array, (ret_array.shape[0],
                                                   ret_array.shape[1],
                                                   ret_array.shape[2],
                                                   1))

                temp = model_convolutional.predict(ret_array)

                prediction_convolutional = np.zeros((factor, factor, 3))
                prediction_convolutional[:, :, 0] = np.squeeze(temp[0, :, :], axis=(2,))
                prediction_convolutional[:, :, 1] = np.squeeze(temp[1, :, :], axis=(2,))
                prediction_convolutional[:, :, 2] = np.squeeze(temp[2, :, :], axis=(2,))

                prediction_convolutional *= 255
                prediction_convolutional = prediction_convolutional.reshape((factor, factor, frame.shape[2]))
                output_frame_convolutional[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :] = prediction_convolutional

        output_frame = output_frame.astype('uint8')
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        output_frame_convolutional = output_frame_convolutional.astype('uint8')
        output_frame_convolutional = cv2.cvtColor(output_frame_convolutional, cv2.COLOR_RGB2BGR)

        noisy_frame = noisy_frame.astype('uint8')
        noisy_frame = cv2.cvtColor(noisy_frame, cv2.COLOR_RGB2BGR)

        total_output = np.append(noisy_frame, output_frame, axis=1)
        total_output = np.append(total_output, frame, axis=1)

        total_output_convolutional = np.append(noisy_frame, output_frame_convolutional, axis=1)
        total_output_convolutional = np.append(total_output_convolutional, frame, axis=1)

        output = np.append(total_output, total_output_convolutional, axis=0)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", output.shape[1] // 2, output.shape[0] // 2)
        cv2.imshow('frame', output)

        out.write(output)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
