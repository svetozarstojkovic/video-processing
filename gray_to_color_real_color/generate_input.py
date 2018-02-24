import os
from builtins import print

import numpy as np
from cv2 import cv2
from tqdm import tqdm

from location import new_girl_location

import matplotlib.pyplot as plt

factor = 10

input_size = 30
test_size = 10


def generate_input(video_path):
    print('Generate output')

    if not os.path.exists('input'):
        os.makedirs('input')

    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    i = 0
    input_array = []
    test_input_array = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            i = i + 1

            if i == (100 + input_size + test_size):
                break

            print('Currently working on ' + str(i))
            if i < 100:
                print('Continue: ' + str(i))
                continue

            # Display the resulting frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            array = []
            for j in tqdm(range(0, frame.shape[0] // factor)):
                temp_array = []
                for k in range(0, frame.shape[1] // factor):
                    temp = np.array(frame)
                    temp = temp[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor]
                    temp = temp.reshape(factor * factor).astype('float32')
                    temp /= 255

                    if temp_array == []:
                        temp_array = np.array([temp])
                    else:
                        temp_array = np.append(temp_array, [temp], axis=0)

                if array == []:
                    array = np.array(temp_array)
                else:
                    array = np.append(array, temp_array, axis=0)
            # cv2.imshow('Frame', frame)

            if i < (100 + input_size):
                # print('Filling train input data: '+str(i))
                if input_array == []:
                    input_array = array
                else:
                    input_array = np.append(input_array, array, axis=0)
            elif i < (100 + input_size + test_size):
                if test_input_array == []:
                    test_input_array = array
                else:
                    test_input_array = np.append(test_input_array, array, axis=0)


            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    np.save('input/input', input_array)
    np.save('input/input_test', test_input_array)


generate_input(video_path=new_girl_location)
