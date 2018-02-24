import numpy as np
from cv2 import cv2

from keras.models import load_model
from tqdm import tqdm

from location import new_girl_location

model = load_model('one_to_one.h5')
factor = 10

cap = cv2.VideoCapture(new_girl_location)

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
        print('Testing currently working on ' + str(i))
        if i < 100:
            print('Continue: ' + str(i))
            continue

        if i == 120:
            break

        # Display the resulting frame
        output_frame = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        array = []
        for j in tqdm(range(0, frame.shape[0] // factor)):
            temp_array = []
            for k in range(0, frame.shape[1] // factor):
                temp = np.array(frame)
                temp = temp[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :]
                temp = temp.reshape((1, factor * factor * frame.shape[2])).astype('float32')
                temp /= 255

                temp = model.predict([temp])

                temp *= 255
                temp = temp.reshape((factor, factor, frame.shape[2]))
                output_frame[j * factor: (j + 1) * factor, k * factor: (k + 1) * factor, :] = temp

        output_frame = output_frame.astype('uint8')
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        total_output = np.append(frame, output_frame, axis=1)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", total_output.shape[1] // 2, total_output.shape[0] // 2)
        cv2.imshow('frame', total_output)

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
