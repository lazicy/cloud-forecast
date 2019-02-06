import numpy as np
import cv2
from lines import *
from digits import *
from helpers import *
from cnn import *
import time
from test import *

import matplotlib
import matplotlib.pyplot as plt

videos = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi',
          'video-5.avi', 'video-6.avi', 'video-7.avi', 'video-8.avi',  'video-9.avi']

sum_list = []



# model = train_cnn()

model = load_trained_model('digit_trained.model')

i = 0


def go_go_power_rangers(video):
    blueP1, blueP2, greenP1, greenP2 = process_lines(video)
    process_video(video, blueP1, blueP2, greenP1, greenP2)

def process_lines(video):

    cap_calc = cv2.VideoCapture(video)

    i = 0

    while cap_calc.isOpened():

        ret, frame_calc = cap_calc.read()

        work_with_lines(frame_calc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if i > 20:
            blueP1, blueP2 = calculate_definite_lines('blue')
            greenP1, greenP2 = calculate_definite_lines('green')

            print('blue line: ', blueP1, blueP2)
            print('green line: ', greenP1, greenP2)

            cv2.line(frame_calc, (blueP1[0], blueP1[1]), (blueP2[0], blueP2[1]), (255, 0, 0), 3)
            cv2.line(frame_calc, (greenP1[0], greenP1[1]), (greenP2[0], greenP2[1]), (150, 50, 204), 3)

            return blueP1, blueP2, greenP1, greenP2

        i = i + 1

def process_video(video, blueP1, blueP2, greenP1, greenP2):

    cool_down_blue = 0;
    cool_down_green = 0;

    sum = 0;

    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break;

        # time.sleep(0.01)

        only_numbers = separate_digits(frame)

        for_roi = prepare_for_roi(only_numbers)

        only_numbers = invert(only_numbers)

        frame_contours, numbers, br_coords = select_roi(frame.copy(), for_roi, only_numbers)

        if (cool_down_blue == 0):

            for i in range(len(br_coords)):

                d_blue = check_distance_from_line(br_coords[i], blueP1, blueP2)
                d_green = check_distance_from_line(br_coords[i], greenP1, greenP2)

                crossed_blue = []
                crossed_green = []

                if d_blue < 2.9 and br_coords[i][0] >= blueP1[0] and br_coords[i][0] <= blueP2[0] and cool_down_blue == 0:
                    print('crossed blue')
                    print('distens blue: ', d_blue)
                    crossed_blue.append(invert(erode_small(numbers[i])))
                    blue_normalized = tf.keras.utils.normalize(crossed_blue)
                    predictions = model.predict(blue_normalized)

                    predict_num = np.argmax(predictions[0])
                    print(np.argmax(predictions[0]))

                    sum = sum + predict_num
                    print('sum = ', sum)

                    # plt.imshow(blue_normalized[0].reshape(28, 28), cmap='Greys')
                    # plt.show()


                    cool_down_blue = 7;


                if d_green < 2.9 and br_coords[i][0] >= greenP1[0] and br_coords[i][0] <= greenP2[
                    0] and cool_down_green == 0:
                    print('crossed green')
                    print('distens green: ', d_green)
                    crossed_green.append(invert(erode_small(numbers[i])))
                    green_normalized =  tf.keras.utils.normalize(crossed_green)
                    predictions = model.predict(green_normalized)
                    print(np.argmax(predictions[0]))

                    predict_num = np.argmax(predictions[0])
                    print(predictions[0])

                    sum = sum - predict_num
                    print('sum = ', sum)

                    # plt.imshow(green_normalized[0].reshape(28, 28), cmap='Greys')
                    # plt.show()

                    cool_down_green = 7;

        if cool_down_blue != 0:
            cool_down_blue -= 1

        if cool_down_green != 0:
            cool_down_green -= 1

        # cv2.imshow('countours frame', frame_contours)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    sum_list.append(sum)

    cap.release()

def write_to_file():


    f = open('out.txt', 'w')
    text = 'RA 78/2015 Milan Lazic\nfile\tsum\n'

    for i in range(10):
        text += videos[i] + '\t' + str(sum_list[i]) + '\n'


    f.write(text)
    f.close()




for video in videos:

    go_go_power_rangers(video)


write_to_file()


do_the_test()




'''
numbers = crossed_blue;

numbers = invert_data(numbers)

normalized = tf.keras.utils.normalize(numbers, axis=1)
predictions = model.predict(normalized)



for i in range(len(normalized)):

    plt.imshow(normalized[i], cmap=plt.cm.binary)
    print(np.argmax(predictions[i]))
    plt.show()
    



#
# print(selected_frame)

'''

cv2.destroyAllWindows()
print('exited')



