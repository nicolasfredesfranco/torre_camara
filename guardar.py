
import cv2
import numpy as np
import argparse
from numpy import linalg as LA
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np

import requests
from os import listdir
from os.path import isfile, isdir
import time


if __name__ == "__main__":
    '''
    corta parte de un video grabado
    '''
    parser = argparse.ArgumentParser(description='This program  tracking people.')
    #path del video a grabar
    parser.add_argument('--path', type=str, help='Path of videos', default='rtsp://admin:lamngen1234@172.17.70.71/live2.sdp')
    #parser.add_argument('--path', type=str, help='Path of videos', default='last_video.avi')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.path)
    #promedio_grafico = []
    count = 0
    frameHeight = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    vid_writer = cv2.VideoWriter('video_prueba_torre.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (frameWidth, frameHeight))

    
    while cap.isOpened():
        # print(str(count))
        # count += 1
        ret, imag = cap.read()
        # Si no hay frame se termina
        if imag is None:
            break
        #cv2.imshow('image1', imag)
        vid_writer.write(imag.astype(np.uint8))
        #print(np.min(np.array(list_texto)))
        #cv2.imshow('image', fondo)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        #arreglo para que el contador no se desborde
        '''
        if count < 1000:
            count += 1
        '''
        count += 1
    cap.release()
    vid_writer.release()
    #si se necesita revisar el filtro
    '''t = np.array(range(len(y)))
    s = np.array(y)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time', ylabel='out', title='salida filtro')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    #vid_writer.release()

    t = np.array(range(len(promedio_grafico)))
    s = np.array(promedio_grafico)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time', ylabel='out', title='salida promedio')
    ax.grid()

    fig.savefig("test_promedio.png")
    plt.show()'''
    # vid_writer.release()
