
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
    #parser.add_argument('--path', type=str, help='Path of videos', default='rtsp://admin:lamngen1234@172.17.70.71/live2.sdp')
    parser.add_argument('--path', type=str, help='Path of videos', default='last_video.avi')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.path)
    #promedio_grafico = []
    init = 80
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    frameHeight = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    #vid_writer = cv2.VideoWriter('video_salida_torre.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (frameWidth, frameHeight))

    list_nums = []
    for i in range(10):
        list_nums.append(cv2.imread('dataset/' + str(i) + '.png'))

    #array_nums = np.array(list_nums)
    list_texto = [100, 100, 100]
    contador = 0

    while cap.isOpened():
        # print(str(count))
        # count += 1
        ret, imag = cap.read()
        # Si no hay frame se termina
        if imag is None:
            break
        time_1 = time.time()
        fondo = np.zeros(imag.shape, dtype=np.uint8)
        canal_blue = imag[:, :, 0]
        canal_green = imag[:, :, 1]
        canal_red = imag[:, :, 2]
        #si es mayor a 200 toma el valor de 1.0
        red_centrado = canal_red.astype(float) - 130
        harto_rojo = np.heaviside(red_centrado, 0.0)
        green_centrado = 80 - canal_green.astype(float)
        poco_verde = np.heaviside(green_centrado, 0.0)
        extracto = np.multiply(harto_rojo, poco_verde)*254

        #si se quiere revisar como funciona la mascara descomentar y comentar el imshow final
        #cv2.imshow('image', harto_rojo*254)
        #tambien puede ser necesario revisar las dilataciones y erosiones

        kernel = np.ones((4, 2), np.uint8)
        extracto = cv2.dilate(extracto, kernel, iterations=1)
        kernel_1 = np.ones((2, 1), np.uint8)
        extracto = cv2.erode(extracto, kernel_1, iterations=1)

        extracto_u8 = np.array(extracto, dtype=np.uint8)
        fondo[:, :, 2] = extracto_u8

        retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(extracto_u8, 8, cv2.CV_32S, cv2.CCL_WU)
        #blobs = []

        # posicion por cada blob
        list_blobs = []
        for i in range(retval):
            # if (stats[i, cv2.CC_STAT_WIDTH] < stats[i, cv2.CC_STAT_HEIGHT]) & (stats[i, cv2.CC_STAT_AREA] > 10):
            if ((stats[i, cv2.CC_STAT_WIDTH] > 2) & (stats[i, cv2.CC_STAT_HEIGHT] > 10) & (stats[i, cv2.CC_STAT_AREA] > 20)& (stats[i, cv2.CC_STAT_AREA] < 5000)):
                xi = stats[i, cv2.CC_STAT_LEFT]
                yi = stats[i, cv2.CC_STAT_TOP]
                xf = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                yf = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                list_blobs.append([fondo[yi:yf, xi:xf, :], xi])
                #si quieres guardar un nuevo data set de numeros
                cv2.imwrite("data_new/prueba_" + str(contador) + ".png", fondo[yi:yf, xi:xf, :])
                contador += 1
                cv2.rectangle(fondo, (xi, yi), (xf, yf), (255, 255, 255), 1)

        list_dist = []
        list_detection = []
        for imag_i, xi in list_blobs:
            for n in range(10):
                #print(imag.shape)
                #print(list_nums[n].shape)
                numero = cv2.resize(list_nums[n], (imag_i.shape[1],imag_i.shape[0]), interpolation=cv2.INTER_AREA)
                matrix_rest = np.subtract(imag_i.astype(float), numero.astype(float))
                magnitud = LA.norm(matrix_rest)
                list_dist.append(magnitud)
            array_dist = np.array(list_dist)
            list_detection.append(np.array([np.argmin(array_dist), xi]))
            list_dist = []
        
        if (len(list_detection) > 0):
            array_detection =  np.array(list_detection).reshape(len(list_detection), 2)
            orden = np.argsort(array_detection[:,1])
            numero_final = 0
            for i in range(len(list_detection)):
                numero_final += array_detection[[orden[i]],0][0]*pow(10, len(list_detection) - i - 1)
                #print(1)
            x[0:24] = x[1:25]
            x[24] = 1
            #print(x)
            #print(numero_final)
        else:
            x[0:24] = x[1:25]
            x[24] = 0
            #print('Nada')

        new_y = x[-3] - x[-5] + 2.88958*y[-1] - 3.21816*y[-2] + 1.6928*y[-3] - 0.370204*y[-4]
        y[0:24] = y[1:25]
        y[24] = new_y
        #print(type(numero_final))
        #no considera el principio del video
        #print((time.time() - time_1)*1000)
        if (count > 103):
            arreglo = np.array(y)
            promedio = np.average(np.abs(arreglo))
            if (promedio > 3):
                cv2.putText(imag, 'parpadea', (int(imag.shape[0] / 2), int(imag.shape[1] / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Do post requests

                data = {'status': 'Falla',
                        'value': numero_final.item(),
                        'id': 2}
                requests.post('http://0.0.0.0:5000', json=data)

            else:
                cv2.putText(imag, 'no parpadea', (int(imag.shape[0] / 2), int(imag.shape[1] / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Do post requests
                data = {'status': 'Velocidad',
                        'value': numero_final.item(),
                        'id': 2}
                requests.post('http://0.0.0.0:5000', json=data)

            #promedio_grafico.append(promedio)
            init += 1
        print((time.time() - time_1)*1000)
        #print(new_y)
        cv2.imshow('image1', imag)
        #vid_writer.write(imag.astype(np.uint8))
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
