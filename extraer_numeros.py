
import cv2
import numpy as np
import argparse
from numpy import linalg as LA
import requests

from os import listdir
from os.path import isfile, isdir



if __name__ == "__main__":
    '''
    corta parte de un video grabado
    '''
    parser = argparse.ArgumentParser(description='This program  tracking people.')
    #path del video a grabar
    parser.add_argument('--path', type=str, help='Path of videos', default='pantalla.mp4')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.path)

    count = 0
    frameHeight = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    vid_writer = cv2.VideoWriter('video_salida.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (frameHeight, frameWidth))

    list_nums = []
    for i in range(10):
        list_nums.append(cv2.imread('numeros/' + str(i) + '.png'))

    #array_nums = np.array(list_nums)
    list_texto = [100, 100, 100]

    while cap.isOpened():
        # print(str(count))
        # count += 1
        ret, imag = cap.read()
        # Si no hay frame se termina
        if imag is None:
            break
        imag_rotate = cv2.rotate(imag, cv2.ROTATE_90_CLOCKWISE)
        fondo = np.zeros(imag_rotate.shape)
        #fondo[:,:,2] = imag_rotate[:,:,2]
        canal_blue = imag_rotate[:, :, 0]
        canal_green = imag_rotate[:, :, 1]
        canal_red = imag_rotate[:, :, 2]
        #si es mayor a 200 toma el valor de 1.0
        red_centrado = canal_red.astype(float) - 230
        harto_rojo = np.heaviside(red_centrado, 0.0)
        green_centrado = 2 - canal_green.astype(float)
        poco_verde = np.heaviside(green_centrado, 0.0)
        extracto = np.multiply(harto_rojo, poco_verde)*254
        #fondo[:, :, 2] = extracto
        kernel = np.ones((5, 5), np.uint8)
        extracto = cv2.dilate(extracto, kernel, iterations=2)
        kernel_1 = np.ones((3, 3), np.uint8)
        extracto = cv2.erode(extracto, kernel_1, iterations=2)
        #extracto = harto_rojo*254
        #print(x.shape)
        numeros_1C = np.array(extracto, dtype=np.uint8)
        fondo[:, :, 2] = numeros_1C

        retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(numeros_1C, 8, cv2.CV_32S, cv2.CCL_WU)
        #blobs = []

        # posicion por cada blob
        list_blobs = []
        for i in range(retval):
            # if (stats[i, cv2.CC_STAT_WIDTH] < stats[i, cv2.CC_STAT_HEIGHT]) & (stats[i, cv2.CC_STAT_AREA] > 10):
            if ((stats[i, cv2.CC_STAT_WIDTH] > 10) & (stats[i, cv2.CC_STAT_HEIGHT] > 10) & (stats[i, cv2.CC_STAT_AREA] > 50)& (stats[i, cv2.CC_STAT_AREA] < 5000)):
                xi = stats[i, cv2.CC_STAT_LEFT]
                yi = stats[i, cv2.CC_STAT_TOP]
                xf = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                yf = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                list_blobs.append([fondo[yi:yf, xi:xf], xi])
                cv2.rectangle(imag_rotate, (xi, yi), (xf, yf), (255, 255, 255), 1)

        list_dist = []
        list_detection = []
        for imag, xi in list_blobs:
            for n in range(10):
                #print(imag.shape)
                #print(list_nums[n].shape)
                numero = cv2.resize(list_nums[n], (imag.shape[1],imag.shape[0]), interpolation=cv2.INTER_AREA)
                matrix_rest = np.subtract(imag.astype(float), numero.astype(float))
                magnitud = LA.norm(matrix_rest)
                list_dist.append(magnitud)
            array_dist = np.array(list_dist)
            list_detection.append([np.argmin(array_dist), xi])
            list_dist = []

        if list_detection[0][1] < list_detection[1][1]:
            texto = str(list_detection[0][0]) + str(list_detection[1][0])
            numero_detectado = 10*list_detection[0][0] + list_detection[1][0]
        else:
            texto = str(list_detection[1][0]) + str(list_detection[0][0])
            numero_detectado = 10 * list_detection[1][0] + list_detection[0][0]

        list_texto[0] = list_texto[1]
        list_texto[1] = list_texto[2]
        list_texto[2] = numero_detectado
        cv2.putText(imag_rotate, texto, (int(imag.shape[1]/2)+300, int(imag.shape[0]/2)+500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 3)
        cv2.imshow('image1', imag_rotate)
        vid_writer.write(imag_rotate.astype(np.uint8))

        data = {'status': 'Velocidad',
                'value': numero_detectado.item()}
        requests.post('http://0.0.0.0:5000', json=data)
        #print(np.min(np.array(list_texto)))
        #cv2.imshow('image', fondo)

        if cv2.waitKey(50) & 0xFF == 27:
            break
        count += 1
    cap.release()
    vid_writer.release()
