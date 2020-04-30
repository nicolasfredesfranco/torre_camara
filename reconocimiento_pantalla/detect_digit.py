from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import time



class DigitRecognition:
    def __init__(self, area):

        # define the dictionary of digit segments so we can identify
        self.DIGITS_LOOKUP = {(1, 1, 1, 0, 1, 1, 1): 0,
                              (0, 0, 1, 0, 0, 1, 0): 1,
                              (1, 0, 1, 1, 1, 0, 1): 2,
                              (1, 0, 1, 1, 0, 1, 1): 3,
                              (0, 1, 1, 1, 0, 1, 0): 4,
                              (1, 1, 0, 1, 0, 1, 1): 5,
                              (1, 1, 0, 1, 1, 1, 1): 6,
                              (1, 0, 1, 0, 0, 1, 0): 7,
                              (1, 1, 1, 1, 1, 1, 1): 8,
                              (1, 1, 1, 1, 0, 1, 1): 9
                              }
        self.area = area

    def found_screen(self, image):
        # pre-process the image by resizing it, converting it to
        # graycale, blurring it, and computing an edge map

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)

        # find contours in the edge map, then sort them by their
        # size in descending order
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None
        # loop over the contours
        for c in cnts:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if the contour has four vertices, then we have found
            # the thermostat display
            if len(approx) == 4:
                displayCnt = approx
                break

        # extract the thermostat display, apply a perspective transform
        # to it
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        output = four_point_transform(image, displayCnt.reshape(4, 2))

        return warped, output

    def detect_cnts_number(self, image):

        out_b, out_g, out_r = cv2.split(image)
        ret, thresh_g = cv2.threshold(out_g, 150, 255, cv2.THRESH_BINARY_INV)
        ret, thresh_r = cv2.threshold(out_r, 200, 255, cv2.THRESH_BINARY)

        thresh = cv2.bitwise_and(thresh_r, thresh_g)

        # se dilata y filtra para unificar el numero
        kernel = np.ones((4, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        kernel = np.ones((2, 1), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            # img = cv2.drawContours(output, [c], -1, (0,255,0), 3)
            (x, y, w, h) = cv2.boundingRect(c)
            # print(x, y, w, h)
            # if the contour is sufficiently large, it must be a digit
            if w >= 9 and (h >= 40 and h <= 70):
                digitCnts.append(c)

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        try:
            digitCnts = contours.sort_contours(digitCnts,
                                               method="left-to-right")[0]
        except:
            pass
        return digitCnts, thresh

    def recognize_digit(self, digitCnts, thresh):

        digits = []
        # loop over each of the digits
        #cv2.imshow('hola', thresh)
        for c in digitCnts:
            #print(c.shape)

            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            # sol parche
            area = w*h
            if area < self.area:
                w_rec = int(self.area/h)
                w_aug = w_rec - w
                x -= w_aug
                w = w_rec
            roi = thresh[y:y + h, x:x + w]
            #cv2.imshow('roi', roi)
            #print('area:', area)


            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.26), int(roiH * 0.24))
            (dW_b, dH_b) = (int(roiW * 0.4), int(roiH * 0.24))
            dHC = int(roiH * 0.05)

            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)),  # top-left
                ((w - dW, 0), (w, h // 2)),  # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),  # bottom-left
                ((w - dW_b, h // 2), (w, h)),  # bottom-right
                ((0, h - dH), (w, h))  # bottom
            ]
            on = [0] * len(segments)

            threshold_area_segments = [0.5, 0.4, 0.5, 0.5, 0.5, 0.4, 0.5]
            #areas = []

            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):

                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                #cv2.imshow('rec', segROI)
                #cv2.waitKey(0)


                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) >= threshold_area_segments[i]:
                    on[i] = 1

                #areas.append(round(total/area, 2))

            # lookup the digit and draw it on the image
            on = tuple(on)

            #print(on)
            #print(areas)

            try:
                digit = self.DIGITS_LOOKUP[on]
            except:
                # caso contrario distancia hamming
                dist = 5
                digit = None
                for i in self.DIGITS_LOOKUP.keys():
                    dist_aux = np.sum(np.abs(np.array(on) - np.array(i)))
                    if dist_aux <= dist:
                        dist = dist_aux
                        digit = self.DIGITS_LOOKUP[i]

            digits.append(digit)

            #print(digits)
        return digits

    def recognition(self, image):
        image = imutils.resize(image, height=500)
        warped, output = self.found_screen(image)
        digitCnts, thresh = self.detect_cnts_number(output)
        digits = self.recognize_digit(digitCnts, thresh)
        n_obj = self.cant_obj(thresh, min_width=2, min_height=10, min_area=250, max_area=5000)

        number = ''
        for digit in digits:
            number += str(digit)

        #cv2.imshow("thresh", thresh)

        return number, n_obj

    @staticmethod
    def cant_obj(image, min_width=1, min_height=1, min_area=10, max_area=1700):
        '''
        cuenta la cantidad de objetos en una imagen basado en componentes conectados
        '''
        retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(image, 8, cv2.CV_32S, cv2.CCL_WU)
        # blobs = []

        # posicion por cada blob
        list_blobs = []
        count = 0
        for i in range(retval):
            # area = stats[i, cv2.CC_STAT_AREA]
            # print(area)
            if ((stats[i, cv2.CC_STAT_WIDTH] > min_width) & (stats[i, cv2.CC_STAT_HEIGHT] > min_height) & (
                    stats[i, cv2.CC_STAT_AREA] > min_area) & (stats[i, cv2.CC_STAT_AREA] < max_area)):
                count += 1
                '''
                xi = stats[i, cv2.CC_STAT_LEFT]
                yi = stats[i, cv2.CC_STAT_TOP]
                xf = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                yf = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                '''
                # cv2.imshow('imag', imagen)
        return count


def extraer_mascara(imag):
    '''
    en base a threshold para cada calar extrae un mascara de color rojo de ka imagen y luego la difumina para que se vea un numero unido
    '''
    fondo = np.zeros(imag.shape, dtype=np.uint8)
    canal_blue = imag[:, :, 0]
    canal_green = imag[:, :, 1]
    canal_red = imag[:, :, 2]
    # threshold por color
    red_centrado = canal_red.astype(float) - 130
    harto_rojo = np.heaviside(red_centrado, 0.0)
    green_centrado = 80 - canal_green.astype(float)
    poco_verde = np.heaviside(green_centrado, 0.0)
    # se consideran ambos threshold multiplicandolos
    extracto = np.multiply(harto_rojo, poco_verde) * 254

    # extracto_2 =  cv2.medianBlur(np.array(extracto, dtype=np.uint8), 1)
    # si se quiere revisar como funciona la mascara descomentar y comentar el imshow final
    # cv2.imshow('image_0', extracto_2)
    # tambien puede ser necesario revisar las dilataciones y erosiones

    # se dilata y filtra para unificar el numero
    kernel = np.ones((4, 2), np.uint8)
    extracto = cv2.dilate(extracto, kernel, iterations=2)
    # cv2.imshow('image_dilate', extracto)
    kernel_1 = np.ones((2, 1), np.uint8)
    extracto = cv2.erode(extracto, kernel_1, iterations=1)
    # kernel_2 = np.ones((1, 2), np.uint8)
    # extracto = cv2.dilate(extracto, kernel_2, iterations=1)
    # cv2.imshow('image', extracto_2)

    extracto_u8 = np.array(extracto, dtype=np.uint8)
    fondo[:, :, 2] = extracto_u8

    return fondo


def main():
    import argparse

    parser = argparse.ArgumentParser(description='This program  tracking people.')
    # path del video a grabar
    parser.add_argument('--path', type=str, help='Path of videos',
                        default='camIzquierda_ultimo_video_torre.avi') #'rtsp://admin:lamngen1234@192.168.0.102/live2.sdp')
    # camIzquierda_ultimo_video_torre.avi
    args = parser.parse_args()

    digit_recognition = DigitRecognition(area=1500)

    ip = '192.168.1.38' 
    path = 'rtsp://admin:lamngen1234@' + ip + '/live2.sdp'

    cap = cv2.VideoCapture(path)

    #frameHeight = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frameWidth = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #vid_writer = cv2.VideoWriter('detection_camizq.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20,
    #                             (frameWidth, frameHeight))

    while cap.isOpened():
        ret, image = cap.read()

        if image is None:
            break

        number = digit_recognition.recognition(image)

        cv2.putText(image, str(number), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        #vid_writer.write(image.astype(np.uint8))
        cv2.imshow('detect', image)

        #cv2.imshow('thresh', thresh)
        #cv2.imshow('output', output)

        #_, y_val = model.predict(thresh)
        #print('model:', y_val)
        #print(pytesseract.image_to_string(thresh))

        if cv2.waitKey(50) & 0xFF == 27:
            break
    cap.release()
    #vid_writer.release()


if __name__ == "__main__":
    main()

''' 
red_centrado = out_r.astype(float) - 200
harto_rojo = np.heaviside(red_centrado, 0.0)
green_centrado = 150 - out_g.astype(float)
poco_verde = np.heaviside(green_centrado, 0.0)
# se consideran ambos threshold multiplicandolos
extracto = np.multiply(harto_rojo, poco_verde) * 254
'''
'''
y_total = np.array(y_total)
p_total = np.array(p_total)
desicion = np.array(desicion)
np.save('y_total_2.npy', y_total)
np.save('p_total_2.npy', p_total)
np.save('desicion_2.npy', desicion)


#si se necesita revisar el filtro

t = np.array(range(len(y_total)))
s = np.array(y_total)
fig, ax = plt.subplots()


ax.set(xlabel='time', ylabel='out', title='salida filtro')
ax.grid()

fig.savefig("test_2.png")
plt.show()

#vid_writer.release()
'''
''' 
t = np.array(range(len(promedio_grafico)))
s = np.array(promedio_grafico)
fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time', ylabel='out', title='salida promedio')
ax.grid()

fig.savefig("test_promedio.png")
plt.show()
'''
