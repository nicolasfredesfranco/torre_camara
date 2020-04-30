import subprocess
import argparse
import time
import multiprocessing as mp
from threading import Thread, Timer
import cv2
import numpy as np
from scipy.stats import mode


# import our objects
from detect_digit import*
from connect import Connect2Server

# estado inicial
status = 'Detenido'
num_mode = 'disconnected'

'''
state: estados del programa
check_server: chequea si todo esta listo para correr reconocimiento
check_camara: chequea si la camara esta conectada
run: correr programa
'''
state = 'check_server'

# Flags
stop_send = False

# change path for test and developed
def sist_recg_fault(ip, area, queue):

    path = 'rtsp://admin:lamngen1234@' + ip + '/live2.sdp'
    #path = '/home/azucar/Work/Lamngen/torre/sist_detect_fails/videos/ultimo_video_torre.avi'
    #path = '/home/azucar/Work/Lamngen/torre/sist_detect_fails/videos/camIzquierda_ultimo_video_torre.avi'
    print(ip)

    cap = cv2.VideoCapture(path)

    global status
    global num_mode

    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vec_numbers = ['0'] * 20

    count = 0

    # para guardar video
    '''
    frameHeight = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_writer = cv2.VideoWriter('../videos/resultado_final.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (frameWidth, frameHeight))
    '''

    # para contar cuanto tiempo pasa en nada
    count_nada = 0

    digit_recognition = DigitRecognition(area=area)

    # para tomar data
    time_1 = time.time()
    data = []
    count_data = 0

    while cap.isOpened():
        ret, image = cap.read()

        # Si no hay frame se termina
        if image is None:
            break

        number, n_obj = digit_recognition.recognition(image)

        # señal de deteccion de numeros
        if n_obj > 0:
            x[0:24] = x[1:25]
            x[24] = 1
            count_nada = 0

        else:
            x[0:24] = x[1:25]
            x[24] = 0

            # sumar 1 a contador con nada
            count_nada += 1

        # pare ver tiempo
        print('number:', number)
        print('cant_obj', n_obj)
        time_new = time.time()
        data.append([n_obj, time_new - time_1])
        time_1 = time_new
        print(len(data))
        if len(data) == 1000:
            data = np.array(data)
            np.save(str(count_data), data)
            data = []
            count_data +=1

        # FILTRO
        new_y = x[-3] - x[-5] + 2.88958 * y[-1] - 3.21816 * y[-2] + 1.6928 * y[-3] - 0.370204 * y[-4]
        y[0:24] = y[1:25]
        y[24] = new_y

        if len(number) > 0:
            vec_numbers[0:len(vec_numbers) - 1] = vec_numbers[1:len(vec_numbers)]
            vec_numbers[len(vec_numbers) - 1] = number
            num_mode = mode(vec_numbers)[0].item()

        # STATE MACHINE
        if count > 103:
            arreglo = np.array(y)
            promedio = np.average(np.abs(arreglo))
            #print(promedio)
            # maquina de estados
            if status == 'Velocidad':
                if count_nada > 15:
                    status = "Detenido"
                elif promedio > 6.5:
                    status = 'Falla'
                    # desicion.append(1)
            elif status == 'Falla':
                if count_nada > 15:
                    status = "Detenido"
                elif promedio < 5:
                    status = 'Velocidad'
                    # desicion.append(0)
            elif status == 'Detenido':
                # arreglo para que la moda no se quede pegada
                vec_numbers = ['0'] * 20
                num_mode = ""
                if (count_nada == 0) & (promedio < 2) & (len(number)!=0):
                    status = "Velocidad"
                    # desicion.append(0)
                elif (count_nada == 0) & (promedio > 6.5):
                    status = "Falla"
                    # desicion.append(1)

            # descomentar para ver la imagen
            #cv2.putText(image, status + " " + num_mode, (int(image.shape[0] / 2), int(image.shape[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            #cv2.imshow("prueba", image)
            queue.put({'status': status,
                       'num_mode': num_mode}
                      )
            #print(status, num_mode)
            # descomentar para guardar
            # vid_writer.write(image)
            # print(number)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # arreglo para que el contador no se desborde
        if count < 1000:
            count += 1

        # count += 1
    cap.release()
    # vid_writer.release()


def send_to_server(id):
    global status
    global num_mode
    global server

    while True:
        time.sleep(2)
        server.send(num_mode, status, id)
        print(status, num_mode, id)


def check_camera(ip):
    ret = subprocess.call(["ping", "-c", "1", ip], stdout=subprocess.DEVNULL)
    return ret==0


def check_server():
    url_server = "http://femto.duckdns.org:8081/torre/oauth/token"
    ret = subprocess.call(["curl", url_server], stdout=subprocess.DEVNULL)
    return ret == 0

if __name__ == "__main__":
    '''
    corta parte de un video grabado
    '''
    parser = argparse.ArgumentParser(description='This program  recognize number en screen and fault.')
    # path del video a grabar
    parser.add_argument('--ip', type=str, help='Path of videos', default='192.168.0.102')

    # id machine
    parser.add_argument('--id', type=int, help='Id maquina a detectar', default=1)

    args = parser.parse_args()

    if args.id==1:
        area=1650
    elif args.id==2:
        area=2000

    # init server
    server = Connect2Server()

    # init queue for p_camera
    queue = mp.Queue(1)

    count_to_send = 0
    send_rate = 200
    # start hilo que envia al servido
    #t = Timer(3.0, send_to_server, args=(args.id,))
    #t.start()
    #cant_total = 0
    #data_enviado = list()
    while True:

        # reviza si la cola tiene datos
        if not queue.empty():
            #cant_total += 1
            data = queue.get()
            if data['status'] != status or data['num_mode'] != num_mode:
                status = data['status']
                num_mode = data['num_mode']
                server.send(num_mode, status, args.id)
                count_to_send = 0
                print(num_mode, status)
            elif (count_to_send + 1) % send_rate == 0:
                count_to_send = 0
                server.send(num_mode, status, args.id)
                print(num_mode, status)
            else:
                count_to_send += 1


            #print(status, num_mode)

        # maquina de estados
        if state == 'check_server':
            if check_server():
                state = 'check_camera'
            else:
                print('error connection server')

        elif state == 'check_camera':
            num_mode = 'disconnected'
            if check_camera(args.ip):
                state = 'run'
                p_camera = mp.Process(target=sist_recg_fault, args=(args.ip, area, queue))
                p_camera.start()
                t1 = time.time()
            else:
                print('error connection camera')

        elif state == 'run':
            if not p_camera.is_alive():
                state = 'check_server'
                p_camera.terminate()
                #dt = time.time() - t1
                #print(cant_total, dt)
                #print('data enviada:\n', data_enviado)
                #print(len(data_enviado)/dt, len(data_enviado))
                #break

        else:
            state = 'check_server'











