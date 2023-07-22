# -*- coding: utf-8 -*-
import io
# import math
# import copy
import socket
import struct
# import threading
from PID import *
from Face import *
import numpy as np
# from Thread import *
# import multiprocessing
from PIL import Image, ImageDraw
# from Command import COMMAND as cmd
# import tensorflow as tf

from myai.detector import ObjectDetectorLite
from myai.visualization_utils import draw_bounding_boxes_on_image_array
from myai.utils import reshape_image


class Client:
    def __init__(self):
        # initializes model object detection
        self.detector = ObjectDetectorLite(model_path='detect.tflite', label_path='labelmap.txt')
        self.face = Face()
        self.pid = Incremental_PID(1, 0, 0.0025)
        self.tcp_flag = False
        self.video_flag = True
        self.fece_id = False
        self.fece_recognition_flag = False
        self.image = ''

    # Add the new function here.
    def tflite_object_detection(self, ori_image, confidence):
        """
        Steps
        1. Reshapes image to appropriate model input dimension
        2. Apply the model to produce predictions
        3. Takes the output and added it back to the video frame
        """

        # reshapes image to lower dimension
        image = reshape_image(ori_image)
        # passes image into model
        boxes, scores, classes = self.detector.detect(image, confidence)
        # draw box into image
        if len(boxes) > 0:
            draw_bounding_boxes_on_image_array(image, boxes, display_str_list=classes)

    def turn_on_client(self, ip):
        self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(ip)

    def turn_off_client(self):
        try:
            self.client_socket.shutdown(2)
            self.client_socket1.shutdown(2)
            self.client_socket.close()
            self.client_socket1.close()
        except Exception as e:
            print(e)

    def is_valid_image_4_bytes(self, buf):
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:
            try:
                Image.open(io.BytesIO(buf)).verify()
            except:
                bValid = False
        return bValid

    def receiving_video(self, ip):
        try:
            self.client_socket.connect((ip, 8002))
            self.connection = self.client_socket.makefile('rb')
        except:
            # print ("command port connect failed")
            pass
        while True:
            try:
                stream_bytes = self.connection.read(4)
                leng = struct.unpack('<L', stream_bytes[:4])
                # reads image from robot
                jpg = self.connection.read(leng[0])
                if self.is_valid_image_4_bytes(jpg):
                    if self.video_flag:
                        # ----------------------------------------------------------------------------------------------
                        #                              Accessing Robot Video Camera
                        # ----------------------------------------------------------------------------------------------
                        # gets the image that appears in video
                        self.image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        # self.image is an ndarray (300,400,3)
                        # image received by the model is (300, 300, 3)
                        # ----------------------------------------------------------------------------------------------
                        # # Add a call to the new function here.
                        self.tflite_object_detection(self.image)
                        #
                        # # Process or display the results as needed.
                        # self.process_detections(boxes, class_labels, scores, num)  # You'll need to implement this.
                        # ----------------------------------------------------------------------------------------------
                        if self.fece_id == False and self.fece_recognition_flag:
                            self.face.face_detect(self.image)
                        self.video_flag = False
                        # ----------------------------------------------------------------------------------------------
            except BaseException as e:
                print(e)
                break

    def send_data(self, data):
        if self.tcp_flag:
            try:
                self.client_socket1.send(data.encode('utf-8'))
            except Exception as e:
                print(e)

    def receive_data(self):
        data = self.client_socket1.recv(1024).decode('utf-8')
        return data


if __name__ == '__main__':
    c = Client()
    c.face_recognition()
