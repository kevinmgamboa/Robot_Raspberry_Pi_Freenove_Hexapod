# -*- coding: utf-8 -*-
import io
import math
import copy
import socket
import struct
import threading
from PID import *
from Face import *
import numpy as np
from Thread import *
import multiprocessing
from PIL import Image, ImageDraw
from Command import COMMAND as cmd
import tensorflow as tf


class Client:
    def __init__(self):
        self.face = Face()
        self.pid = Incremental_PID(1, 0, 0.0025)
        self.tcp_flag = False
        self.video_flag = True
        self.fece_id = False
        self.fece_recognition_flag = False
        self.image = ''

    # Initialize model fro object detection
    def init_object_detector(self):
        # Load & initialize model
        detector = ObjectDetectorLite(model_path=model_path, label_path=label_path)

    # Add the new function here.
    def tflite_object_detection(self, image):
        """
        Steps
        1. Reshapes image to appropriate model input dimension
        2. Apply the model to produce predictions
        3. Takes the output and added it back to the video frame
        """

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="detect.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess your image or frame here.
        # This code assumes your model expects an image of a certain size and range.
        preprocessed_image = self.preprocess(image)  # You'll need to implement this.

        # Run the model's interpreter.
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        interpreter.invoke()

        # Extract the output and postprocess it.
        boxes = interpreter.get_tensor(output_details[0]['index'])
        class_labels = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])

        return boxes, class_labels, scores, num

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
                jpg = self.connection.read(leng[0])
                if self.is_valid_image_4_bytes(jpg):
                    if self.video_flag:
                        # ----------------------------------------------------------------------------------------------
                        #                              Accessing Robot Video Camera
                        # ----------------------------------------------------------------------------------------------
                        # gets the image that appears in video
                        self.image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        # image is an ndarray (300,400,3)
                        # ----------------------------------------------------------------------------------------------
                        # # Add a call to the new function here.
                        # boxes, class_labels, scores, num = self.tflite_object_detection(self.image)
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
