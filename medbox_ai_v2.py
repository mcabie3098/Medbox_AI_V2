import os
import argparse
import cv2
import numpy as np
import sys
import glob
import http.client
import importlib.util
import time
import git
import requests_cache
import urllib
import requests 
from urllib.request import urlopen
from threading import Thread
import threading 
from collections import Counter 
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
global str
import pandas as pd
from tkinter import *
from tkinter import messagebox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(614, 544)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("medbox/art/prescript.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbl_Title = QtWidgets.QLabel(self.centralwidget)
        self.lbl_Title.setGeometry(QtCore.QRect(10, 40, 581, 71))
        self.lbl_Title.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_Title.setText("")
        self.lbl_Title.setPixmap(QtGui.QPixmap("medbox/art/medbox.ico"))
        self.lbl_Title.setScaledContents(False)
        self.lbl_Title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_Title.setObjectName("lbl_Title")
        self.lbl_Run = QtWidgets.QLabel(self.centralwidget)
        self.lbl_Run.setGeometry(QtCore.QRect(50, 180, 211, 31))
        self.lbl_Run.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_Run.setText("")
        self.lbl_Run.setPixmap(QtGui.QPixmap("medbox/art/run.ico"))
        self.lbl_Run.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_Run.setObjectName("lbl_Run")
        self.btn_Run = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Run.setGeometry(QtCore.QRect(80, 220, 161, 131))
        self.btn_Run.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("medbox/art/play.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Run.setIcon(icon1)
        self.btn_Run.setIconSize(QtCore.QSize(150, 150))
        self.btn_Run.setFlat(True)
        self.btn_Run.setObjectName("btn_Run")
        self.lbl_ImpImg = QtWidgets.QLabel(self.centralwidget)
        self.lbl_ImpImg.setGeometry(QtCore.QRect(360, 170, 161, 41))
        self.lbl_ImpImg.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_ImpImg.setText("")
        self.lbl_ImpImg.setPixmap(QtGui.QPixmap("medbox/art/import.ico"))
        self.lbl_ImpImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_ImpImg.setObjectName("lbl_ImpImg")
        self.btn_Image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Image.setGeometry(QtCore.QRect(360, 220, 161, 131))
        self.btn_Image.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("medbox/art/image.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Image.setIcon(icon2)
        self.btn_Image.setIconSize(QtCore.QSize(150, 150))
        self.btn_Image.setFlat(True)
        self.btn_Image.setObjectName("btn_Image")
        self.img_Rx = QtWidgets.QLabel(self.centralwidget)
        self.img_Rx.setGeometry(QtCore.QRect(500, 20, 81, 91))
        self.img_Rx.setText("")
        self.img_Rx.setPixmap(QtGui.QPixmap("medbox/art/prescript.ico"))
        self.img_Rx.setScaledContents(True)
        self.img_Rx.setObjectName("img_Rx")
        self.lbl_aboutImg = QtWidgets.QLabel(self.centralwidget)
        self.lbl_aboutImg.setGeometry(QtCore.QRect(310, 370, 261, 41))
        self.lbl_aboutImg.setText("")
        self.lbl_aboutImg.setPixmap(QtGui.QPixmap("medbox/art/about_img.ico"))
        self.lbl_aboutImg.setObjectName("lbl_aboutImg")
        self.lbl_aboutRun1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_aboutRun1.setGeometry(QtCore.QRect(40, 370, 221, 31))
        self.lbl_aboutRun1.setText("")
        self.lbl_aboutRun1.setPixmap(QtGui.QPixmap("medbox/art/about_run1.ico"))
        self.lbl_aboutRun1.setObjectName("lbl_aboutRun1")
        self.lbl_aboutRun2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_aboutRun2.setGeometry(QtCore.QRect(50, 390, 191, 31))
        self.lbl_aboutRun2.setText("")
        self.lbl_aboutRun2.setPixmap(QtGui.QPixmap("medbox/art/about_run2.ico"))
        self.lbl_aboutRun2.setObjectName("lbl_aboutRun2")
        self.lbl_Check = QtWidgets.QLabel(self.centralwidget)
        self.lbl_Check.setGeometry(QtCore.QRect(240, 470, 111, 21))
        self.lbl_Check.setText("")
        self.lbl_Check.setPixmap(QtGui.QPixmap("medbox/art/check.ico"))
        self.lbl_Check.setScaledContents(True)
        self.lbl_Check.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_Check.setObjectName("lbl_Check")
        self.btn_Update = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Update.setGeometry(QtCore.QRect(350, 470, 21, 21))
        self.btn_Update.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("medbox/art/update.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Update.setIcon(icon3)
        self.btn_Update.setShortcut("")
        self.btn_Update.setFlat(True)
        self.btn_Update.setObjectName("btn_Update")
        self.btn_Webcam = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Webcam.setGeometry(QtCore.QRect(580, 470, 21, 21))
        self.btn_Webcam.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("art/webcam.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Webcam.setIcon(icon4)
        self.btn_Webcam.setFlat(True)
        self.btn_Webcam.setObjectName("btn_Webcam")
        self.btn_version2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_version2.setGeometry(QtCore.QRect(240, 110, 101, 21))
        self.btn_version2.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("medbox/art/version2.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_version2.setIcon(icon5)
        self.btn_version2.setIconSize(QtCore.QSize(100, 100))
        self.btn_version2.setAutoDefault(False)
        self.btn_version2.setFlat(True)
        self.btn_version2.setObjectName("btn_version2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 614, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Medbox AI"))

        self.btn_Webcam.clicked.connect(self.pushButton_Webcam)
        self.btn_Image.clicked.connect(self.pushButton_Image)
        self.btn_Run.clicked.connect(self.pushButton_Run)
        self.btn_Update.clicked.connect(self.pushButton_Update)
        self.btn_version2.clicked.connect(self.pushButton_ver2)
        
#Version 2
    def pushButton_ver2(self):
        # Details
        window=Tk()
        lbl=Label(window, text='Medbox AI v0.0.2', fg='blue', font=("Calibri", 18))
        lbl.place(x=65, y=35)
        lbl2=Label(window, text='What is new?', fg='black', font=("Helvetica", 12))
        lbl2.place(x=65, y=115)
        lbl3=Label(window, text='-System can now detect 5 new presciptions amoxicillin, doxycyline, clindamycin, ciproflaxin and cephalexin', fg='black', font=("Helvetica", 12))
        lbl3.place(x=65, y=155)
        lbl4=Label(window, text='-Accuracy is now up to 98.7% from 96.7%', fg='black', font=("Helvetica", 12))
        lbl4.place(x=65, y=185)
        lbl5=Label(window, text='-Detection speed increased slightly', fg='black', font=("Helvetica", 12))
        lbl5.place(x=65, y=215)
        window.title('Medbox AI')
        window.geometry("1200x500+50+50")
        window.mainloop()

#Update
    def pushButton_Update(self):
        # PLEASE WAIT
        window=Tk()
        lbl=Label(window, text='Please wait', fg='black', font=("Helvetica", 12))
        lbl.place(x=65, y=35)
        window.title('Medbox AI')
        window.geometry("400x100+10+10")
        window.after(2500, lambda: window.destroy())
        window.mainloop()
        
        def is_internet():
            try:
                urlopen('https://www.google.com', timeout=1)
                return True
            except urllib.error.URLError as Error:
                return False

        requests_cache.install_cache('demo_cache')
        if is_internet():
            print('may net')
            url = 'https://github.com/mcabie3098/Medbox_AI_V2'
            r = requests.get(url)
            print(r)
            page = r.status_code
           
            if page==200:
                # NO UPDATES YET
                window = Tk()
                window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
                window.withdraw()
                messagebox.showinfo("Medbox AI", "No Updates Available")
                window.deiconify()
                window.destroy()
                window.quit() 
                
            elif page==404:
                # UPDATE IS HERE!
                # NOTE: YOU HAVE TO RENAME GITHUB FOLDER EVERY TIME YOU UPDATE eg. Medbox_AIv2
                window = Tk()
                window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
                window.withdraw()                
                if messagebox.askyesno('Medbox AI', 'Update is now available! \n\nDo you want to update now?', icon='info') == True:
                    # DOWNLOADING
                    window=Tk()
                    lbl=Label(window, text='Downloading...', fg='black', font=("Helvetica", 12))
                    lbl.place(x=65, y=35)
                    window.title('Medbox AI')
                    window.geometry("400x100+10+10")
                    window.after(2500, lambda: window.destroy())
                    window.mainloop()
                    # YOU ARE NOW UP TO DATE
                    git.Git("").clone("https://github.com/mcabie3098/Medbox_AI_V3") # Future purpose Medbox_AIv2 if exist will Update
                    window = Tk()
                    window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
                    window.withdraw()
                    messagebox.showinfo("Medbox AI V3", "You are now up to date! \nThe file is in Medbox_AI_V3 folder")
                    sys.exit()

                else:
                    print('no')
                window.deiconify()
                window.destroy()
                window.quit()      
                
        else:
            #NO INTERNET
            window = Tk()
            window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
            window.withdraw()
            messagebox.showinfo("Medbox AI", "No Internet Connection")  
            window.deiconify()
            window.destroy()
            window.quit()

#Run
    def pushButton_Run(self):
        videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        result = True
        while(result):
            ret,frame = videoCaptureObject.read()
            cv2.imwrite("image.jpg",frame)
            result = False
        cv2.destroyAllWindows()
        self.open_dialog_box3()

    def open_dialog_box3(self):
        img_elec = 'image.jpg'

        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            default='medbox')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                            default=None)
        parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                            default=None)
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args = parser.parse_args()

        MODEL_NAME = args.modeldir
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        min_conf_threshold = float(args.threshold)
        use_TPU = args.edgetpu

        # Parse input image name and directory. 
        IM_NAME = args.image
        IM_DIR = args.imagedir

        # If both an image AND a folder are specified, throw an error
        if (IM_NAME and IM_DIR):
            print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
            sys.exit()

        # If neither an image or a folder are specified, default to using 'test1.jpg' for image name
        if (not IM_NAME and not IM_DIR):
            IM_NAME = img_elec

        # Import TensorFlow libraries
        # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tensorflow')
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'


        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Define path to images and grab all image filenames
        if IM_DIR:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
            images = glob.glob(PATH_TO_IMAGES + '/*')

        elif IM_NAME:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
            images = glob.glob(PATH_TO_IMAGES)

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5
        
        start_time = time.time()
        # Loop over every image and perform detection
        for image_path in images:

            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s' % (object_name) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # All the results have been drawn on the image, now display the image
            cv2.imshow('Q=quit', image)
            print("PROCESSING TIME: %s seconds" % (time.time() - start_time))

            # Press any key to continue to next image, or press 'q' to quit
            if cv2.waitKey(0) == ord('q'):
                break

        # Clean up
        os.remove("image.jpg")
        cv2.destroyAllWindows()
 
#Webcam
    def pushButton_Webcam(self):
        self.open_dialog_box2()
    def open_dialog_box2(self):
        print("Starting...")
        
        class VideoStream:
            """Camera object that controls video streaming from the Picamera"""
            def __init__(self,resolution=(640,480),framerate=30):
                # Initialize the PiCamera and the camera image stream
                self.stream = cv2.VideoCapture(0)
                ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                ret = self.stream.set(3,resolution[0])
                ret = self.stream.set(4,resolution[1])
                    
                # Read first frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

                # Variable to control when the camera is stopped
                self.stopped = False

            def start(self):
                # Start the thread that reads frames from the video stream
                Thread(target=self.update,args=()).start()
                return self

            def update(self):
                # Keep looping indefinitely until the thread is stopped
                while True:
                    # If the camera is stopped, stop the thread
                    if self.stopped:
                        # Close camera resources
                        self.stream.release()
                        return

                    # Otherwise, grab the next frame from the stream
                    (self.grabbed, self.frame) = self.stream.read()

            def read(self):
                # Return the most recent frame
                return self.frame

            def stop(self):
                # Indicate that the camera and thread should be stopped
                self.stopped = True

        # Define and parse input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            default='medbox')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='920x640')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args = parser.parse_args()

        MODEL_NAME = args.modeldir
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        min_conf_threshold = float(args.threshold)
        resW, resH = args.resolution.split('x')
        imW, imH = int(resW), int(resH)
        use_TPU = args.edgetpu

        # Import TensorFlow libraries
        # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tensorflow')
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        # Initialize video stream
        videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        time.sleep(1)

        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()
            start_time1 = time.time()

            # Grab frame from video stream
            frame1 = videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    print(label)
                    print("PROCESSING TIME: %s seconds" % (time.time() - start_time1))

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Q=quit', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()
        videostream.stop()

#Image
    def pushButton_Image(self):
        self.open_dialog_box()
    def open_dialog_box(self):
        img = QFileDialog.getOpenFileName()
        img_elec = img[0]

        # Define and parse input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            default='medbox')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                            default=None)
        parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                            default=None)
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args = parser.parse_args()

        MODEL_NAME = args.modeldir
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        min_conf_threshold = float(args.threshold)
        use_TPU = args.edgetpu

        # Parse input image name and directory. 
        IM_NAME = args.image
        IM_DIR = args.imagedir

        # If both an image AND a folder are specified, throw an error
        if (IM_NAME and IM_DIR):
            print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
            sys.exit()

        # If neither an image or a folder are specified, default to using 'test1.jpg' for image name
        if (not IM_NAME and not IM_DIR):
            IM_NAME = img_elec

        # Import TensorFlow libraries
        # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tensorflow')
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'


        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Define path to images and grab all image filenames
        if IM_DIR:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
            images = glob.glob(PATH_TO_IMAGES + '/*')

        elif IM_NAME:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
            images = glob.glob(PATH_TO_IMAGES)

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5
        
        start_time = time.time()
        # Loop over every image and perform detection
        for image_path in images:

            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                                     

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s' % (object_name) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # All the results have been drawn on the image, now display the image
            cv2.imshow('detected', image)
            print("PROCESSING TIME: %s seconds" % (time.time() - start_time))

            # Press any key to continue to next image, or press 'q' to quit
            if cv2.waitKey(0) == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())