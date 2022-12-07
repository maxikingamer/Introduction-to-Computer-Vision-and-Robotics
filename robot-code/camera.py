import os
import cv2
import subprocess
import threading
import numpy as np
import yaml
from rich import print


class Camera:
    def __init__(self) -> None:

        self.camera_matrix, self.dist_coeffs = self.read_camera_intrinsics()

        self.IM_WIDTH = 848 # better FOV than 640
        self.IM_HEIGHT = 480

        device = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0"

        subprocess.call("v4l2-ctl -d " + device + " -c exposure_auto=1 -c exposure_auto_priority=0 -c exposure_absolute=200", shell=True)

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE , 5)

        self.cap.set(cv2.CAP_PROP_FPS, 30)
        subprocess.call("v4l2-ctl -c power_line_frequency=1", shell=True)

        self.ret = None
        self.frame = None
        self.is_running = True

        # warm up
        self.ret , self.frame = self.cap.read()

        # we run the camera read frames in a thread to always get the latest frame
        self.thread1 = threading.Thread(target=self.worker_thread)
        self.thread1.start()

    def worker_thread(self):
        while self.is_running:
            self.ret , self.frame = self.cap.read()

    def read(self):
        return self.ret, cv2.flip(self.frame, -1)

    def close(self):
        self.is_running = False
        self.cap.release()

    @staticmethod
    def read_camera_intrinsics(camera_intrinsics_file="./camera_intrinsics.yml"):

        with open(camera_intrinsics_file, "r") as stream:
            try:
                camera_intrinsics = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        camera_matrix = np.array(camera_intrinsics["camera_matrix"]["data"]).reshape((camera_intrinsics["camera_matrix"]["rows"], camera_intrinsics["camera_matrix"]["cols"]))

        dist_coeffs = np.array(camera_intrinsics["distortion_coefficients"]["data"]).reshape((camera_intrinsics["distortion_coefficients"]["rows"], camera_intrinsics["distortion_coefficients"]["cols"]))

        return camera_matrix, dist_coeffs