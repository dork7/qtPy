from detector import detect
import os
import os.path
from pathlib import Path
import cv2
from multiprocessing import Process, Manager, Queue
# from camera import generate_stream
from webcam import generate_stream
from PIL import Image
import time

if __name__ == "__main__":

    image = Image
    manager = Manager()
    imageList = Queue(maxsize=1)
    locsList = Queue(maxsize=1)
    predsList = manager.list()
    predsList2 = manager.list()
    predsTfList = Queue(maxsize=1)
    tempList = manager.list()
    predsImageList = Queue(maxsize=1)
    processes = ['p', 's']

    for i in range(len(processes)):
        if i == 0:
            worker = generate_stream
        else:
            worker = detect
        processes[i] = Process(target=worker, args=[imageList, locsList, predsList, predsList2,
                                                    predsTfList, tempList, predsImageList])
        processes[i].start()
    for process in processes:
        process.join()
