import time
from picamera import *
from picamera.array import *
from datetime import datetime
import screeninfo
# import config_with_yaml as config
import cv2

def generate_stream(imageList,locsList,predsList,predsList2,predsTfList,tempList,predsImageList):
        # initialize the camera and grab a reference to the raw camera capture
        anterior = 0
        screen_id = 0
        screen = screeninfo.get_monitors()[screen_id]
        width, height = screen.width, screen.height
        print(width, height)

        camera = PiCamera()
        camera.framerate = 35
        camera.rotation = 90
        camera.hflip=True
        camera.resolution = (480, 640)
        raw_capt = PiRGBArray(camera, size=(480, 640))
        # allow the camera to warmup
        time.sleep(2)
        
        boxes=None
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2

        count=0
        last_detected = None
        pred_tf = None
        print('#####CAMERA STREAM STARTING#####')
        for frame in camera.capture_continuous(raw_capt, format="bgr", use_video_port=True):
            frame = frame.array
            imageList.put(frame)
            (targetStartX,targetEndX,targetStartY,targetEndY) = ( 150, 340, 140, 350 )
            cv2.rectangle(frame, (targetStartX, targetStartY), (targetEndX,targetEndY ), (0, 255, 255), 2)

            if not locsList.empty(): 
                boxes = locsList.get()
                for box in boxes:
                    (startX, startY, endX, endY) = box
                    cv2.rectangle(frame, (startX*4, startY*4), ((startX + endX)*4, (startY + endY)*4), (0, 255, 0), 2)
                    upperCorner = startX*4,startY*4
                    bottomCorner = (startX + endX)*4, (startY + endY)*4                    

                if not predsTfList.empty():
                    pred_tf = predsTfList.get()
                    temp = tempList[-1]
                    (mask, withoutMask) = pred_tf
                    mask_prediction = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if (mask_prediction == "Mask" and temp<38) else (0, 0, 255)
                    mask_prediction = "{}".format(mask_prediction)
                    
                    label = str(round(temp,1))+mask_prediction
                    last_detected = datetime.now()                      

                if last_detected is not None:  
                    if (datetime.now() - last_detected).total_seconds() < 3:
                        cv2.putText(frame,label,upperCorner,font,fontScale,color,lineType)                           

            window_name = 'Stream'
            # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            final_frame = cv2.resize(frame, (600,1024), interpolation=cv2.INTER_AREA)
            final_crop = final_frame[60:-60, :]
            cv2.imshow("Frame", final_frame)
            key = cv2.waitKey(1) & 0xFF
            raw_capt.truncate(0)
            raw_capt.seek(0)        
            if key == ord("q"):
                cv2.destroyAllWindows() 
                break
            