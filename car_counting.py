import time
import cv2
import numpy as np


class Kordinat:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Sensor:
    def __init__(self,kordinat1,kordinat2,width,higth):
        self.kordinat1=kordinat1
        self.kordinat2 = kordinat2
        self.width=width
        self.hight=higth
        self.mask=np.zeros((width,higth,1),np.uint8)*abs(self.kordinat2.y-self.kordinat1.y)
        self.full_mask_area=abs(kordinat2.x-kordinat1.x)
        cv2.rectangle(self.mask,(self.kordinat1.x,self.kordinat1.y),(self.kordinat2.x,self.kordinat2.y),(255),thickness=cv2.FILLED)
        self.heavy_vehicle=0
        self.car=0
        self.situation=False
        self.bounding=0


cap=cv2.VideoCapture(r"D:\WORKSPACE\computer_vision\videos_and_images\car_new_video.mp4")
ret,frame=cap.read()
cropped_image= frame[0:450, 0:450]

subtractor=cv2.createBackgroundSubtractorMOG2()

Sensorx=Sensor(Kordinat(1,cropped_image.shape[1]-35),
               Kordinat(340,cropped_image.shape[1]-30),
               cropped_image.shape[0],
               cropped_image.shape[1])
kernel=np.ones((5,5),np.uint8)


while True:
    ret,frame=cap.read()
    cropped_image = frame[0:450, 0:450]
    deleted_background=subtractor.apply(cropped_image)
    opening_image=cv2.morphologyEx(deleted_background,cv2.MORPH_OPEN,kernel)
    _,opening_image=cv2.threshold(opening_image,125,255,cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(opening_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    zeros_image=np.zeros((cropped_image.shape[0],cropped_image.shape[1],1),np.uint8)
    result=cropped_image.copy()

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 75 and h > 75 and w < 160 and h < 160):
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            cv2.rectangle(zeros_image, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)
            Sensorx.bounding=1
        if (w > 160 and h > 160 ):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv2.rectangle(zeros_image, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)
            Sensorx.bounding=2

    mask=Sensorx.mask
    mask1=np.zeros((zeros_image.shape[0],zeros_image.shape[1],1),np.uint8)
    mask_result=cv2.bitwise_or(zeros_image,zeros_image,mask=mask)
    white_cell=np.sum(mask_result==255)

    sensor_rate=white_cell/Sensorx.full_mask_area
    if sensor_rate>0:
        print("result:",sensor_rate)
    if sensor_rate>=0.8 and sensor_rate<=7 and Sensorx.situation==False:
        cv2.rectangle(result,(Sensorx.kordinat1.x,Sensorx.kordinat1.y),(Sensorx.kordinat2.x,Sensorx.kordinat2.y),(0,255,0),thickness=cv2.FILLED)
        Sensorx.situation=True

    elif sensor_rate<0.8 and Sensorx.situation==True:
        cv2.rectangle(result, (Sensorx.kordinat1.x, Sensorx.kordinat1.y), (Sensorx.kordinat2.x, Sensorx.kordinat2.y),
                          (0, 0, 255), thickness=cv2.FILLED)


        if Sensorx.bounding==1 :
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                cv2.rectangle(zeros_image, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)

                Sensorx.car += 1
                Sensorx.situation=False

        if Sensorx.bounding==2:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                cv2.rectangle(zeros_image, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)
                Sensorx.heavy_vehicle += 1
                Sensorx.situation = False


    if white_cell>0:
        cv2.putText(result,"heavy traffic", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1)
        time.sleep(0.01)


    else:
        cv2.rectangle(result, (Sensorx.kordinat1.x, Sensorx.kordinat1.y), (Sensorx.kordinat2.x, Sensorx.kordinat2.y),
                          (0, 0, 255), thickness=cv2.FILLED)
        cv2.putText(result, "Traffic is empty", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)

    cv2.putText(result,str("car:{}").format(Sensorx.car),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0),thickness=1)

    cv2.putText(result, str("heavy vehicle:{}").format(Sensorx.heavy_vehicle), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0),
                    thickness=1)
    #cv2.putText(result,"Egehan Yalcin",(230,330),cv2.FONT_ITALIC,1,(0,255,0),thickness=1)


    cv2.imshow("frame",result)
    cv2.imshow("zeros_image",zeros_image)
    cv2.imshow("opening_image",opening_image)


    if cv2.waitKey(30) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
