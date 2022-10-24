#LBPHFaceRecognizer_creat------------------生成LBPH识别器实例模型
#label,confidence = cv.face.LBPHFaceRecongnizer_creat().predict(param)
#param--输入图片
#label--返回识别结果标签
#confidence--返回的置信度   0表示完全匹配  小于50说明是可以接收的  大于80则认为差别较大

#人脸识别

import cv2 as cv
import os   #系统模块
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "chinese.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

#导入已经训练完成的模型
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
names = []  #存储label

#人脸识别函数
def face_recongnition(img):
    #img = cv.flip(img, 1)  # 图像翻转
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY) #转为灰度图
    face_detect = cv.CascadeClassifier('E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')#调用分类器
    face = face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)#检测到10次 成功认定
    for x,y,w,h in face:
        cv.rectangle(img, (x,y), (x+w,y+h), color=(0,255,0), thickness=3)#画框
        lable,confidence = recognizer.predict(gray[y:y+h, x:x+w])#返回标签和置信度
        #print(lable)
        if confidence >= 60:  #置信度大于50，返回unknow
            cv.putText(img, 'unknow', (x+10,y-10), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0), 1)
        else:
            img = cv2AddChineseText(img,names[lable-1],(x+10,y-30),(0,255,0),30)
            #cv.putText(img,str(names[lable-1]),(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0),1)
        cv.putText(img, (str(round(confidence, 3))), (x + 125, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
    cv.imshow('result', img)

#获取用户label
def get_label():
    path = 'savePhotos/'
    imagePaths = [os.path.join(path,f) for f in sorted(os.listdir(path))]#获取该文件夹下所有的文件的完整路径，并从小到大排序
    #print(imagePaths)
    for imagepath in imagePaths:
        name = str(os.path.split(imagepath)[1].split('.')[0])  #获取姓名
        print(name)
        names.append(name)   #将姓名存入列表当中


#进行人脸识别
cap = cv.VideoCapture(0)  #打开摄像头
get_label()     #调用函数存储label
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_recongnition(frame)    #传入图像进行人脸识别
    if cv.waitKey(1) == 27:
        break

#释放内存
cap.release()
cv.destroyAllWindows()


#识别次数太大 会很卡  --------降低置信度  减少识别次数

#BGR  RBG？？？



