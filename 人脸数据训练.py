import cv2 as cv
import numpy as np
import os
from PIL import Image   #python 图像处理库


#------cv.face.LBPHFaceRecognizer_create().train(param1,param2)-----
#param1:获取人脸ID
#param2：获取人脸特征
#LBPH：将检测到的人脸分为最小单元，并将其与模型中的对应单元进行比较，对每个区域的匹配值产生一个直方图

#os  系统相关库模块
#os.listdir(path)---- 获取目标文件夹的内容，并用列表以字母顺序进行排序保存
#so.path.join(path,f)-----将path和f合并
#os.path.split(path)------将path从‘/’或‘\’分开  并以元组的形式保存

#人脸数据训练

#存储人脸数据函数
def saveFacefunc(path):
    faceSample = []  #存储人脸数据
    faceName = [] #存储人脸姓名
    imagePath = [os.path.join(path,f) for f in os.listdir(path)] #获取到所有目标文件的完整路径
    #os.listdir(path) 获取目标文件夹的内容，并以字母顺序进行排序-----------------------为什么要排序？
    #os.path.join(path,f) 表示获取到文件的完整路径
    print(f'列表已存储文件：{imagePath}')

    face_detect = cv.CascadeClassifier('E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')#调用人脸检测的级联分类器
    for imagepath in imagePath:
        PIL_img = Image.open(imagepath).convert('L')#打开该图片，L表示转化为灰度图像 ---简化矩阵、加快运算速度
        img_numpy = np.array(PIL_img,'uint8') #将图像数据转换为数组
        faces = face_detect.detectMultiScale(img_numpy) #获取图片的人脸特征
        #print(f'脸部特征为:{faces}')
        id = int(os.path.split(imagepath)[1].split('.')[1]) #仅获取序号
        #这里的路径也可以根据实际需求来写
        print(f'当前id为{id}')

        for x,y,w,h in faces:
            faceName.append(id)
            faceSample.append(img_numpy[y:y+h, x:x+w]) #numpy数组切片，从y取到y+h行，从x取到x+w列，构成新的数组，把所画的方框放入列表中
    return faceName,faceSample
if __name__ == '__main__':
    faceName,faceSample = saveFacefunc(path='.//savePhotos') #获取姓名和脸部特征
    #采用LBPH算法
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faceSample,np.array(faceName)) #训练

    #保存训练好的文件
    recognizer.write('trainer.yml')


#多符号分割问题？
#train 只接受整型  Id要转化为int