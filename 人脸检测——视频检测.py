import cv2 as cv

#人脸检测函数
def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #转化为灰度图片，简化矩阵、提高运算速度
    #调用人脸检测的级联分类器
    face_classifier = cv.CascadeClassifier('E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    #对人脸进行检测，每次图像缩小的比例为1.1，每一个目标至少检测5次
    face_feature = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #face_feature  返回的是人脸的特征数据，一个人脸返回一组特征；x个人脸返回x组特征
    for x,y,w,h in face_feature:   #遍历x个人脸
        cv.rectangle(img, (x,y), (x+w,y+h), color=(0,0,255), thickness=3)#对人脸画框
        #print(x,y,w,h)
    cv.imshow('人脸检测',img)
#读取摄像头
cap = cv.VideoCapture(0)    #获取摄像头  也可以读取视频路径

num = 1
print('请输入名字\n')
name = input()
#获取图像
while cap.isOpened():
    flag,frame = cap.read() #获取帧图片
    if not flag:
        break
    face_detect(frame)  #人脸检测
    if cv.waitKey(1) == ord('s'):
        #imgFile = './savePhotos/'+'capture_' + str(num) + '.jpg'
        #imgFile = 'E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\' + 'capture_' + str(num) + '.jpg'
        #cv.imwrite("E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\" + name + "." + str(num) + ".jpg", frame)
        imgFile = 'E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\' + name + '.' + str(num) + '.jpg'
        cv.imencode('.jpg', frame)[1].tofile(imgFile)   #存储中文图片

        #path = './savePhotos/'
       # cv.imwrite(path + 'capture_' + str(num) + '.jpg', frame)
        print('保存成功!')
        num += 1
    if cv.waitKey(1) == 27: #Esc对应的ascall码
        break


cap.release()
cv.destroyAllWindows()