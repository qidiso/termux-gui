from cv import *
import facerecognition
import numpy as np
facerecog = facerecognition.FaceRecognition("./models", 0.63)

img = cv2.imread('point.jpg')
image_char = img.astype(np.uint8).tostring()
#facerecog.recognize(img.shape[0], img.shape[1], image_char)
print 'add person:sxg',img.shape
ret=facerecog.add_person("sxg", img.shape[0], img.shape[1], image_char)
print 'ret',ret
print 'add sucess!'
img = cv2.imread('sunyi.png')
image_char = img.astype(np.uint8).tostring()
facerecog.add_person("sunyi", img.shape[0], img.shape[1], image_char)

def main():
    cap=VideoCapture()
    #facerecog = facerecognition.FaceRecognition("./models",0.6)
    while True:
        sleep(30)
        img = cap.read()

        if img is None :
            continue
        imshow(img)
        continue

        #img=cv2.resize(img,(112,112))
        image_char = img.astype(np.uint8).tostring()
        rets = facerecog.recognize(img.shape[0], img.shape[1], image_char)

        print 'rets:',rets
        for ret  in  rets:
            #for ret in each:
            print 'draw bounding box for the face'
            rect = ret['rect']
            draw_name(img, rect, ret['name'])
            final = cv2.copyMakeBorder(img,0,0,192,192, cv2.BORDER_CONSTANT,value=[255,255,255])

        imshow(img)

if __name__ == '__main__':

    initcv(main)
    startcv()

