import cv2 as cv
import numpy as np
from numpy.lib.function_base import append
cap = cv.VideoCapture(0)
wht = 320
nmsThreshold = 0.3
conf_threshold = 0.3
classFile = 'coco.names'
classNames = []
with open(classFile , 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelconfig = 'yolov3tiny.cfg'
modelweights  = 'yolov3-tiny.weights'
net = cv.dnn.readNetFromDarknet(modelconfig,modelweights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
def find_objects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                w,h = (int(det[2]*wT) , int(det[3]*hT))
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confs,conf_threshold,nmsThreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(img , f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%",(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2)

while True:
    isTrue,img = cap.read()
    blob = cv.dnn.blobFromImage(img , 1/255 , (wht,wht),[0.,0,0],crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    # print(net.getUnconnectedOutLayers())
    output_names = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(output_names)
    output = net.forward(output_names)
    # print(len(output[0][0]))
    find_objects(output,img)
    # print(layers_names)
    cv.imshow("Webcam" ,img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()