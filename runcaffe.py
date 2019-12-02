import sys
sys.path.insert(0,"/home/huolei/ssd/caffe_mo/python")
import caffe
import numpy as np
import cv2


caffe_net = caffe.Net("Pelee.prototxt","Pelee.caffemodel",caffe.TEST)

src = cv2.imread("001763.jpg")
src = cv2.resize(src,(304,304))
src = np.transpose(src,(2,0,1))
src = src.astype(np.float32)
src[0]=src[0]-103.94
src[1]=src[1]-116.78
src[2]=src[2]-123.68

src = src*0.017

src = src[np.newaxis,...]

caffe_net.blobs['blob1'].data[...] = src

output = caffe_net.forward()

out1 = output['view_blob11']

out2 = output['softmax_blob1']

print(out1)
print(out2)


