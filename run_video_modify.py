import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

#C is the totoal channel number,T is the total frame number
#c represents the c'th channel(start from one),t represents the t'th frame(start from one)
def GetColorScheme(C,T,c,t):
    c=c-1
    path=(T-1)/(C-1)
    index=int((t-1)/path)

    f1=1-(t-(path*index+1))/(path*(index+1)-path*index)
    c1=index
    f2=1-f1
    c2=index+1

    if c==c1:
        return f1
    elif c==c2:
        return f2
    else:
        return 0
# x = np.linspace(1, 5, 1000)  # 这个表示在-5到5之间生成1000个x值
# y = [GetColorScheme(3,5,2,i) for i in x]  # 对上述生成的1000个数循环用sigmoid公式求对应的y
# plt.plot(x, y)  # 用上述生成的1000个xy值对生成1000个点
# plt.show()


def GetFinalFeatMat(allHeatMat):
    allHeatMat =np.array(allHeatMat)
    C=2
    T=len(allHeatMat)
    featureMat=np.zeros((C,T,allHeatMat.shape[1],allHeatMat.shape[2],allHeatMat.shape[3]),dtype=np.float32)
    for c in range(C):
        for t in range(T):
            ratio=GetColorScheme(C,T,c+1,t+1)
            featureMat[c,t,:,:,:]=ratio*allHeatMat[t,:,:,:]
    SMat=featureMat.sum(axis=1)
    for c in range(C):
        for j in range(SMat.shape[3]):
            SMat[c, :, :, j]=SMat[c, :, :, j]/SMat[c,:,:,j].max()
    UMat=SMat
    LMat=np.zeros((1,UMat.shape[1],UMat.shape[2],UMat.shape[3]),dtype=np.float32)
    LMat[0,:,:,:]=UMat.sum(axis=0)
    e=1
    NMat=np.zeros((UMat.shape[0],UMat.shape[1],UMat.shape[2],UMat.shape[3]),dtype=np.float32)
    for c in range(C):
        NMat[c,:,:,:]=UMat[c,:,:,:]/(1+LMat[0,:,:,:])

    mergeMat=np.concatenate((UMat,NMat,LMat),axis=0)
    mergeMat=mergeMat.reshape((mergeMat.shape[0]*mergeMat.shape[3],mergeMat.shape[1],mergeMat.shape[2]))

    return mergeMat
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(512, 512))
    cap = cv2.VideoCapture(args.video)

    allHeatMat=[]

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        if image is None:
            break
        humans = e.inference(image,resize_to_default=True, upsample_size=1.0)
        allHeatMat.append(e.heatMat.clip(0,1))
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    finalMat=GetFinalFeatMat(allHeatMat)

logger.debug('finished+')
