import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    # if w == 0 or h == 0:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    # else:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    e = TfPoseEstimator(get_graph_path(args.model), target_size=(512, 512))
    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)


    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    #humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

    humans = e.inference(image, resize_to_default=True, upsample_size=1.0)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

    heatMat=np.clip(e.heatMat,0,1)

    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)


    # show network output
    a = fig.add_subplot(2, 2, 2)
    #plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
