import gc

import numpy as np

path = 'soccer_goal.txt'
goal_arr = []
# print(len('1\n'))
f = open(path)
line = f.readline()
# print(int(line))
goal_arr.append(int(line))
while line:
    line = f.readline()
    if len(line) >= 2:
        # print(int(line))
        goal_arr.append(int(line))
f.close()

x_train = []
y_train = []
for i in range(11, 601):
    line = 'demo_new(' + str(i) + ').npy'
    print(line)
    featMat =  np.load(line)
    x_train.append(featMat)
    #np.save('demo_new(' + str(i) + ').npy', np.array(featMat))
    #del featMat
    #gc.collect()
    if i in goal_arr:
        y_train.append(1)
    else:
        y_train.append(0)
np.save('x_train.npy', np.array(x_train))
np.save('y_train.npy', np.array(y_train))

x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')

x_test = []
y_test = []
for i in range(601, 709):
    line = 'demo_new(' + str(i) + ').mpg'
    print(line)
    featMat = np.load(line)
    x_test.append(featMat)
    #np.save('demo_new(' + str(i) + ').npy', np.array(featMat))
    #del featMat
    #gc.collect()
    if i in goal_arr:
        y_test.append(1)
    else:
        y_test.append(0)
np.save('x_test.npy', np.array(x_test))
np.save('y_test.npy', np.array(y_test))