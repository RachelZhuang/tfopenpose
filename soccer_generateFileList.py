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

trainlist = open('trainlist1.txt','a')
testlist = open('testlist1.txt','a')

# x_train = []
# y_train = []
for i in range(11, 609):
    line = 'soccer_video/demo_new(' + str(i) + ').mpg'
    #print(line)
    #featMat =  np.load(line)
    #x_train.append(featMat)
    #np.save('demo_new(' + str(i) + ').npy', np.array(featMat))
    #del featMat
    #gc.collect()
    if i in goal_arr:
        #y_train.append(1)
        #print(1)
        line=line+' 1\n'
    else:
        #y_train.append(0)
        #print(0)
        line=line+' 0\n'
    print(line)
    trainlist.write(line)
trainlist.close()

for i in range(609, 709):
    line = 'soccer_video/demo_new(' + str(i) + ').mpg'
    if i in goal_arr:
        line=line+' 1\n'
    else:
        line=line+' 0\n'
    print(line)
    testlist.write(line)
testlist.close()
