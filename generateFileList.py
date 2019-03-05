import os

rootdir = 'splits'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
trainData=[]
testData=[]

trainlist = open('trainlist1.txt','a')
testlist = open('testlist1.txt','a')

for i in range(0,len(list)):
       path = os.path.join(rootdir,list[i])
       pre=list[i][:-16]
       #if os.path.isfile(path):
       if 'split1'in list[i]:
        #print(list[i])
        # read txt method one
        f = open(path)
        line = f.readline()
        if line[-2] == '1':
            line=pre+'/'+line[:-2]+str(i)+'\n'
            trainData.append(line)
            trainlist.write(line)
        elif line[-2] == '2':
            line=pre+'/'+line[:-2]+str(i)+'\n'
            testData.append(line)
            testlist.write(line)
        while line:
            print(line)
            line = f.readline()
            if line=='':
                continue
            elif line[-2]=='1':
                line = pre + '/' + line[:-2] + str(i) + '\n'
                trainData.append(line)
                trainlist.write(line)
            elif line[-2]=='2':
                line = pre + '/' + line[:-2] + str(i) + '\n'
                testData.append(line)
                testlist.write(line)
        f.close()
trainlist.close()
testlist.close()
print(1)


