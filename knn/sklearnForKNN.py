import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
def img2vector(filename):
    '''
    将一个32*32的二进制图像矩阵转换为1*1024的向量
    :param filename:
    :return:
    '''
    returnVect=np.zeros((1,1024)) #创建1x1024零向量
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def handwritingClassTest():
    labels=[]
    trainingFileList=listdir('trainingDigits')  #返回trainingDigits目录下的文件名
    m=len(trainingFileList) #返回文件夹下文件的个数
    trainingMat=np.zeros((m,1024)) #初始化训练的Mat矩阵
    for i in range(m):
        fileNameStr=trainingFileList[i]  #获得文件的名字
        classNumber=int(fileNameStr.split('_')[0]) #获得分类的数字
        labels.append(classNumber)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr) #将每一个文件的1x1024数据存储到trainingMat矩阵中
    neigh=kNN(n_neighbors=3,algorithm='auto')  #构建kNN分类器
    neigh.fit(trainingMat,labels) #拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        result=neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (result, classNumber))
        if (result != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))
if __name__=='__main__':
    handwritingClassTest()