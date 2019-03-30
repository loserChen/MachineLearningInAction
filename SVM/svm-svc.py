import numpy as np
from os import listdir
from sklearn.svm import SVC

def img2vec(filename):
    '''
    将32x32的二进制图像转换为1x1024向量。
    :param filename:文件名
    :return:向量
    '''
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]     #测试集的labels
    trainingFileList=listdir('trainingDigits')      #返回trainingDigits文件夹目录下的文件名
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))              #返回文件夹下的文件名
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=img2vec('trainingDigits/%s' % (fileNameStr))
    clf=SVC(C=200,kernel='rbf')
    clf.fit(trainingMat,hwLabels)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vec('testDigits/%s' % (fileNameStr))
        classifierResult=clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))

if __name__=='__main__':
    handwritingClassTest()


