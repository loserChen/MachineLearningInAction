import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadDataSet(filename):
    numFeat=len((open(filename).readline().split('\t')))
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__=='__main__':
    dataArr,classLabels=loadDataSet('/Users/chenzeyuan/PycharmProjects/practice/AdaBoost/horseColicTraining2.txt')
    testArr,testLabelArr=loadDataSet('/Users/chenzeyuan/PycharmProjects/practice/AdaBoost/horseColicTest2.txt')
    bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=10)
    bdt.fit(dataArr,classLabels)
    predictions=bdt.predict(dataArr)
    errArr=np.mat(np.ones((len(dataArr),1)))
    print('训练集的误差%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))
