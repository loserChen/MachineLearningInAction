import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet(filename):
    '''
    载入数据
    :param filename: 数据集路径
    :return: 数据与对应的标签
    '''
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def showDataSet(dataMat,labelMat):
    '''
    展示数据集
    :param dataMat: x
    :param labelMat: y
    :return: None
    '''
    data_positive=[]
    data_negative=[]
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_positive.append(dataMat[i])
        else:
            data_negative.append(dataMat[i])
    data_positive=np.array(data_positive)           #转换成numpy矩阵
    data_negative=np.array(data_negative)          #转换成numpy矩阵
    plt.scatter(np.transpose(data_positive)[0],np.transpose(data_positive)[1])    #正样本散点图
    plt.scatter(np.transpose(data_negative)[0],np.transpose(data_negative)[1])    #负样本散点图
    plt.show()

def selectJrand(i,m):
    '''
    随机选择第二个参数alpha j
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数量
    :return: 随机选择的j
    '''
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    '''
    调整大于H或小于L的alpha的值
    :param aj:alpha
    :param H:alpha的上界
    :param L:alpha的下界
    :return:经过调整的alpha
    '''
    if aj>H:
        aj=H
    elif aj<L:
        aj=L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''
    SMO算法的一个有效版本
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 最大的循坏次数
    :return:返回优化后的alpha和b
    '''
    dataMatrix=np.mat(dataMatIn)         #转换成numpy的mat存储
    labelMat=np.mat(classLabels).transpose()
    b=0
    m,n=np.shape(dataMatrix)
    alphas=np.mat(np.zeros((m,1)))       #初始化alpha的参数，设为0
    iter_num=0                           #初始化迭代次数
    while(iter_num<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei=fXi-float(labelMat[i])          #计算误差
            #优化alpha，更设定一定的容错率。q
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j=selectJrand(i,m)      #随机选择j
                #计算误差Ej
                fXj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()    #保存更新前的alpha值，使用深拷贝
                alphaJold=alphas[j].copy()
                # 计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print("L==H")
                    continue
                #计算eta,注意这里的eta计算与统计学习方法是相反的，因此这里的eta是永远小于等于0
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j,:] * dataMatrix[j, :].T
                if eta>=0:
                    print("eta>=0")
                    continue
                #更新alpha j
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                #将alpha j进行剪辑
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha_j变化太小")
                    continue
                # 更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged+=1            #统计优化次数
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
                # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b,alphas

def showClassifier(dataMat,w,b):
    data_positive = []
    data_negative = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_positive.append(dataMat[i])
        else:
            data_negative.append(dataMat[i])
    data_positive = np.array(data_positive)  # 转换成numpy矩阵
    data_negative = np.array(data_negative)  # 转换成numpy矩阵
    plt.scatter(np.transpose(data_positive)[0], np.transpose(data_positive)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_negative)[0], np.transpose(data_negative)[1])  # 负样本散点图
    #绘制直线
    x1=max(dataMat)[0]
    x2=min(dataMat)[0]
    a1,a2=w
    b=float(b)
    a1=float(a1[0])
    a2=float(a2[0])
    y1,y2=(-b-a1*x1)/a2,(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    for i,alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

if __name__=='__main__':
    dataMat,labelMat=loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifier(dataMat, w, b)