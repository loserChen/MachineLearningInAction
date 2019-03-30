import matplotlib.pyplot as plt
import numpy as np
import random


class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2))) #根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。


def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def calEk(oS,k):
    '''
    计算误差
    :param oS:数据结构
    :param k:标号为k的数据
    :return:标号为k的误差
    '''
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek

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

def selectJ(i,oS,Ei):
    '''
    内循环的启发式方法
    :param i:标号为i的数据的索引值
    :param oS:数据结构
    :param Ei:标号为i的数据误差
    :return:使得Ei-Ej最大的alpha的索引值与差值
    '''
    maxK=-1
    maxDelteE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]      #返回非零E值所对应的alpha值的索引
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calEk(oS,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDelteE:
                maxK=k
                maxDelteE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    '''
    计算Ek并更新误差缓存
    :param oS:数据结构
    :param k:索引值
    :return:None
    '''
    Ek=calEk(oS,k)
    oS.eCache[k]=[1,Ek]

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

def innerL(i,oS):
    '''
    优化的SMO算法
    :param i:索引值
    :param oS:数据结构
    :return:返回是否有一对alpha值发生变化
    '''
    Ei=calEk(oS,i)      #步骤1：计算误差Ei
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej=selectJ(i,oS,Ei)           #使用内循环启发式方法选择alpha_j，并计算Ej
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[
            j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[
            j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter):
    '''
    完整的SMO算法
    :param dataMatIn:数据矩阵
    :param classLabels:类别标签
    :param C:惩罚参数
    :param toler:容错阈值
    :param maxIter:最大迭代次数
    :return:SMO计算的b，alphas
    '''
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while (iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet=False
        elif alphaPairsChanged==0:
            entireSet=True
        print("迭代次数: %d" % iter)
    return oS.b,oS.alphas

def showClassifier(dataMat,classLabels,w,b):
    data_positive = []
    data_negative = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
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

def calcWs(alphas,dataArr,classLabels):
    """
    计算w
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        alphas - alphas值
    Returns:
        w - 计算得到的w
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__ == '__main__':
	dataArr, classLabels = loadDataSet('testSet.txt')
	b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
	w = calcWs(alphas,dataArr, classLabels)
	showClassifier(dataArr, classLabels, w, b)