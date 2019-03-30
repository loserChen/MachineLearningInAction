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

class optStruct:
    """
    数据结构，维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
    """
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2))) #根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K=np.mat(np.zeros((self.m,self.m)))  #初始化核K
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def kernelTrans(X,A,kTup):
    '''
    通过核函数将数据转换到更高维的空间
    :param X:数据矩阵
    :param A:单个数据的向量
    :param kTup:包含核函数信息的元组
    :return:计算的核K
    '''
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('核函数无法识别')
    return K
def calEk(oS,k):
    '''
    计算误差
    :param oS:数据结构
    :param k:标号为k的数据
    :return:标号为k的误差
    '''
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
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
        eta = 2.0 * oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
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
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i]- oS.labelMat[
            j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[
            j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
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

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    '''
    完整的SMO算法
    :param dataMatIn:数据矩阵
    :param classLabels:类别标签
    :param C:惩罚参数
    :param toler:容错阈值
    :param maxIter:最大迭代次数
    :return:SMO计算的b，alphas
    '''
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
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

def testRbf(k1 = 1.3):
    """
    测试函数
    Parameters:
        k1 - 使用高斯核函数的时候表示到达率
    Returns:
        无
    """
    dataArr,labelArr = loadDataSet('testSetRBF.txt')                        #加载训练集
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))        #根据训练集计算b和alphas
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]                                        #获得支持向量
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))                #计算各个点的核
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b     #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1        #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("训练集错误率: %.2f%%" % ((float(errorCount)/m)*100))             #打印错误率
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')                         #加载测试集
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))                 #计算各个点的核
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b         #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1        #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("测试集错误率: %.2f%%" % ((float(errorCount)/m)*100))             #打印错误率

if __name__=='__main__':
    # dataArr, labelArr = loadDataSet('testSetRBF.txt')  # 加载训练集
    # showDataSet(dataArr, labelArr)
    testRbf()