import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    xArr=[]
    yArr=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))

    return xArr,yArr

def plotDataSet():
    xArr,yArr=loadDataSet('ex0.txt')            #加载数据集
    n=len(xArr)
    xcord=[]
    ycord=[]
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig=plt.figure()
    ax=g=fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

#w=(X^TX)^(-1)X^Ty
def standRegres(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print('矩阵为奇异矩阵，不能求逆')
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

def plotRegression():
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plotlwlrRegression():
    '''
    绘制多条局部加权回归曲线
    :return:None
    '''
    font=FontProperties(fname='/Library/Fonts/Songti.ttc',size=14)
    xArr,yArr=loadDataSet('ex0.txt')
    yHat_1=lwlrTest(xArr,xArr,yArr,1.0)
    yHat_2=lwlrTest(xArr,xArr,yArr,0.01)
    yHat_3=lwlrTest(xArr,xArr,yArr,0.003)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    srtInd = xMat[:, 1].argsort(0)  # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig,axs=plt.subplots(nrows=3,ncols=1,sharex=False,sharey=False,figsize=(10,8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def lwlr(testpoint,xArr,yArr,k=1.0):
    '''
    使用局部加权线性回归计算回归系数w
    :param testpoint:测试样本点
    :param xArr:x数据集
    :param yArr:y数据集
    :param k:高斯核的k
    :return:w
    '''
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye(m))
    for j in range(m):
        diffMat=testpoint-xMat[j,:]
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testpoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__=='__main__':
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)  # 计算回归系数
    # xMat = np.mat(xArr)  # 创建xMat矩阵
    # yMat = np.mat(yArr)  # 创建yMat矩阵
    # yHat = xMat * ws
    # print(np.corrcoef(yHat.T, yMat))
    # plotRegression()
    plotlwlrRegression()

