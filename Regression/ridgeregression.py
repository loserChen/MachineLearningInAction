from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    xArr=[]
    yArr=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

def ridgeRegres(xMat,yMat,lam=0.2):
    '''
    岭回归就是带l2正则的线性回归
    :param xMat:数据集
    :param yMat:标签
    :param lam:学习率
    :return:回归系数
    '''
    xTx=xMat.T*xMat
    denom=xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom)==0.0:
        print('矩阵为奇异矩阵，不能转置')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)  # 行与行操作，求均值
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis=0)  # 行与行操作，求均值
    xVar = np.var(xMat, axis=0)  # 行与行操作，求方差
    xMat = (xMat - xMeans) / xVar  # 数据减去均值除以方差实现标准化
    numTestPts = 30  # 30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))  # 初始回归系数矩阵
    for i in range(numTestPts):  # 改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))  # lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T  # 计算回归系数矩阵
    return wMat


def plotwMat():
    """
    函数说明:绘制岭回归系数矩阵
    """
    font = FontProperties(fname='/Library/Fonts/Songti.ttc', size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()
if __name__ == '__main__':
    plotwMat()
