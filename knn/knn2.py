import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import operator
'''
使用k-近邻算法改进约会网站的配对效果
利用数据集datingTestSet.txt

海伦收集的样本数据主要包含以下3种特征：
x1:每年获得的飞行常客里程数
x2:玩视频游戏所消耗时间百分比
x3:每周消费的冰淇淋公升数

对数据的分类为：
y:不喜欢的人
y:魅力一般的人
y:极具魅力的人
'''
def file2matrix(filename):
    '''
    将文件转换为矩阵以使用我们在knn1中实现的分类函数
    :param filename: 文件名
    :return: 特征矩阵，label向量
    '''
    str2Int={'didntLike':1,'smallDoses':2,'largeDoses':3}
    with open(filename) as fr:
        arrayOLines=fr.readlines()  #读取数据
    numberOfLines=len(arrayOLines) #求数据的数量
    returnMat=np.zeros((numberOfLines,3)) #numberOfLines行，3列的值为0的矩阵
    classLabelVector=[] #需要返回的分类标签列表
    index=0 #为方便下面一行行的传值
    for line in arrayOLines: #对arrayOLines一行行遍历
        line=line.strip()   #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        listFromLine=line.split('\t')  # #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        returnMat[index,:]=listFromLine[0:3]    #将得到的前三列提取到returnMat中
        classLabelVector.append(int(str2Int[listFromLine[-1]])) #将得到的label添加到classLabel中
        index+=1
    return returnMat,classLabelVector

def plotData(x,y):
    # 设置汉字格式,下述设置针对mac而言
    #windows设置：font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    font = FontProperties(fname='/Library/Fonts/Songti.ttc', size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabel=len(y)
    labelColors=[]
    for i in y:
        if i==1:
            labelColors.append('black')
        elif i==2:
            labelColors.append('orange')
        elif i==3:
            labelColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=x[:, 0], y=x[:, 1], color=labelColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=x[:, 0], y=x[:, 2], color=labelColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=x[:, 1], y=x[:, 2], color=labelColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    # 显示图片
    plt.show()

def autoNorm(dataSet):
    '''
    从数据集中可以看出，每列数据集的量级是不一样的，
    所以数字差值最大的属性对计算结果影响比较大
    因此为了平衡每个特征的重要性
    我们使用该函数对数据集进行归一化，使特征都处于同一个量级
    :param dataSet: 数据集
    :return: 归一化后的数据集
    '''
    min=dataSet.min(0)
    max=dataSet.max(0)
    ranges=max-min
    normDataSet=np.zeros(np.shape(dataSet)) #创建一个与dataSet一个大小的矩阵
    m=dataSet.shape[0] #获取行数
    normDataSet=dataSet-np.tile(min,(m,1))  #原始值减去最小值
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet
def classify0(inX,dataSet,labels,k):
    '''
    分类,与knn1.py中的一样
    :param inX:用于分类的输入向量
    :param dataSet:输入的训练样本集
    :param labels:标签向量
    :param k:选择最近邻居的数目
    :return:
    '''
    dataSetSize=dataSet.shape[0] #获得数据集的数量
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet  #np.tile()函数就是将inX在行方向上复制dataSetSize倍，在列方向上复制1倍
    sqDiffMat=diffMat**2  #对diffMat求平方
    sqDistances=sqDiffMat.sum(axis=1)  #对列方向上的数据进行求和
    distances=sqDistances**0.5 #开根号
    sortedDistIndices=distances.argsort()  #排序，得到距离从小到大排序的序号
    classCount={} #计算类的数量的字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]  #获得欧式距离最小的k个标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #每得到一个对应的标签就在classCount上加一
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]  #返回得票数最多的label

def datingClassTest(filename):
    x,y=file2matrix(filename)
    plotData(x, y)
    ratio=0.1  #切割的比例
    normMat=autoNorm(x)
    m=normMat.shape[0]
    numTestVecs=int(m*ratio)  #得到10%的数据集长度
    errorCount=0.0

    for i in range(numTestVecs):#将测试集中的数据进行分类
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],
        y[numTestVecs:m],4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, y[i]))
        if classifierResult != y[i]: #如果分类错误，errorCount加1
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))

def classifyPerson(filename):
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开并处理数据
    x,y = file2matrix(filename)
    minVals = x.min(0)
    maxVals = x.max(0)
    ranges = maxVals - minVals
    #训练集归一化
    normMat= autoNorm(x)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, y, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

if __name__=='__main__':
    filename='datingTestSet.txt'
    datingClassTest(filename) #得到分类错误的百分比
    classifyPerson(filename) #可交互的程序，用户输入数据，得到结果