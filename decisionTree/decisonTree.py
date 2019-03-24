from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pickle
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']		#分类属性
    return dataSet, labels                             #返回数据集和分类属性

def calShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet: 数据集
    :return: 计算出的香农熵
    '''
    numEntires=len(dataSet)
    labelCounts={}          #用于保存标签出现次数的字典
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():   #当前数据的label不在labelCounts中
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1    #每出现一次加一
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntires
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    该函数主要用来删去数据集中的某一列等于具体值的数据
    :param dataSet: 数据集
    :param axis: 哪一列
    :param value: 特征值
    :return: 分割后的数据集
    '''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])   #注意extend和append的区别
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择一个信息熵最大的特征进行分割
    :param dataSet: 数据集
    :return: 返回一个信息熵最大的特征
    '''
    numFeatures=len(dataSet[0])-1  #特征数
    baseEntropy=calShannonEnt(dataSet) #整体的香农熵
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[x[i] for x in dataSet] #得到某列的数据
        uniqueVals=set(featList) #利用set来得到唯一的值
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy   #信息熵的计算
        print("第%d个特征的增益为%.3f" % (i,infoGain))
        if infoGain>bestInfoGain:   #得到最大信息熵的对应特征
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    '''
    如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子结点，
    在这种情况下，我们通常会采用多数表决的方法来决定叶子结点的分类
    :param classList: 类的列表
    :return: 返回出现次数最多的类
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #根据字典的值降序排序
    return sortedClassCount[0][0]   #返回classList中出现次数最多的元素

def createTree(dataSet,labels,featLabels):
    '''
    递归地创建树
    :param dataSet: 数据集
    :param labels: 每个特征对应的名字
    :return: 创建的树的字典表示
    '''
    classList=[x[-1] for x in dataSet]  #取分类标签
    if classList.count(classList[0])==len(classList):    #如果类别完全相同则停止继续
        return classList[0]
    if len(dataSet[0])==1:    #如果遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)   #选择最优的特征
    bestFeatLabel=labels[bestFeat]     #获取最优特征的标签
    myTree={bestFeatLabel:{}}   #根据最优特征的标签生成树
    del(labels[bestFeat])
    featLabels.append(bestFeatLabel)
    featValues=[x[bestFeat] for x in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:  #递归的创建树
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value),labels,featLabels)
    return myTree
#以上就是我们建立一颗树的全部过程，但是树的表示是使用字典的，这无法让我们有个直观的感受，
#因此我们需要利用matplotlib来将树绘制出来
def getNumLeafs(myTree):
    '''
    获取该树的叶子结点数
    :param myTree: 我们上面生成的字典类型的树
    :return: 返回叶子数
    '''
    numLeafs=0 #初始化叶子树
    firstStr=list(myTree.keys())[0]  #获得当前字典的第一个key
    secondDict=myTree[firstStr] #获取该key下面的字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict)     #递归
        else:numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取树的深度
    :param myTree: 我们上面生成的字典类型的树
    :return: 返回树的层数
    '''
    maxDepth=0
    firstStr = list(myTree.keys())[0]  #获得当前字典的第一个key
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
    return maxDepth

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    '''
    绘制结点
    :param nodeTxt:结点名
    :param centerPt:文本位置
    :param parentPt:标注的箭头位置
    :param nodeType:结点格式
    :return:无
    '''
    arrow_args=dict(arrowstyle="<-")   #设置箭头格式
    font=FontProperties(fname='/Library/Fonts/Songti.ttc', size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotMidText(cntrPt,parentPt,txtString):
    '''
    绘制有向边的属性值
    :param cntrPt: 子结点坐标
    :param parentPt: 父结点坐标
    :param txtString: 标注的内容
    :return: 无
    '''
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]  #取父子结点之间x坐标的中点
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]  #取父子结点之间y坐标的中心
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotation=30)

def plotTree(myTree,parentPt,nodeTxt):
    '''
    画出整个树
    :param myTree: 树的字典表示
    :param parentPt: 父节点
    :param nodeTxt: 结点名称
    :return: 无
    '''
    decisionNode=dict(boxstyle="sawtooth",fc="0.8")  #设置结点格式
    leafNode=dict(boxstyle="round4",fc="0.8")   #设置叶结点格式
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff) #中心位置
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    '''
    调用我们上述实现的函数，从而进行绘制
    :param inTree: 树的字典表示
    :return: 无
    '''
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()                                                                                 #显示绘制结果

def classify(inputTree,featLabels,testVec):
    '''
    分类函数，利用我们已经生成的树模型对需要分类的向量进行分类
    :param inputTree: 树的字典表示
    :param featLabels: 被分割的特征列表
    :param testVec: 需要分类的向量
    :return: 返回分类的类别
    '''
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:   classLabel=secondDict[key]
    return classLabel
#针对上述一系列函数，我们已经完成了树的生成，树的绘制，树的分类，但是我们目前我们
#每分类一次就需要生成一次决策树，那么我们考虑可以将生成的树保存下来。
def storeTree(inputTree,filename):
    '''
    保存我们生成的树
    :param inputTree:树的字典形式
    :param filename: 文件名
    :return: 无
    '''
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)

def grabTree(filename):
    '''
    将我们之前保存的树重新加载出来
    :param filename: 文件名
    :return: 生成的树的字典表示
    '''
    fr=open(filename,'rb')
    return pickle.load(fr)
if __name__=='__main__':
    dataSet,labels=createDataSet()
    #featLabels=[]
    #myTree=createTree(dataSet,labels,featLabels)
    #storeTree(myTree,'classifierStorage.txt')
    #storeTree(featLabels,'featLabels.txt')

    #通过上两行我们已经在文件夹中生成了树的存储文件
    #我们只要直接将树加载出来就好了
    myTree=grabTree('classifierStorage.txt')
    featLabels=grabTree('featLabels.txt')
    #createPlot(myTree)
    print(classify(myTree,featLabels,[0,0,0,0]))