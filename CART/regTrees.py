import numpy as np

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))              #映射成浮点数
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]     #找到某个feature大于value的数据
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]    #找到某个feature小于等于value的数据
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])               #当函数确定不在对数据进行切分时，将调用该函数来得到叶结点的模型

def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0] #容许的误差下降值
    tolN=ops[1] #是切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:              #如果所有标签值相等即退出
        return None,leafType(dataSet)
    m,n=np.shape(dataSet)
    S=errType(dataSet)
    bestS=float('inf')
    bestIndex=0
    bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if np.shape(mat0)[0]<tolN or np.shape(mat1)[0]<tolN:
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if S-bestS<tolS:
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

def isTree(obj):
    #判断输入变量是否为一棵树
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testData):
    '''
    后剪枝函数
    :param tree:生成的树
    :param testData:测试集
    :return:剪枝后的树
    '''
    # 如果测试集为空,则对树进行塌陷处理
    if np.shape(testData)[0]==0:
        return getMean(tree)
    if  isTree(tree['left']) or isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    elif isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(
            np.power(rSet[:, -1] - tree['right'], 2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree

if __name__=='__main__':
    print('剪枝前:')
    train_filename = 'ex2.txt'
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = createTree(train_Mat)
    print(tree)
    print('剪枝后:')
    test_filename = 'ex2test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))