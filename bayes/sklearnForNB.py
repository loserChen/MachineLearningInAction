import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

def TextProcessing(folder_path,test_size=0.2):
    folder_list=os.listdir(folder_path)   #查看路径下的文件夹
    data_list=[]
    class_list=[]

    for folder in folder_list:                  #对文件夹遍历
        new_folder_path=os.path.join(folder_path,folder)   #根据子文件夹生成新的文件路径
        files=os.listdir(new_folder_path)   #获取子文件夹下的txt文件列表

        j=1
        for file in files:
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:  #打开txt文件
                raw=f.read()

            word_cut=jieba.cut(raw,cut_all=False)  #精简模式，返回一个可迭代的generator
            word_list=list(word_cut)     #转换为list

            data_list.append(word_list)
            class_list.append(folder)
            j+=1

    data_class_list=list(zip(data_list,class_list))    #zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)                    #将data_class_list乱序
    index=int(len(data_class_list)*test_size)+1        #将训练集和测试集切分的索引值
    train_list=data_class_list[index:]                 #训练集
    test_list=data_class_list[:index]                   #测试集
    train_data_list,train_class_list=zip(*train_list)  #训练集解压缩
    test_data_list,test_class_list=zip(*test_list)     #测试集解压缩

    all_words_dict={}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word]+=1
            else:
                all_words_dict[word]=1

    all_words_tuple_list=sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
    all_words_list,all_words_nums=zip(*all_words_tuple_list)
    all_words_list=list(all_words_list)
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
#利用上述函数确实能进行分词，但是我们可以看到分词得到很多符号，像逗号，点号啥的，这对我们进行判断是没用的
#因此我们使用stopwords_cn.txt来将这些不重要的单词去除
def MakeWordsSet(words_file):
    words_set=set()
    with open(words_file,'r',encoding='utf-8') as f:    #打开文件
        for line in f.readlines():
            word=line.strip()
            if len(word)>0:
                words_set.add(word)
    return words_set

def words_dict(all_words_list,deleteN,stopwords_set=set()):
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_words_list),1):
        if n>1000:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n+=1
    return feature_words

def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):                        #出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list                #返回结果

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__=='__main__':
    folder_path='./SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)


    test_accuracy_list=[]
    # deleteNs=range(0,1000,20)
    feature_words = words_dict(all_words_list, 800, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)

    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.show()
    ave=lambda c:sum(c)/len(c)
    print(ave(test_accuracy_list))
