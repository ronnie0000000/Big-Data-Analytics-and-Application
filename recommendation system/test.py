# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import math

# 最大item的ID
MaxItem_ID=0
MaxUser_ID=0

def getAttribute():
    itemattr=dict()
    with open("data/itemAttribute.txt",'r') as f:
        line=f.readline()
        while line:
            item,attr1,attr2=line.split('|')
            item=int(item)
            attr1=int(attr1) if attr1[0]!='N' else 0
            attr2=int(attr2) if attr2[0]!='N' else 0
            itemattr[item]=[attr1,attr2]
            line=f.readline()
    f.close()
    return itemattr

#获取相似的物品评分
def get_related_person(mark,itemattr,item,num=100):#mark是该用户之前的打分
    attr1,attr2=itemattr[item]
    ans=[]
    mean1=(attr1+attr2)/2
    std1=math.sqrt(((attr1-mean1)**2+(attr2-mean1)**2)/2)
    for id,score in mark.items():
        #计算皮尔逊相关系数
        temp_attr1=itemattr[id][0]
        temp_attr2=itemattr[id][1]
        mean2=(temp_attr1+temp_attr2)/2
        std2=math.sqrt(((temp_attr1-mean2)**2+(temp_attr2-mean2)**2)/2)
        cov=((attr1-mean1)*(temp_attr1-mean2)+(attr2-mean1)*(temp_attr2-mean2))/2
        similarity=cov/(std1*std2)
        ans.append([id,similarity])
        if len(ans)>num:
            return ans[:num]
        return ans

def get_related(mark,itemattr,item,user_id,num=100):#mark是该用户之前的打分
    attr1,attr2=itemattr[item]
    ans=[]
    for id,score in mark[user_id].items():
        similarity=0
        if id not in itemattr:
            continue
        if attr1!=0 and attr2!=0 and itemattr[id][0]!=0 and itemattr[id][1]!=0:
            similarity=(attr1*itemattr[id][0] + attr2*itemattr[id][1])/(((attr1**2+attr2**2)**0.5)*((itemattr[id][0]**2+itemattr[id][1]**2)**0.5))
        ans.append([id,similarity])
    ans.sort(key=lambda x:x[1],reverse=True)
    if len(ans)>num:
        return ans[:num]
    return ans

def GetData(filepath):
    '''
    读取数据
    '''
    data={}
    with open(filepath,'r')as f:
        line=f.readline()#<user id>|<numbers of rating items>
        while line:
            user_id,num=line.split('|')
            ItemToScore={}#item->score
            for i in range(0,int(num)):
                inline=f.readline()
                item,score=inline.split('  ')
                ItemToScore[int(item)]=int(score)
                #获取最大ID
                global MaxItem_ID
                if int(item)>MaxItem_ID:
                    MaxItem_ID=int(item)
            #获取最大ID
            global MaxUser_ID
            if int(user_id)>MaxUser_ID:
                MaxUser_ID=int(user_id)
            data[int(user_id)]=ItemToScore
            line=f.readline()
            #print('user_id is {}, num is {}'.format(int(user_id),int(num)))
    f.close()
    print('load data finished')
    return data

def split(data,rate=0.2):
    '''
    data: dict
    :return train dev
    返回训练集和验证集
    '''
    seed=10
    dev = {}
    np.random.seed(seed)
    for UserId,ItemToScore in data.items():
        # 每个UserId中随机选取出来的ItemID
        ItemId = np.random.choice(list(ItemToScore.keys()),int(len(ItemToScore)*rate),replace=False)
        # temp dict
        temp = {}
        for i in ItemId:
            # i = key(item)-> socre
            temp[i]=ItemToScore[i]
            # 从训练集中去除
            data[UserId].pop(i)
        dev[UserId]=temp
    return data,dev

def GetMean(data):
    '''
    求均值
    data:dict
    '''
    sum=0
    num=0
    for u_id in data.keys():
        for item in data[u_id].keys():
            sum+=data[u_id][item]
            num+=1
    return sum/num

class SVD:
    '''
    R P Q
    '''
    def __init__(self,R,SVDdev,k,traindata):
        self.R=R
        self.k=k
        self.bi = np.zeros(MaxItem_ID+1)
        self.bu = np.zeros(MaxUser_ID+1)
        self.qi = np.random.rand(MaxItem_ID+1,k)
        self.pu = np.random.rand(MaxUser_ID+1,k)
        # 均值
        self.mean=GetMean(R)
        # 验证集,取10%
        self.dev = SVDdev
        self.traindata=traindata

    def eval(self,epoch,proportion=0.5):
        # 均方误差
        RMSE = 0
        num = 0
        # 绝对误差
        ABSE = 0
        #np.savez("best_result.npz", self.qi, self.pu, self.bi, self.bu)
        params = np.load("best_result.npz")
        self.qi = params['arr_0']
        self.pu = params['arr_1']
        self.bi = params['arr_2']
        self.bu = params['arr_3']
        itemattr = getAttribute()
        for u_id in self.dev:
            for item in self.dev[u_id].keys():
                TrueScore = self.dev[u_id][item]
                predict = np.dot(self.qi[item], self.pu[u_id]) + self.mean + self.bi[item] + self.bu[u_id]
                if item in itemattr:
                    reference = get_related(self.traindata, itemattr, item,u_id)
                    if len(reference)>0 and reference[0][1]>0:
                        predict *= proportion
                        sum = 0
                        for j in reference:
                            sum += j[1]
                        for j in reference:
                            predict += self.traindata[u_id][j[0]] * (j[1] / sum) * (1 - proportion)
                erro = TrueScore - predict
                RMSE += erro **2
                ABSE += abs(erro)
                num += 1
                print(num,u_id,item)
        print("epoch:{}\t均方误差: {}\t 绝对误差{}".format(epoch,np.sqrt(RMSE/num),ABSE/num))
        return np.sqrt(RMSE/num)

    def WriteResult(self,filepath,train,proportion=0.5):
        '''
            写结果
        '''
        # 加载参数
        params = np.load("best_result.npz")
        self.qi = params['arr_0']
        self.pu = params['arr_1']
        self.bi = params['arr_2']
        self.bu = params['arr_3']
        itemattr = getAttribute()
        with open('data/test.txt','r') as f1:
            with open(filepath,'w') as f2:
                line = f1.readline()
                while line:
                    u_id,num = line.split('|')
                    u_id = int(u_id)
                    f2.write(line)
                    for i in range(0,int(num)):
                        inline = f1.readline()
                        item = int(inline)
                        score = np.dot(self.pu[u_id],self.qi[item])+self.bi[item]+self.bu[u_id]+self.mean
                        if item in itemattr:
                            reference=get_related(train,itemattr,item,u_id)
                            if len(reference)>0 and reference[0][1]>0:
                                score*=proportion
                                sum=0
                                for j in reference:
                                    sum+=j[1]
                                for j in reference:
                                    score+=train[u_id][j[0]]*(j[1]/sum)*(1-proportion)
                        f2.write("{}  {}\n".format(item,score))
                    #print(line)
                    line = f1.readline()
if __name__ == '__main__':
    TrainData = GetData("data/train.txt")
    TrainData, dev = split(TrainData)
    traindata = GetData("data/train.txt")
    model = SVD(TrainData, dev, 100, traindata)
    #model.eval(0)
    dev = GetData("data/train.txt")
    model.WriteResult("result_SVD_cousine.txt", dev)

