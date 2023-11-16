# -*- coding: utf-8 -*-
import numpy as np
import random
import os

# 最大item的ID
MaxItem_ID=0
MaxUser_ID=0
def GetData(filepath):
    '''
    读取数据
    '''
    data={}
    '''
    usrId -> small dict
    itemid -> score
    '''
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
    def __init__(self,R: dict[int, dict[int, int]],SVDdev,k=100):
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

    def eval(self,epoch):
        # 均方误差
        RMSE = 0
        num = 0
        # 绝对误差
        ABSE = 0
        for u_id in self.dev:
            for item in self.dev[u_id].keys():
                TrueScore = self.dev[u_id][item]
                predict = np.dot(self.qi[item], self.pu[u_id]) + self.mean + self.bi[item] + self.bu[u_id]
                erro = TrueScore - predict
                RMSE += erro **2
                ABSE += abs(erro)
                num += 1
        print("epoch:{}\t均方误差: {}\t 绝对误差{}".format(epoch,np.sqrt(RMSE/num),ABSE/num))
        return np.sqrt(RMSE/num)

    def train(self,epochs=20,lamada=2e-2,gamma=2e-4):
        for epoch in range(0,epochs):
            for u_id in self.R.keys():
                for item in self.R[u_id].keys():
                    TrueScore=self.R[u_id][item]
                    predict=np.dot(self.qi[item],self.pu[u_id].T)+self.mean+self.bi[item]+self.bu[u_id]
                    #print("prediction:{}\t True:{}".format(predict,TrueScore))
                    erro = TrueScore - predict
                    #更新参数
                    self.bu[u_id]+=gamma*(erro-lamada*self.bu[u_id])
                    self.bi[item]+=gamma*(erro-lamada*self.bi[item])
                    temp_qi=self.qi[item]
                    self.qi[item]+=gamma*(erro*self.pu[u_id]-lamada*self.qi[item])
                    self.pu[u_id]+=gamma*(erro*temp_qi-lamada*self.pu[u_id])
            gamma *= 0.95 #动态学习率
            # 验证集上误差
            self.eval(epoch)

    def WriteResult(self,filepath):
        '''
            写结果
        '''
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
                        f2.write("{}  {}\n".format(item,score))
                    #print(line)
                    line = f1.readline()


if __name__ == '__main__':
    TrainData=GetData("data/train.txt")
    TrainData, dev = split(TrainData)
    model=SVD(TrainData,dev)
    model.train()
    model.WriteResult("result.txt")
