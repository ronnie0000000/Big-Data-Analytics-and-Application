# -*- coding: utf-8 -*-
"""
@author: 李昱桥 2012797
"""
import numpy as np
import random
import os
import math
import copy

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
            attr1=int(attr1) if (attr1[0]!='N') else 0
            attr2=int(attr2) if (attr2[0]!='N') else 0
            norm=(attr1**2+attr2**2)**0.5
            itemattr[item]=[attr1,attr2,norm]
            #这样构造了一个字典，从item映射到其属性1，属性2，L2范数
            line=f.readline()
    f.close()
    return itemattr

def GetData(filepath):
    '''
    读取数据
    '''
    data={}
    rw_item_user={}
    with open(filepath,'r')as f:
        line=f.readline()#<user id>|<numbers of rating items>
        while line:
            user_id,num=line.split('|')
            user_id=int(user_id)
            num=int(num)
            ItemToScore={}#item->score
            for i in range(0,int(num)):
                inline=f.readline()
                item,score=inline.split('  ')
                item=int(item)
                score=int(score)
                ItemToScore[int(item)]=int(score)
                if item in rw_item_user:
                    rw_item_user[item][user_id]=int(score)
                else:
                    rw_item_user[item]={}
                    rw_item_user[item][user_id]=int(score)
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
    rw_user_item=copy.deepcopy(data)
    return data,rw_user_item,rw_item_user

def split(data,rate=0.001):
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

class CF:
    #初始化传入被划分好的训练集，item的属性集合，未被划分的用户对item的打分以及item
    def __init__(self,train_data,itemattribute,raw_user_item,raw_item_user,dev):
        self.itemattribute=itemattribute
        self.train_data=train_data
        self.user2item=raw_user_item
        self.item2user=raw_item_user
        self.all_mean=0
        self.user2item_bias={}
        self.item2user_bias={}
        self.dev=dev
        self.res={}

    def get_bias(self):
        all_num=0
        for key,value in self.user2item.items():
            all_num+=len(value.values())
            self.all_mean+=sum(value.values())
        self.all_mean/=all_num
        for key,value in self.user2item.items():
            avg=sum(value.values())/len(value.values())
            avg-=self.all_mean
            self.user2item_bias[key]=avg
        for key,value in self.item2user.items():
            avg = sum(value.values()) / len(value.values())
            avg -= self.all_mean
            self.item2user_bias[key] = avg

    def eval(self,ans):
        # 均方误差
        RMSE = 0
        num = 0
        # 绝对误差
        ABSE = 0
        for u_id,item_and_score in ans.items():
            for item,score in item_and_score.items():
                erro = score - self.user2item[u_id][item]
                RMSE += erro **2
                ABSE += abs(erro)
                num += 1
        print("均方误差: {}\t 绝对误差{}".format(np.sqrt(RMSE/num),ABSE/num))
        return np.sqrt(RMSE/num)

    def train(self):
        train_res={}
        num=0
        for use_id, item_and_score in self.dev.items():
            for item_id, item_score in item_and_score.items():
                related_item={}
                for marked_item, marked_item_score in self.user2item[use_id].items():
                    if item_id not in self.itemattribute or marked_item not in self.itemattribute:
                        item_attribute_similarity=0
                    elif self.itemattribute[item_id][2]==0 or self.itemattribute[marked_item][2]==0:
                        item_attribute_similarity=0
                    else:
                        item_attribute_similarity=(self.itemattribute[item_id][0]*self.itemattribute[marked_item][0]+\
                                                    self.itemattribute[item_id][1]*self.itemattribute[marked_item][1])\
                                                    /(self.itemattribute[item_id][2]*self.itemattribute[marked_item][2])
                    norm1=0
                    norm2=0
                    num_sum=0
                    person_similarity=0
                    for mark_user,mark_user_score in self.item2user[item_id].items():
                        if mark_user in self.item2user[marked_item]:
                            num_sum+=1
                            factor1=self.user2item[mark_user][item_id]-self.all_mean-self.item2user_bias[item_id]
                            factor2=self.user2item[mark_user][marked_item]-self.all_mean-self.item2user_bias[marked_item]
                            person_similarity+=factor1*factor2
                            norm1+=factor1**2
                            norm2+=factor2**2
                    item_attribute_similarity+=0 if (num_sum<20 or norm1*norm2==0) else person_similarity/((norm2*norm1)**0.5)
                    related_item[marked_item]=[item_attribute_similarity,marked_item_score]
                related_item=zip(related_item.keys(),related_item.values())
                related_item=sorted(related_item,reverse=True,key=lambda x: x[1][0])
                if len(related_item)>100:
                    related_item=related_item[:100]
                all_similarity=0
                score=0
                for i in related_item:#[item_id,[similarity,score]]
                    if i[0] not in self.item2user_bias:
                        self.item2user_bias[i[0]]=0
                    score+=i[1][0]*(i[1][1]-self.all_mean-self.item2user_bias[i[0]]-self.user2item_bias[use_id])
                    all_similarity+=i[1][0]
                score=0 if all_similarity==0 else score/all_similarity
                score+=self.all_mean+self.user2item_bias[use_id]#可以换成item
                score=min(100,max(score,0))
                if use_id not in train_res:
                    train_res[use_id]={}
                train_res[use_id][item_id]=score
                num+=1
                if num%1000==0:
                    print("now completed:",num)
        print("RMSE:",self.eval(train_res))


#这里的proportion为SVD预测值在最终结果中的权重
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
                        item_id = int(inline)
                        related_item = {}
                        for marked_item, marked_item_score in self.user2item[u_id].items():
                            if item_id not in self.itemattribute or marked_item not in self.itemattribute:
                                item_attribute_similarity = 0
                            elif self.itemattribute[item_id][2] == 0 or self.itemattribute[marked_item][2] == 0:
                                item_attribute_similarity = 0
                            else:
                                item_attribute_similarity = (self.itemattribute[item_id][0] *self.itemattribute[marked_item][0] +\
                                                             self.itemattribute[item_id][1] *self.itemattribute[marked_item][1]) \
                                                            / (self.itemattribute[item_id][2] *self.itemattribute[marked_item][2])
                            norm1 = 0
                            norm2 = 0
                            num_sum = 0
                            person_similarity = 0
                            if item_id in self.item2user:
                                for mark_user, mark_user_score in self.item2user[item_id].items():
                                    if mark_user in self.item2user[marked_item]:
                                        num_sum += 1
                                        factor1 = self.user2item[mark_user][item_id] - self.all_mean - self.item2user_bias[item_id]
                                        factor2 = self.user2item[mark_user][marked_item] - self.all_mean - \
                                                  self.item2user_bias[marked_item]
                                        person_similarity += factor1 * factor2
                                        norm1 += factor1 ** 2
                                        norm2 += factor2 ** 2
                            item_attribute_similarity += 0 if (num_sum < 20 or norm1 * norm2 == 0) else person_similarity / ((norm2 * norm1) ** 0.5)
                            related_item[marked_item] = [item_attribute_similarity, marked_item_score]
                        related_item = zip(related_item.keys(), related_item.values())
                        related_item = sorted(related_item, reverse=True, key=lambda x: x[1][0])
                        if len(related_item) > 100:
                            related_item = related_item[:100]
                        all_similarity = 0
                        score = 0
                        for i in related_item:  # [item_id,[similarity,score]]
                            if i[0] not in self.item2user_bias:
                                self.item2user_bias[i[0]] = 0
                            score += i[1][0] * (i[1][1] - self.all_mean - self.item2user_bias[i[0]] - self.user2item_bias[u_id])
                            all_similarity += i[1][0]
                        score = 0 if all_similarity == 0 else score / all_similarity
                        score += self.all_mean + self.user2item_bias[u_id]  # 可以换成item
                        score = min(100, max(score, 0))
                        f2.write("{}  {}\n".format(item_id,score))
                    #print(line)
                    line = f1.readline()


if __name__ == '__main__':
    TrainData,raw_user_item,raw_item_user=GetData("data/train.txt")
    itemattribute=getAttribute()
    TrainData, dev = split(TrainData)
    model=CF(TrainData,itemattribute,raw_user_item,raw_item_user,dev)
    model.get_bias()

    model.train()
    #model.WriteResult("result.txt")
