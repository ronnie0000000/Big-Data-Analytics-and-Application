relation=[]#source,degree,[destination]
all_node=set()
block_num=10#即630个节点对应一个block,一共是6263个节点
beta=0.85
ans=[]
all_node_list=[]

def search_in_relation(node):
    for i in range(len(relation)):
        if relation[i][0]==node:
            return i
    return -1

def load_data():#加载数据，之后处理dead end，在计算中直接考虑计算spider
#节点从3~8297(并非所有编号都有对应节点)
    max_num=-1
    print("loading data")
    f=open("Data.txt")
    min_num=200
    for line in f:
        linee=line.strip()
        node,node1=linee.split()
        s=eval(node)#start
        e=eval(node1)#end
        max_num=max(max_num,s,e)
        min_num=min(min_num,s,e)
        all_node.add(s)
        all_node.add(e)
        flag=search_in_relation(s)
        if flag==-1:
            temp=[s,1,[e]]
            relation.append(temp)
        else:
            if e not in relation[flag][2]:
                relation[flag][1] += 1
                relation[flag][2].append(e)
    relation.sort(key=lambda x:x[0])
    temp=[i for i in all_node]
    for i in all_node:
        if search_in_relation(i)==-1:
            relation.append([i,len(all_node),temp])
    relation.sort(key=lambda x: x[0])
    #对每个节点的终点进行排序
    for i in range(len(relation)):
        relation[i][2].sort()
    return max_num

def matrix_transform():
    all_node_list=list(all_node)
    temp_nodelist=[]
    #将标号转化到0~6262的范围中
    temp_all=[i for i in range(6263)]
    for i in range(len(relation)):
        if relation[i][1]==6263:
            temp_nodelist.append([all_node_list.index(relation[i][0]),6263,temp_all])
            continue

        temp_nodelist.append([0,relation[i][1],[]])
        temp_nodelist[-1][0]=all_node_list.index(relation[i][0])
        for j in range(len(relation[i][2])):
            temp_nodelist[-1][2].append(all_node_list.index(relation[i][2][j]))
    real_relation=temp_nodelist
    #转化到稀疏矩阵
    #提前把每个block的节点对应的情况加到文件中
    for i in range(block_num):
        temp_relation=[]
        low_bound=i*630
        high_bound=(i+1)*630-1
        for j in real_relation:
            flag=False
            for k in j[2]:
                if low_bound<=k<=high_bound:
                    if flag==False:
                        temp_relation.append([j[0],j[1],[k]])
                        flag=True
                    else:
                        temp_relation[-1][2].append(k)
        #temp_relation写入
        file_name="middle_result/graph/"+str(i)+".txt"
        with open(file_name,"w") as f:
            for j in range(len(temp_relation)):
                text=str(temp_relation[j][0])+" "+str(temp_relation[j][1])
                for k in temp_relation[j][2]:
                    text+=" "+str(k)
                text+="+"
                f.write(text)
    return all_node_list
def init():
    for i in range(block_num-1):
        text=str(1/6263)
        text+=(" "+str(1/6263))*(630-1)
        file_name="middle_result/middle_result/"+str(i)+".txt"
        with open(file_name,"w") as f:
            f.write(text)
    text=str(1/6263)
    text+=(" "+str(1/6263))*(6263%630-1)
    file_name = "middle_result/middle_result/" + str(block_num-1) + ".txt"
    with open(file_name,"w") as f:
        f.write(text)

def calculate_result():
    for i in range(block_num):
        source_name="middle_result/middle_result/"+str(i)+".txt"
        des_name="middle_result/temp_result/"+str(i)+".txt"
        f=open(source_name)
        text=""
        for line in f:
            text+=line
        with open(des_name,"w") as w:
            w.write(text)
    return_value = 0
    for i in range(block_num):#大循环，对每一个被分组的块，重建关系图
        relation_graph=[]
        file_name="middle_result/graph/"+str(i)+".txt"
        f=open(file_name)
        for line in f:
            temp=line.split("+")
            for j in temp:
                k=j.split()
                if len(j)<=2:#避开结尾的一行空白
                    continue
                relation_graph.append([eval(k[0]),eval(k[1]),[]])
                for p in range(2,len(k)):
                    relation_graph[-1][2].append(int(k[p]))
        #把对应值的位置放好
        if i<9:
            newres=[0]*630
        else:
            newres=[0]*(6263%630)
        for j in range(block_num):#因为对一块的更新需要遍历所有的块
            filename = "middle_result/temp_result/" + str(j) + ".txt"
            f = open(filename)
            for line in f:
                result = line.split()
            summ=0
            for k in range(len(result)):
                summ+=float(result[k])
            for w in range(len(newres)):
                newres[w] += (1 - beta) * float(summ) / 6263
            for k in range(len(result)):#注意查一下最后一个！！！！！！！
                index=k+630*j#实际对应的在图中的点
                index_in_graph=-1
                #找到这个index在关系图中是否存在
                for w in range(len(relation_graph)):
                    if int(relation_graph[w][0])==index:
                        index_in_graph=w
                        break
                if index_in_graph!=-1:
                    for h in relation_graph[index_in_graph][2]:
                        newres[h%630]+=beta*float(result[k])/relation_graph[index_in_graph][1]
        #计算结束进行回写
        text=""
        text+=str(newres[0])
        for j in range(1,len(newres)):
            text+=" "+str(newres[j])
        file_name = "middle_result/middle_result/" + str(i) + ".txt"
        with open(file_name,"w") as w:
            w.write(text)
        filename = "middle_result/temp_result/" + str(i) + ".txt"
        f = open(filename)
        for line in f:
            oldres = line.split()
        for j in range(len(oldres)):
            return_value+=abs(float(oldres[j])-newres[j])
    return return_value

def get_result(w=False):
    ans=[]
    for i in range(block_num):
        file_name = "middle_result/middle_result/" + str(i) + ".txt"
        f=open(file_name)
        for line in f:
            result=line.split()
            for j in range(len(result)):
                if (j+i*630)>6262:
                    break
                ans.append([j+i*630,float(result[j])])
        ans.sort(key=lambda x: x[1],reverse=True)
        ans=ans[0:100]
    print(ans[0:100])
    if not w:
        return
    text = str(all_node_list[ans[0][0]]) + " " + str(ans[0][1])
    for i in range(1, 100):
        text += "\n" + str(all_node_list[ans[i][0]]) + " " + str(ans[i][1])
    with open("result.txt", "w") as w:
        w.write(text)
    print(text)

if __name__ == '__main__':
    print("Start Pagerank")
    max_node=load_data()
    #print(max_node)
    all_node_list=matrix_transform()
    init()
    while True:
        temp=calculate_result()
        print(temp)
        get_result()
        if temp<=1e-3:
            break
    get_result(True)