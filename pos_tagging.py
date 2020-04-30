# pos_tagging.py
import numpy as np
#用于将词性tag转化为id 方便在矩阵中使用id 代替tag的值
tag2id,id2tag={},{}

#用于将单词word转化为id 方便在矩阵中使用id 代替word的值
word2id,id2word={},{}
#文件的位置要根据实际情况自己调整。
with open('traindata.txt','r') as f:
	for x in f:
		l=x.split('/')
		word=l[0]
		tag=l[1].rstrip()#去除换行符号
		if word not in word2id:#如果字典中不存在，将其添加到字典
			word2id[word]=len(word2id)
			id2word[len(id2word)]=word
		if tag not in tag2id:#同上
			tag2id[tag]=len(tag2id)
			id2tag[len(id2tag)]=tag
M=len(word2id)#词典大小 #of words in dictionary
N=len(tag2id)#词性种类个数 #of tags in tag set
# print(M)
# print(tag2id)
#到此 ，词典和id之间的转化工作完成。
#————————————————————————————————————————————————————————————————
# init matrix of word and tag
#初始化A B pi 矩阵
pi = np.zeros(N)# 每个词性出现在句首的概率
A  = np.zeros((N,M))#给定tag i 出现该单词的概率 p(word|tag)
B  = np.zeros((N,N))#状态转化矩阵 

with open('traindata.txt','r') as f:
	prev_tag="."
	for x in f:
		l=x.split('/')
		word=l[0]
		tag=l[1].rstrip()
		#将单词转化为id
		wordid,tagid=word2id[word],tag2id[tag]
		if prev_tag==".":#上一个词性是句号，表示当前单词在句首出现
			pi[tagid]+=1
			A[tagid][wordid]+=1#表示该tag词性下的word_id的单词出现一次。
		else:#表示非句首
			A[tagid][wordid]+=1
			B[tag2id[prev_tag]][tagid]+=1#表示该prev_tag词性下,下一次是tag_id的词性。
		prev_tag=tag

#将词出现的频率 转化为概率 使用add one smoothing进行平滑处理

pi=(pi+1)/(2*sum(pi))
A=A+1
B=B+1
for i in range(N):
	#A和B的每一行都代表一个词性，所以每一行的概率和为1,
	A[i]=(A[i])/(2*sum(A[i]))
	B[i]=(B[i])/(2*sum(B[i]))

#到此 A ，B ，pi 计算完毕
#———————————————————————————————————————————————————————————————————— 
# 用于取对数
def log(v):
    if v == 0:
        return np.log(v+0.000001)
    return np.log(v)

def vitebi(x,A,B,pi):
    """
    x: user input string/sentence: x: "I like playing soccer"
    pi: initial probability of tags
    A: 给定tag, 每个单词出现的概率
    B: tag之间的转移概率
    """
    k=len(word2id)
    ls=[]
    #如果单词词库不存在，则将其id 设置为词库+1，依次累加
    for word in x.split(" "):
        if word2id.__contains__(word):
           ls.append(word2id[word])  # x: [4521, 412, 542 ..]
        else:
            k=k+1
            ls.append(k)
    x=ls
    T = len(x)

    
    dp = np.zeros((T,N))  # dp[i][j]: w1...wi, 假设wi的tag是第j个tag
    ptr = np.array([[0 for x in range(N)] for y in range(T)] ) # T*N
    # TODO: ptr = np.zeros((T,N), dtype=int)

    for j in range(N): # basecase for DP算法
        dp[0][j] = log(pi[j]) + log(A[j][x[0]] if x[0] < len(A[j]) else 0.0000001 )

    for i in range(1,T): # 每个单词
        for j in range(N):  # 每个词性
            dp[i][j] = -9999999
            for k in range(N): # 从每一个k可以到达j 若A中不存在该单词id 则出现概率设置为很小的数
                score = dp[i-1][k] + log(B[k][j]) + log(A[j][x[i]] if x[i] < len(A[j]) else 0.0000001)
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k
    
    # decoding: 把最好的tag sequence 打印出来
    best_seq = [0]*T  # best_seq = [1,5,2,23,4,...]  
    # step1: 找出对应于最后一个单词的词性
    best_seq[T-1] = np.argmax(dp[T-1])
    # step2: 通过从后到前的循环来依次求出每个单词的词性
    for i in range(T-2, -1, -1): # 从T-2 到 0 步长为-1
        best_seq[i] = ptr[i+1][best_seq[i+1]]
    # 到目前为止, best_seq存放了对应于x的 词性序列,下面将其转化回词性内容
    tag_ls=[id2tag[i] for i in best_seq]
    print (tag_ls)
s="I love sport ."  
# s="A full , color page in Newsweek will cost scs ."
vitebi(s,A,B,pi)
'''
A/DT
full/JJ
,/,
four-color/JJ
page/NN
in/IN
Newsweek/NNP
will/MD
cost/VB
$/$
100,980/CD
./.
'''
