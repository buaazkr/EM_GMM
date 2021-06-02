import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import random
import math

L_value = []

np.random.seed(5)  # 固定随机数种子（随机函数初始输入值），保证每次运行使用的数据相同
#男女生身高正态分布
#男生：均值：175，标准差：15
#女生：均值：165，标准差：10
male=np.random.normal(175,15,400)
female=np.random.normal(165,10,200)

#将男女生身高随机混合、确保没有任何人为因素影响混合
#使用的600个数据是相同，只是每次都打乱顺序进行实验，结果表明样本的顺序与结果无关
h=[]
temp_num_male = 0
temp_num_female = 0
for i in range(600):
    temp = random.random()
    if temp>0.7:
        if temp_num_male<400:
            h.append(male[temp_num_male])
            temp_num_male = temp_num_male + 1
        else:
            h.append(female[temp_num_female])
            temp_num_female = temp_num_female + 1
    else:
        if temp_num_female<200:
            h.append(female[temp_num_female])
            temp_num_female = temp_num_female + 1
        else:
            h.append(male[temp_num_male])
            temp_num_male = temp_num_male + 1
#保存混合后的数据（每次运行都保存重新保存一次）
f = open('mix_data.txt','w')
for i in range(len(h)):
    f.write(str(h[i]))
    f.write('\n')
f.close()
f = open('male_data.txt','w')
for i in range(len(male)):
    f.write(str(male[i]))
    f.write('\n')
f.close()
f = open('female_data.txt','w')
for i in range(len(female)):
    f.write(str(female[i]))
    f.write('\n')
f.close()

def em(h, mu1, sigma1, w1, mu2, sigma2):
    N = len(h)  # 样本长度

    # E-step,利用当前参数（mu1,sigma1,mu2,sigma2）以及男生比例w1
    # 计算响应度y1、y2
    w2 = 1 - w1
    p1 = w1 * stats.norm.pdf(h, loc=mu1, scale=sigma1)
    p2 = w2 * stats.norm.pdf(h, loc=mu2, scale=sigma2)
    y1 = p1 / (p1 + p2)
    y2 = p2 / (p1 + p2)
    #L为对数似然函数
    # L = sum(y1) * math.log(w1) + sum(y2) * math.log(w2)+sum(y1 * (np.log(stats.norm(mu1, sigma1).pdf(h))))+ \
    #   sum(y2 * (np.log(stats.norm(mu2, sigma2).pdf(h))))
    L = sum(np.log(p1 + p2))
    # M-step
    # 利用响应度，更新mu、sigma与混合概率w1
    mu1 = np.sum(y1 * h) / np.sum(y1)
    mu2 = np.sum(y2 * h) / np.sum(y2)
    sigma1 = np.sqrt(np.sum(y1 * np.square(h - mu1)) / (np.sum(y1)))
    sigma2 = np.sqrt(np.sum(y2 * np.square(h - mu2)) / (np.sum(y2)))
    w1 = np.sum(y1) / N
    print(L)
    return mu1, sigma1, w1, mu2, sigma2, L

#利用EM求解混合高斯分布
#Step 1.对男女生的均值、方差和男女比例进行初始化
mu1_ini=180;sigma1_ini=10;w1_ini=0.5 #男生
mu2_ini=160;sigma2_ini=5 #w2=0.5#女生

mu1 = mu1_ini;sigma1=sigma1_ini;w1=w1_ini
mu2 = mu2_ini;sigma2=sigma2_ini

N = len(h)  # 样本长度
# 开始EM算法的主循环
print(len(stats.norm(mu1, sigma1).pdf(h)))
for iteration in range(500):
    mu1,sigma1,w1,mu2,sigma2,L=em(h,mu1,sigma1,w1,mu2,sigma2)
    L_value.append(L)


#展示计算结果
print("***********INITIAL_VALUE************")
print("男生比例初值：" + str(w1_ini))
print("男生身高平均值初值：" + str(mu1_ini))
print("男生身高标准差初值： " + str(sigma1_ini))
print("女生比例初值： " + str(1 - w1_ini))
print("女生身高平均值初值： " + str(mu2_ini))
print("女生身高标准差初值： " + str(sigma2_ini))
print("*****************EM*****************")
print("男生比例： " + str(w1))
print("男生身高平均值预测结果： " + str(mu1))
print("男生身高标准差预测结果： " + str(sigma1))
print("女生比例： " + str(1 - w1))
print("女生身高平均值预测结果： " + str(mu2))
print("女生身高标准差预测结果： " + str(sigma2))
print("*****************END*****************")

#男生女生以及混合后身高的概率密度曲线
#分别画出男女生身高数据
plt.figure(1)
plt.hist(male,bins=100)
plt.hist(female,bins=100)
plt.title('data')
plt.xlabel('height/cm')
plt.ylabel('num/1')

#展示混合后的数据
plt.figure(2)
plt.hist(h,bins=150)
plt.title('mix data')#混合后概率密度函数
plt.xlabel('height/cm')
plt.ylabel('num/1')

plt.figure(3)
t=np.linspace(120,220,550)#500个
m = stats.norm.pdf(t,loc=mu1, scale=sigma1) # 男生分布的预测结果曲线
f = stats.norm.pdf(t,loc=mu2, scale=sigma2) # 女生分别的预测结果曲线
mix=w1*m+(1-w1)*f#混合后
plt.plot(t, m, color='b')
plt.plot(t, f, color='r')
plt.plot(t, mix, color='k')
#男生女生以及混合后身高的概率密度曲线
plt.title('Probability density curve for mixed height')
plt.legend(["male","female","mix"],loc='upper right')
plt.xlabel('height/cm')
plt.ylabel('Probability density')

plt.figure(4)
plt.plot(L_value[1:len(L_value)])
plt.show()