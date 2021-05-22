import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import EM
import random

np.random.seed(5)  # 固定随机数种子（随机函数初始输入值），保证每次运行使用的数据相同
#男女生身高正太分布
#男生：均值：175，标准差：15
#女生：均值：165，标准差：10
male=np.random.normal(175,15,400)
female=np.random.normal(165,10,200)

#将男女生身高随机混合、确保没有任何人为因素影响混合
h=[]
temp_num_male = 0;
temp_num_female = 0;
for i in range(600):
    temp = random.random();
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
#保存混合后的数据
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

#利用EM求解混合高斯分布
#Step 1.对男女生的均值、方差和男女比例进行初始化
mu1=180;sigma1=10;w1=0.5#男生
mu2=160;sigma2=5;w2=0.5#女生

N = len(h)  # 样本长度
# 开始EM算法的主循环
print(len(stats.norm(mu1, sigma1).pdf(h)))
for iteration in range(500):
    mu1,sigma1,w1,mu2,sigma2,w2=EM.em(h,mu1,sigma1,w1,mu2,sigma2,w2)

#展示计算结果
print("男生比例： " + str(w1))
print("男生身高平均值预测结果" + str(mu1))
print("男生身高标准差预测结果： " + str(sigma1))
print("女生比例： " + str(w2))
print("女生身高平均值预测结果" + str(mu2))
print("女生身高标准差预测结果： " + str(sigma2))

#男生女生以及混合后身高的概率密度曲线
#分别画出男女生图像
plt.figure(1)
plt.hist(male,bins=100)
plt.hist(female,bins=100)
plt.title('data')
plt.xlabel('height/cm')
plt.ylabel('num/1')

#展示混合后的数据、确保混合均匀
# h=np.array(h)
# h1=pd.Series(h).hist(bins=150)
# h1.plot()
plt.figure(2)
plt.hist(h,bins=150)
plt.title('mix data')#混合后概率密度函数
plt.xlabel('height/cm')
plt.ylabel('num/1')

plt.figure(3)
t=np.linspace(120,220,550)#500个
m = stats.norm.pdf(t,loc=mu1, scale=sigma1) # 男生分布的预测
f = stats.norm.pdf(t,loc=mu2, scale=sigma2) # 女生分别的预测
mix=w1*m+w2*f#混合后
plt.plot(t, m, color='b')
plt.plot(t, f, color='r')
plt.plot(t, mix, color='k')
#男生女生以及混合后身高的概率密度曲线
plt.title('Probability density curve for mixed height')
plt.legend(["male","female","mix"],loc='upper right')
plt.xlabel('height/cm')
plt.ylabel('Probability density')
plt.show()