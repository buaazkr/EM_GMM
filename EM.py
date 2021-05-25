# EM算法的实现
import numpy as np
from scipy import stats
import  math

#
def em(h, mu1, sigma1, w1, mu2, sigma2):
    N = len(h)  # 样本长度

    # E-step,利用当前参数（mu1,sigma1,mu2,sigma2）以及男生比例w1
    # 计算响应度y1、y2
    w2 = 1 - w1
    p1 = w1 * stats.norm(mu1, sigma1).pdf(h)
    p2 = w2 * stats.norm(mu2, sigma2).pdf(h)
    y1 = p1 / (p1 + p2)
    y2 = p2 / (p1 + p2)
    #L为对数似然函数
    #L = sum(y1) * math.log(w1) + sum(y2) * math.log(w2)+sum(y1 * (np.log(stats.norm(mu1, sigma1).pdf(h))))+ \
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