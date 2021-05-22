# EM算法的实现
import numpy as np
from scipy import stats


#
def em(h, mu1, sigma1, w1, mu2, sigma2, w2):
    N = len(h)  # 样本长度

    # E-step,利用现有参数（mu1,sigma1,mu2,sigma2）以及两个分布的混合概率w1、w2
    # 实际上w1+w2=1
    # 计算响应度y1、y2
    p1 = w1 * stats.norm(mu1, sigma1).pdf(h)
    p2 = w2 * stats.norm(mu2, sigma2).pdf(h)
    y1 = p1 / (p1 + p2)
    y2 = p2 / (p1 + p2)

    # M-step
    # 利用响应度，更新mu、sigma与混合概率w1、w2
    mu1 = np.sum(y1 * h) / np.sum(y1)
    mu2 = np.sum(y2 * h) / np.sum(y2)
    sigma1 = np.sqrt(np.sum(y1 * np.square(h - mu1)) / (np.sum(y1)))
    sigma2 = np.sqrt(np.sum(y2 * np.square(h - mu2)) / (np.sum(y2)))
    w1 = np.sum(y1) / N
    w2 = np.sum(y2) / N

    return mu1, sigma1, w1, mu2, sigma2, w2