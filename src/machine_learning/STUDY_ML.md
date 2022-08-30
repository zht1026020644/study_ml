# 入门
## python package
* Numpy：做数值运算：FFT/Gauss/LSQ/SVD
* pandas: DataFrame/Series:读取Excel/csv/tsv,封装了Numpy
* Scipy:科学运算，Gamma/Comb
* matplotib:画图
* scikit-learn:ML
* tensorflow(Keras封装了)/pytorch/Caffe:DL
* 安装
    * pip install xxx
## 机器学习
### 监督学习
#### 回归
目标值y是连续值
##### 线性回归
* 高斯分布
* 最大似然估计MLE
* 最小二乘

**过拟合解决办法**
加入正则项：
* ridge 二阶正则项
* lasso 绝对值正则项（降维）
##### 逻辑回归（Logistic，2分类）
* 分类问题的首选算法
#### 分类
目标值y是离散值
* 多分类：softmax回归
##### 技术点
* 梯度下降
   * 随机梯度下降法（每个样本梯度）
   * mini-batch梯度下降
解决求解矩阵解析逆的问题
* 最大似然估计
* 特征选择
