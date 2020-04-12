# 信用评分卡（Learning Note）

## 项目流程
* 数据获取
* 数据预处理
* 探索性分析
* 变量选择
* 模型开发
* 模型评估
* 信用评分
* 评分系统

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/procedure.png" alt="error"/>

## 一、数据获取
### Give Me Some Credit (Kaggle)
```
https://www.kaggle.com/c/GiveMeSomeCredit/data
```

### ***变量说明***
* 基本属性：包括了借款人当时的年龄。
* 偿债能力：包括了借款人的月收入、负债比率。
* 信用往来：两年内35-59天逾期次数、两年内60-89天逾期次数、两年内90天或高于90天逾期的次数。
* 财产状况：包括了开放式信贷和贷款数量、不动产贷款或额度数量。
* 贷款属性：暂无。
* 其他因素：包括了借款人的家属数量（不包括本人在内）。
* 时间窗口：自变量的观察窗口为过去两年，因变量表现窗口为未来两年。

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/data_variable.png" alt="error"/>

## 二、数据预处理
### ***数据集整体情况***
用describe函数获取数据集情况

```
data = pd.read_csv('')
data.describe()
```
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/data_describe.jpg" alt="error"/>

### ***缺失值处理***
#### 1、直接删除元祖
* 这种方法在对象有多个属性缺失值、被删除的含缺失值的对象与信息表中的数据量相比非常小的情况下是非常有效的。
* 缺点
  * ***在信息表中对象很少的情况下会影响到结果的正确性，可能导致数据发生偏离，从而引出错误的结论。***

#### 2、数据补齐
* 人工填写（Filling Manually）
  * 这个方法产生数据偏离最小，是填充效果最好的一种。当数据规模很大、空值很多的时候，该方法是不可行的。
  * ***不适合大数据***

* 特殊值填充（Treating Missing Attribute values as Special values）
  * 将空值作为一种特殊的属性值来处理，它不同于其他的任何属性值。如所有的空值都用“unknown”填充。
  * ***可能导致数据偏离，一般不使用***

* 平均值填充 (Mean/Mode Completer)
  * 将信息表中的属性分为数值属性和非数值属性来分别进行处理。
  * 数值型
    * 根据该属性在其他所有对象的取值的平均值来填充该缺失的属性值
  * 非数型
    * 根据统计学中的众数原理，用该属性在其他所有对象的取值次数最多的值(即出现频率最高的值)来补齐该缺失的属性值。

  * 条件平均值填充法（Conditional Mean Completer）
    * 失属性值的补齐同样是靠该属性在其他对象中的取值求平均得到，但不同的是用于求平均的值并不是从信息表所有对象中取，而是从与该对象具有相同决策属性值的对象中取得。
  
  * 两种方法的出发点
    * 以最大概率可能的取值来补充缺失的属性值，只是在具体方法上有一点不同
  
  * ***与其他方法相比，它是用现存数据的多数信息来推测缺失值。***
  
* 热卡填充（Hot deck imputation，或就近补齐）
  * 对于一个包含空值的对象，热卡填充法在完整数据中找到一个与它最相似的对象，然后用这个相似对象的值来进行填充。不同的问题选用不同的标准来对相似进行判定。
  * ***缺点：难以定义相似标准，主观因素较多。***

* 聚类填充(clustering imputation)
  * 常用方法： K-NN
      * 先根据欧式距离或相关分析来确定距离具有缺失数据样本最近的K个样本，将这K个值加权平均来估计该样本的缺失数据。
      * 假设X=(X1,X2…Xp)为信息完全的变量，Y为存在缺失值的变量，那么首先对X或其子集行聚类，然后按缺失个案所属类来插补不同类的均值。
      * ***如果在以后统计分析中还需以引入的解释变量和Y做分析，那么这种插补方法将在模型中引入自相关，给分析造成障碍。***

      * ```
        fancyimpute 安装（其中包含TensorFlow）
        1、conda install ecos
        2、conda install CVXcanon
        3、pip install fancyimpute
        
        from fancyimpute import KNN
        dataframe = KNN(k=3).fit_transform(dataframe)
    
        ```

* 使用所有可能的值填充（Assigning All Possible values of the Attribute）
  * 这种方法是用空缺属性值的所有可能的属性取值来填充，能够得到较好的补齐效果。
  * ***当数据量很大或者遗漏的属性值较多时，其计算的代价很大，可能的测试方案很多。***

* 组合完整化方法（Combinatorial Completer）
  * 这种方法是用空缺属性值的所有可能的属性取值来试，并从最终属性的约简结果中选择最好的一个作为填补的属性值。这是以约简为目的的数据补齐方法，能够得到好的约简结果；但是，当数据量很大或者遗漏的属性值较多时，其计算的代价很大。
  * 条件组合完整化方法（Conditional Combinatorial Complete）
    * 填补遗漏属性值的原则是一样的，不同的只是从决策相同的对象中尝试所有的属性值的可能情况，而不是根据信息表中所有对象进行尝试。条件组合完整化方法能够在一定程度上减小组合完整化方法的代价。
    * ***在信息表包含不完整数据较多的情况下，可能的测试方案将巨增。***

* 回归（Regression）
  * 基于完整的数据集，建立回归方程（模型）。对于包含空值的对象，将已知属性值代入方程来估计未知属性值，以此估计值来进行填充。
  * ***当变量不是线性相关或预测变量高度相关时会导致有偏差的估计。***
  
  * 常用方法： 随机树林（Random Forest）
    * ```
      from sklearn.ensemble import RandomForestRegressor
      rfr = RandomForestRegressor()
      rfr.fit(X, Y) # X: data Y: label
      predicted = rfr.predict(test_data)
      ```

* 极大似然估计（Max Likelihood ，ML）
  * 常用方法： EM
  * 在缺失类型为随机缺失的条件下，假设模型对于完整的样本是正确的，通过观测数据的边际分布可以对未知参数进行极大似然估计。
  * ***它一个重要前提：适用于大样本。有效样本的数量足够以保证ML估计值是渐近无偏的并服从正态分布。***
  * ***这种方法可能会陷入局部极值，收敛速度也不是很快，并且计算很复杂。***

* 多重插补（Multiple Imputation，MI）
  * 多值插补的思想来源于贝叶斯估计，认为待插补的值是随机的，它的值来自于已观测到的值。具体实践上通常是估计出待插补的值，然后再加上不同的噪声，形成多组可选插补值。根据某种选择依据，选取最合适的插补值。
  * 3个步骤
    * （1）为每个空值产生一套可能的插补值，这些值反映了无响应模型的不确定性；每个值都可以被用来插补数据集中的缺失值，产生若干个完整数据集合。
    * （2）每个插补数据集合都用针对完整数据集的统计方法进行统计分析。
    * （3）对来自各个插补数据集的结果，根据评分函数进行选择，产生最终的插补值。

  * 多重插补和贝叶斯估计的思想是一致的，但是多重插补弥补了贝叶斯估计的几个不足。
    * （1）贝叶斯估计以极大似然的方法估计，极大似然的方法要求模型的形式必须准确，如果参数形式不正确，将得到错误得结论，即先验分布将影响后验分布的准确性。而多重插补所依据的是大样本渐近完整的数据的理论，在数据挖掘中的数据量都很大，先验分布将极小的影响结果，所以先验分布的对结果的影响不大。
    * （2）贝叶斯估计仅要求知道未知参数的先验分布，没有利用与参数的关系。而多重插补对参数的联合分布作出了估计，利用了参数间的相互关系。

  * ***多重插补保持了单一插补的两个基本优点***
    * （1）应用完全数据分析方法
    * （2）融合数据收集者知识的能力

  * ***相对于单一插补，多重插补有3个极其重要的优点***
    * （1）为表现数据分布，随机抽取进行插补，增加了估计的有效性。
    * （2）当多重插补是在某个模型下的随机抽样时，按一种直接方式简单融合完全数据推断得出有效推断，即它反映了在该模型下由缺失值导致的附加变异。
    * （3）在多个模型下通过随机抽取进行插补，简单地应用完全数据方法，可以对无回答的不同模型下推断的敏感性进行直接研究。

  * ***多重插补的3个缺点***
    * （1）生成多重插补比单一插补需要更多工作
    * （2）贮存多重插补数据集需要更多存储空间
    * （3）分析多重插补数据集比单一插补需要花费更多精力

#### 3、不处理
* 直接在包含空值的数据上进行数据挖掘。
* 这类方法包括贝叶斯网络和人工神经网络等。

### ***异常值处理***
异常值，即在数据集中存在不合理的值，又称离群点。

#### 1、判断方法
* 简单统计分析
  * 对属性值进行一个描述性的统计（规定范围），从而查看哪些值是不合理的（范围以外的值）。

* 3δ原则
  * 数据服从正态分布：根据正态分布的定义可知，距离平均值3δ之外的概率为 P(|x-μ|>3δ) <= 0.003 ，这属于极小概率事件，在默认情况下我们可以认定，距离超过平均值3δ的样本是不存在的。因此，当样本距离平均值大于3δ，认为该样本为异常值。
  * 根据概率值的大小可以判断 x 是否属于异常值。

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/exception_value_1.png" alt="error"/>

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/exception_value_2.png" alt="error"/>

* 使用距离检测多元离群点
  * 当数据不服从正态分布时，可以通过远离平均距离多少倍的标准差来判定，多少倍的取值需要根据经验和实际情况来决定。

#### 2、处理方法
* 删除含有异常值的记录
* 将异常值视为缺失值，使用缺失值处理方法来处理
* 用平均值来修正
* 不处理

### ***数据切分***
* 将数据分为训练集和验证集
```
# X: data, Y: label, test_size: 验证集比例
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

```

## 三、探索性分析（Exploratory Data Analysis）
* 在建立模型之前，我们一般会对现有的数据进行 探索性数据分析
* EDA是指对已有的数据(特别是调查或观察得来的原始数据)在尽量少的先验假定下进行探索。
* 常用方法
  * 直方图
  * 散点图
  * 箱线图

* 客户年龄和收入的分布图

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/EDA.png" alt="error"/>

## 四、变量筛选
* 好的变量（特征）选择能够提升模型的性能，更能帮助我们理解数据的特点、底层结构，这对进一步改善模型、算法都有着重要作用。
* 变量筛选需要考虑的因素
  * ***变量的预测能力***
  * ***变量之间的线性相关性***
  * 变量的简单性（容易生成和使用）
  * 变量的强壮性（不容易被绕过）
  * 变量在业务上的可解释性（被挑战时可以解释的通）
  * 等等

### ***分箱处理（Binning）***
* 变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。
* 分箱的几种方法
  * 1、无监督分箱
    * (1) 等频分箱：把自变量按从小到大的顺序排列，根据自变量的个数等分为k部分，每部分作为一个分箱。
    * (2) 等距分箱：把自变量按从小到大的顺序排列，将自变量的取值范围分为k个等距的区间，每个区间作为一个分箱。
    * (3) 聚类分箱：用k-means聚类法将自变量聚为k类，但在聚类过程中需要保证分箱的有序性。

    * ***由于无监督分箱仅仅考虑了各个变量自身的数据结构，并没有考虑自变量与目标变量之间的关系，因此无监督分箱不一定会带来模型性能的提升。***

  * 2、有监督分箱
    * （1）Split 分箱
      * 一种自上而下(即基于分裂)的数据分段方法。如下图所示，Split 分箱和决策树比较相似，切分点的选择指标主要有 entropy，gini 指数和 IV 值等。

      <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/split_binning.jpg" alt="error"/>
  
    * （2）Chimerge 分箱（卡方分箱）
      * 一种自底向上(即基于合并)的数据离散化方法。
      * 其基本思想是如果两个相邻的区间具有类似的类分布，则这两个区间合并；否则，它们应保持分开。
      * Chimerge通常采用卡方值来衡量两相邻区间的类分布情况。

      <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/chimerge_binning.jpg" alt="error"/>

    * （3）Monotonic Binning
      * 要求各组的单调事件率呈单调。
      * ```
        https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm
        ```

* 代码
```
# https://github.com/ShichenXie/scorecardpy
# y: label, method: tree or chimerge

from binning.woebin import *
cutoff = woebin(data, y, method='tree')
```

***WOE (Weight of Evidence)***

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/WOE.jpg" alt="error"/>

* WOE编码的优势
  * 可提升模型的预测效果
  * 将自变量规范到同一尺度上
  * WOE能反映自变量取值的贡献情况
  * 有利于对变量的每个分箱进行评分
  * 转化为连续变量之后，便于分析变量与变量之间的相关性
  * 与独热向量编码相比，可以保证变量的完整性，同时避免稀疏矩阵和维度灾难

***单变量筛选***
* 单变量的筛选基于变量预测能力
* 常用方法
  * 基于IV值的变量筛选（代码中使用）
  * 基于stepwise的变量筛选
  * 基于特征重要度的变量筛选：RF, GBDT…
  * 基于LASSO正则化的变量筛选

## Reference
#### 1、项目
```
https://www.jianshu.com/p/f931a4df202c
https://www.jianshu.com/p/2759e090bd53?t=123
https://zhuanlan.zhihu.com/p/36539125 (分箱等等）
```
#### 2、缺失值处理
```
https://blog.csdn.net/xiedelong/article/details/81607598
https://blog.csdn.net/w352986331qq/article/details/78639233
```
#### 3、Random Forest
```
https://blog.csdn.net/qq_34106574/article/details/82016442
```
#### 4、特征选择
```
https://www.zhihu.com/question/28641663?sort=created
```
#### 5、分箱
```
https://github.com/ShichenXie/scorecardpy (tree and chimerge)
https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm (Monotonic-WOE-Binning)
```