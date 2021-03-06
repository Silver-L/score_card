# 信用评分卡（Learning Note）
此处为信用评分卡学习笔记，大致总结了评分卡的基础知识。感谢各位大佬的文章！

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

<div align=center>
    <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/data_variable.png" alt="error"/>
</div>


## 二、数据预处理
### ***2.1、数据集整体情况***
用describe函数获取数据集情况

```
data = pd.read_csv('')
data.describe()
```
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/data_describe.jpg" alt="error"/>

### ***2.2、缺失值处理***
#### 2.2.1、直接删除元祖
* 这种方法在对象有多个属性缺失值、被删除的含缺失值的对象与信息表中的数据量相比非常小的情况下是非常有效的。
* 缺点
  * ***在信息表中对象很少的情况下会影响到结果的正确性，可能导致数据发生偏离，从而引出错误的结论。***

#### 2.2.2、数据补齐
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

#### 2.2.3、不处理
* 直接在包含空值的数据上进行数据挖掘。
* 这类方法包括贝叶斯网络和人工神经网络等。

### ***2.3、异常值处理***
异常值，即在数据集中存在不合理的值，又称离群点。

#### 2.3.1、判断方法
* 简单统计分析
  * 对属性值进行一个描述性的统计（规定范围），从而查看哪些值是不合理的（范围以外的值）。

* 3δ原则
  * 数据服从正态分布：根据正态分布的定义可知，距离平均值3δ之外的概率为 P(|x-μ|>3δ) <= 0.003 ，这属于极小概率事件，在默认情况下我们可以认定，距离超过平均值3δ的样本是不存在的。因此，当样本距离平均值大于3δ，认为该样本为异常值。
  * 根据概率值的大小可以判断 x 是否属于异常值。

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/exception_value_1.png" width="700" height="324" alt="error"/>

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/exception_value_2.png" width="700" height="284" alt="error"/>

* 使用距离检测多元离群点
  * 当数据不服从正态分布时，可以通过远离平均距离多少倍的标准差来判定，多少倍的取值需要根据经验和实际情况来决定。

#### 2.3.2、处理方法
* 删除含有异常值的记录
* 将异常值视为缺失值，使用缺失值处理方法来处理
* 用平均值来修正
* 不处理

#### 2.3.3、Kaggle数据集的试验结果
* 变量NumberOfTime30-59DaysPastDueNotWorse， NumberOfTimes90DaysLate，NumberOfTime60-89DaysPastDueNotWorse中包含的异常值
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/exception_value_3.png" alt="error"/>

* 本次采用方法：删除

### ***2.4、数据切分***
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

### ***4.1、分箱处理（Binning）***
* 变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。
* 分箱的几种方法
  * 4.1.1、无监督分箱
    * (1) 等频分箱：把自变量按从小到大的顺序排列，根据自变量的个数等分为k部分，每部分作为一个分箱。
    * (2) 等距分箱：把自变量按从小到大的顺序排列，将自变量的取值范围分为k个等距的区间，每个区间作为一个分箱。
    * (3) 聚类分箱：用k-means聚类法将自变量聚为k类，但在聚类过程中需要保证分箱的有序性。

    * ***由于无监督分箱仅仅考虑了各个变量自身的数据结构，并没有考虑自变量与目标变量之间的关系，因此无监督分箱不一定会带来模型性能的提升。***

  * 4.1.2、有监督分箱
    * （1）Split 分箱
      * 一种自上而下(即基于分裂)的数据分段方法。如下图所示，Split 分箱和决策树比较相似，切分点的选择指标主要有 entropy，gini 指数和 IV 值等。
      <div align=center>
        <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/split_binning.jpg" alt="error"/>
      </div>

    * （2）Chimerge 分箱（卡方分箱）
      * 一种自底向上(即基于合并)的数据离散化方法。
      * 其基本思想是如果两个相邻的区间具有类似的类分布，则这两个区间合并；否则，它们应保持分开。
      * Chimerge通常采用卡方值来衡量两相邻区间的类分布情况。

      <div align=center>
        <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/chimerge_binning.jpg" alt="error"/>
      </div>

    * （3）Monotonic Binning
      * 要求各组的单调事件率呈单调。
      * ```
        https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm
        ```

* 代码
```
# https://github.com/ShichenXie/scorecardpy
# y: label, method: tree or chimerge

import scorecardpy as sc
cutoff = sc.woebin(data, y, method='tree')
```

### ***4.2、WOE (Weight of Evidence)***

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/WOE.jpg" width="700" height="823" alt="error"/>

* WOE编码的优势
  * 可提升模型的预测效果
  * 将自变量规范到同一尺度上
  * WOE能反映自变量取值的贡献情况
  * 有利于对变量的每个分箱进行评分
  * 转化为连续变量之后，便于分析变量与变量之间的相关性
  * 与独热向量编码相比，可以保证变量的完整性，同时避免稀疏矩阵和维度灾难

### ***4.3、单变量筛选***
* 单变量的筛选基于变量预测能力
* 常用方法
  * 基于IV值的变量筛选（代码中使用）
  * 基于stepwise的变量筛选
  * 基于特征重要度的变量筛选：RF, GBDT…
  * 基于LASSO正则化的变量筛选

#### 4.3.1、基于IV值的变量筛选
* IV称为信息价值(information value)，是目前评分卡模型中筛选变量最常用的指标之一。
* ***自变量的IV值越大，表示自变量的预测能力越强***
* 类似的指标还有信息增益、基尼(gini)系数等。

* 常用判断标准
  * <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/IV_1.jpg" width="700" height="168" alt="error"/>

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/IV_2.jpg" width="800" height="660" alt="error"/>

* WOE和IV值的特点
  * ***注： 此处使用的WOE计算公式中，坏客户为分母，好客户为分子***
  * 当前分箱中，坏客户占比越大，WOE值越大
  * 当前分箱中WOE的正负，由当前分箱中好坏客户比例，与样本整体好坏客户比例的大小关系决定
  * WOE的取值范围是(-∞,+∞)，当分箱中好坏客户比例等于整体好坏客户比例时，WOE为0。
  * 对于变量的一个分箱，这个分组的好坏客户比例与整体好坏客户比例相差越大，IV值越大，否则，IV值越小。
  * IV值的取值范围是[0,+∞)，当分箱中只包含好客户或坏客户时，IV = +∞，当分箱中好坏客户比例等于整体好坏客户比例时，IV为0。

* kaggle数据集的IV值计算结果
  * DebtRatio、MonthlyIncome、NumberOfOpenCreditLinesAndLoans、NumberRealEstateLoansOrLines和NumberOfDependents变量的IV值明显较低，所以予以删除。

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/IV.png" alt="error"/>

#### 4.3.2、基于stepwise的变量筛选
* 基于基于stepwise的变量筛选方法也是评分卡中变量筛选最常用的方法之一。
* 包括三种筛选变量的方式
  * （1）前向选择forward：逐步将变量一个一个放入模型，并计算相应的指标，如果指标值符合条件，则保留，然后再放入下一个变量，直到没有符合条件的变量纳入或者所有的变量都可纳入模型。
  * （2）后向选择backward：一开始将所有变量纳入模型，然后挨个移除不符合条件的变量，持续此过程，直到留下所有最优的变量为止。
  * （3）逐步选择stepwise：该算法是向前选择和向后选择的结合，逐步放入最优的变量、移除最差的变量。

#### 4.3.3、基于特征重要度的变量筛选
* 其原理主要是通过随机森林和GBDT等集成模型选取特征的重要度
* 随机森林计算特征重要度的步骤

  * <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/rf_feature.jpg" width="700" height="107" alt="error"/>

  * ***当改变样本在该特征的值，若袋外数据准确率大幅度下降，则该特征对于样本的预测结果 有很大影响，说明特征的重要度比较高。***

* GBDT计算特征重要度原理
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/gbdt_feature.jpg" width="691" height="270" alt="error"/>

#### 4.3.4、基于LASSO正则化的变量筛选
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/lasso_feature.jpg" width="700" height="100" alt="error"/>

### ***4.4、变量相关性分析***
* ***即使不进行线性相关性分析也不会影响模型的整体性能***
* 变量相关性的分析
  * 为了让模型更易于解释
  * 保证不同的分箱的得分正确

#### 4.4.1、变量两两相关性分析

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/Compatibility_analysis_1.jpg" width="700" height="320" alt="error"/>

* 代码
  * ```
    corr = data.corr()
    ```

* 两两相关性用heatmap表示
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/heatmap.png" alt="error"/>

#### 4.4.2、变量的多重共线性分析

<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/Compatibility_analysis_2.jpg" width="700" height="500" alt="error"/>

## 五、模型开发/模型评估
### ***5.1、WOE编码***
* 根据分箱时得到的WOE值，将数据转换为WOE值
* WOE转换可以将Logistic回归模型转变为标准评分卡格式。
* 引入WOE转换的目的并不是为了提高模型质量，只是一些变量不应该被纳入模型，这或者是因为它们不能增加模型值，或者是因为与其模型相关系数有关的误差较大
* ***建立标准信用评分卡也可以不采用WOE转换***
  * Logistic回归模型需要处理更大数量的自变量
  * 尽管这样会增加建模程序的复杂性，但最终得到的评分卡都是一样的

### ***5.2、逻辑回归建立***
#### 5.2.1、常用模型的对比
* 逻辑回归模型具有简单，稳定，可解释性强，技术成熟和易于检测和部署等优势，逻辑回归是评分卡模型最经常使用的算法。
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/model_list.jpg" alt="error"/>

* 根据系数符号进行筛选
  * 检查逻辑回归模型中各个变量的系数，如果所有***变量的系数均为正数***，模型有效。
  * ***假如有一些变量的系数出现了负数，说明有一些自变量的线性相关性较强，需要进一步进行变量筛选。***
  * 通常的做法
    * 综合考虑变量的IV值和业务的建议，按照变量的优先级进行降序排列
    * 选择优先级最高的4-5个基本变量
    * 按优先级从高到低逐渐添加变量，当新添加的变量之后，出现系数为负的情况，舍弃该变量
    * 直到添加最后一个变量

  * ***为什么回归模型中各个变量的系数均为正数？***
    * 对于分箱的WOE编码，分箱中坏客户占比越大，WOE值越大
    * 也就是说该分箱中客户为坏客户的概率就越大，对应的WOE值越大，即WOE与逻辑回归的预测结果 (坏客户的概率) 成正比。

  * ***为什么说假如有一些变量的系数出现了负数，说明有一些自变量的线性相关性较强？***
    * 正常情况下，WOE编码后的变量系数一定为正值。
    * 由上面为什么进行线性相关性分析的问题可知，由于一些自变量线性相关，导致系数权重会有无数种取法，使得可以为正数，也可以为负数。

* 根据p-value进行筛选
  * 模型假设某自变量与因变量线性无关，p-value可以理解为该假设成立的可能性 (便于理解，不太准确)。
  * 当p-value大于阈值时，表示假设显著，即自变量与因变量线性无关
  * 当p-value小于阈值时，表示假设不显著，即自变量与因变量线性相关
  * 阈值又称为显著性水平，通常取0.05
  * ***因此当某个字段的 p-value 大于0.05时，应该删除此变量。***

* ***先根据系数符号进行筛选，再进行p-value筛选？***
  * 先根据系数符号进行筛选，再进行p-value筛选。

#### 5.2.2、模型评价
* ***注意：以下各个指标的意义依赖于WOE的定义***

* TPR和FRP
  * TPR (或Recall) 为坏客户的查全率，表示被模型抓到的坏客户占总的坏客户的比例
  * FPR 为好客户误判率，表示好客户中倍模型误误判的比例
<div align=center>
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/TPR_FPR.jpg" width="400" height="356" alt="error"/>
</div>

  * 可以把TPR看做模型的收益，FPR看做模型付出的代价
  * TPR越大，表示模型能够抓到的坏客户比例越大，即收益越大
  * FPR越大，表示模型能够将好客户误抓的比例越大，即代价越大

* AUC（Area Under Curve）和ROC（receiver operating characteristic）
  * AUC 表示模型对任意坏客户的输出结果为大于模型对任意好客户的输出结果的概率
  * AUC的取值范围在0.5和1之间
  * ***AUC 越大，表示模型预测性能越好***
<div align=center>
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/AUC.png" width="700" height="568" alt="error"/>
</div>

* KS值
  * KS值表示了模型区分好坏客户的能力。
  * 其实质是TPR-FPR随好坏客户阈值变化的最大值。
  * KS的取值范围在0.5和1之间，值越大，模型的预测准确性越好。
  * 一般，KS > 0.4 即认为模型有比较好的预测性能。

* Kaggle数据集的实验结果
  * train_data
  <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/train_perf.png" alt="error"/>

  * test_data
  <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/test_perf.png" alt="error"/>

## 六、信用评分及评分系统
#### ***6.1、相关概念***
* 将客户违约的概率表示为p，则正常的概率为1-p。
  * 这两个事件相互排斥并互为补集，即其概率之和等于1。
  * 因此可以得到Odds（客户违约的相对概率）
    * <img src="http://latex.codecogs.com/gif.latex?Odds = \frac{P}{1-P}" />
  * 或者可以用如下方法计算违约的概率P
    * <img src="http://latex.codecogs.com/gif.latex?P = \frac{Odds}{1+Odds}" />

* 评分卡设定的分值刻度可以通过将分值表示为比率对数的线性表达式来定义。
  * <img src="http://latex.codecogs.com/gif.latex?Socre = A - B\log(Odds)" />

  * 其中，A和B是常数。负号可以使得违约概率越低，得分越高。
  * 通常，这是分值的理想方向，高分值代表低分险

* A和B的值可以通过将两个***已知或假设的分值***代入公式计算得到
  * 两个假设
    * （1）在某个特定的比率 <img src="http://latex.codecogs.com/gif.latex?Odds = \theta_0"/>时，设定特定的预期分值<img src="http://latex.codecogs.com/gif.latex?P_0"/>
    * （2）指定违约概率翻倍的评分（PDO）

  * 计算方法
    * （1）设定比率为<img src="http://latex.codecogs.com/gif.latex?Odds = \theta_0"/>的特定点的分值为<img src="http://latex.codecogs.com/gif.latex?P_0"/>
    * （2）比率为<img src="http://latex.codecogs.com/gif.latex?Odds = 2\theta_0"/>的点的分值为<img src="http://latex.codecogs.com/gif.latex?P_0 + PDO"/>
    * 代入socre公式
      * <img src="http://latex.codecogs.com/gif.latex?P_0 = A - B\log(\theta_0)"/>
      * <img src="http://latex.codecogs.com/gif.latex?P_0 + PDO = A - B\log(2\theta_0)"/>
    * 解上述两个方程中的常数A和B，可以得到
      * <img src="http://latex.codecogs.com/gif.latex?B = \frac{PDO}{\log(2)}"/>
      * <img src="http://latex.codecogs.com/gif.latex?A = P_0 + B\log(\theta_0)"/>

  * 例如，假设想要设定评分卡刻度使得比率为{1:60}（违约比正常）时的分值为600分，PDO = 20（每低20分，坏客户比率翻倍）。然后，给定 B = 28.85, A = 481.86。则可以计算分值为：
    * <img src="http://latex.codecogs.com/gif.latex?Score = 481.86 + 28.85\log(Odds)"/>
    * 常数A通常被称为补偿，常数B被称为刻度。

    * 根据如上公式，可以得到以下结果
<div align=center>
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/score_card.jpg" width="700" height="363" alt="error"/>
</div>

* 逻辑回归模型计算比率公式
  * <img src="http://latex.codecogs.com/gif.latex?\log(Odds) = \beta_0 + \beta_1{x_1} + ... + \beta_p{x_p}"/>

* 将Score公式和逻辑回归模型的公式合并
  * <img src="http://latex.codecogs.com/gif.latex?Score = A - B{\beta_0 + \beta_1{x_1} + ... + \beta_p{x_p}}"/>
  * 其中，变量<img src="http://latex.codecogs.com/gif.latex?x_1, ... , x_p"/>是出现在最终模型中的自变量。
  * 由于所有变量都用WOE转换进行了转换，可以将这些变量中的每一个都写成如下的展开式。

<div align=center>
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/logistic_score_1.jpg" width="700" height="510" alt="error"/>
</div>

<div align=center>
<img src="https://github.com/Silver-L/score_card/blob/master/data/fig/logistic_score_2.jpg" width="700" height="579" alt="error"/>
</div>

* 从以上公式中，我们发现每个分箱的评分都可以表示为<img src="http://latex.codecogs.com/gif.latex?-B(\beta_i{\omega_ij}"/>，也就是说影响每个分箱的因素包括三部分
  * 参数<img src="http://latex.codecogs.com/gif.latex?B"/>
  * 变量系数<img src="http://latex.codecogs.com/gif.latex?\theta_i"/>
  * 对应分箱的WOE编码<img src="http://latex.codecogs.com/gif.latex?\omega_ij"/>

#### ***6.2、Kaggle数据集的实验结果***
* 变量对应的评分，例：变量age
<div align=center>
    <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/age_score.png" alt="error"/>
</div>

* 一部分测试数据的总评分（basepoint=588, odds0=1 / 20, pdo=20）
  * <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/score.png" alt="error"/>

* 数据集整体的评分分布
  * <img src="https://github.com/Silver-L/score_card/blob/master/data/fig/psi.png" alt="error"/>

## 7、小结
简单学习了评分卡的创建流程。分箱等特征工程的相关知识还需多加学习。

## Reference
#### 1、评分卡的总结性文章（干货较多）
```
https://www.jianshu.com/p/f931a4df202c
https://www.jianshu.com/p/2759e090bd53?t=123
https://zhuanlan.zhihu.com/p/36539125 (分箱等等）
https://zhuanlan.zhihu.com/p/90251922
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
https://blog.csdn.net/App_12062011/article/details/103120776
https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm (Monotonic-WOE-Binning)
```
#### 6、开源项目
```
https://github.com/ShichenXie/scorecardpy (scorecardpy)
```
#### 7、书籍
```
《信用风险评分卡研究》Mamdouh Refaat著
```