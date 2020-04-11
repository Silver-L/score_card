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

```
https://www.jianshu.com/p/f931a4df202c
```
## 1、数据获取
### Give Me Some Credit (Kaggle)
```
https://www.kaggle.com/c/GiveMeSomeCredit/data
```

### 变量说明
* 基本属性：包括了借款人当时的年龄。
* 偿债能力：包括了借款人的月收入、负债比率。
* 信用往来：两年内35-59天逾期次数、两年内60-89天逾期次数、两年内90天或高于90天逾期的次数。
* 财产状况：包括了开放式信贷和贷款数量、不动产贷款或额度数量。
* 贷款属性：暂无。
* 其他因素：包括了借款人的家属数量（不包括本人在内）。
* 时间窗口：自变量的观察窗口为过去两年，因变量表现窗口为未来两年。

## 2、数据预处理
### 数据集整体情况
用describe函数获取数据集情况

```
data = pd.read_csv('')
data.describe()
```

### 缺失值处理
```
https://blog.csdn.net/xiedelong/article/details/81607598
https://blog.csdn.net/w352986331qq/article/details/78639233
```
#### 1、直接删除元祖

#### 2、数据补齐
* 人工填写（Filling Manually）
  * 这个方法产生数据偏离最小，是填充效果最好的一种。当数据规模很大、空值很多的时候，该方法是不可行的。
  * ***不适合大数据***

* 特殊值填充（Treating Missing Attribute values as Special values）
  * 将空值作为一种特殊的属性值来处理，它不同于其他的任何属性值。如所有的空值都用“unknown”填充。
  * ***可能导致数据偏离，一般不使用***

* 平均值填充 (Mean/Mode Completer)
  * 如果空值是数值属性，就使用该属性在其他所有对象的取值的平均值来填充该缺失的属性值.
如果空值是非数值属性，就根据统计学中的众数原理，用该属性在其他所有对象出现频率最高的值来补齐该缺失的属性值。

* 热卡填充（Hot deck imputation，或就近补齐）
  * 对于一个包含空值的对象，热卡填充法在完整数据中找到一个与它最相似的对象，然后用这个相似对象的值来进行填充。不同的问题选用不同的标准来对相似进行判定。

* 聚类填充(clustering imputation)
  * 常用方法： K-NN
      * 先根据欧式距离或相关分析来确定距离具有缺失数据样本最近的K个样本，将这K个值加权平均来估计该样本的缺失数据。
      * ```
        fancyimpute 安装（其中包含TensorFlow）
        1、conda install ecos
        2、conda install CVXcanon
        3、pip install fancyimpute
        
        from fancyimpute import KNN
        dataframe = KNN(k=3).fit_transform(dataframe)
    
        ```

* 使用所有可能的值填充（Assigning All Possible values of the Attribute）
  * 这种方法是用空缺属性值的所有可能的属性取值来填充，能够得到较好的补齐效果。但是当数据量很大或者遗漏的属性值较多时，其计算的代价很大，可能的测试方案很多。

* 组合完整化方法（Combinatorial Completer）

* 回归（Regression）
  * 基于完整的数据集，建立回归方程（模型）。对于包含空值的对象，将已知属性值代入方程来估计未知属性值，以此估计值来进行填充。当变量不是线性相关或预测变量高度相关时会导致有偏差的估计。

* 极大似然估计（Max Likelihood ，ML）
  * 常用方法： EM
  * 在缺失类型为随机缺失的条件下，假设模型对于完整的样本是正确的，通过观测数据的边际分布可以对未知参数进行极大似然估计。它一个重要前提：适用于大样本。有效样本的数量足够以保证ML估计值是渐近无偏的并服从正态分布。但是这种方法可能会陷入局部极值，收敛速度也不是很快，并且计算很复杂。

* 多重插补（Multiple Imputation，MI）
