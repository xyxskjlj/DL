#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#不做数量显示限制
# pd.options.display.max_rows=20#Notebook 的一个cell的显示行数
# pd.options.display.max_columns=20#Notebook 的一个cell的显示列数

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,cross_val_score

# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF

# from sklearn.svm import SVC
from tqdm import tqdm 
from sklearn import metrics
import csv #调用数据保存文件
import os
#聚类
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fcmeans import FCM
from collections import Counter
#过采样
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE as BSMOTE,ADASYN,SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import  DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from fancyimpute import KNN, IterativeSVD, MatrixFactorization,SimpleFill


# In[2]:


import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

ori_data = pd.read_csv('COM ADNI DATA.csv', header=None)
data_meta = ori_data.copy()

data_meta = data_meta[1:][:]
data_meta_nor = preprocessing.scale(data_meta.iloc[:, 2:10])
print(data_meta_nor)

data = data_meta_nor.copy()
data = pd.DataFrame(data_meta_nor)
data[8] = data_meta.iloc[:, 1].reset_index(drop=True).astype('int')

data = pd.DataFrame(data)
data.corr()
plt.figure(figsize=(25, 13))
sns.heatmap(data.corr(), annot=True)
data = pd.DataFrame(data)
data


# In[3]:


ori_data


# In[4]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE,ADASYN,SMOTE

X= data.iloc[:,0:8]  #选择了前8列作为特征集
y=data.iloc[:,8]#选择了第9列（因为索引从0开始）作为标签集
print('不经过任何采样处理的原始 y_train 中的分类情况：{}'.format(Counter(y)))#打印了原始标签y中每个类别的计数，以显示数据的平衡情况

# 综合采样（先过采样再欠采样）
kos = SMOTETomek(random_state=0)  # 综合采样,设置一个固定的随机种子使得每次生成的合成样本都是相同的。
X_kos, y_kos = kos.fit_resample(X,y)#对特征集X和标签集y进行采样,它接受特征集X和标签集y作为输入，并返回新的采样特征集X_kos和新的采样标签集y_kos。
print('综合采样后，训练集 y_kos 中的分类情况：{}'.format(Counter(y_kos)))

nos = SMOTEENN(random_state=1)  # 综合采样
X_nos, y_nos = nos.fit_resample(X, y)
print('综合采样后，训练集 y_nos 中的分类情况：{}'.format(Counter(y_nos)))

Bos = BorderlineSMOTE(random_state=2)  # 综合采样
X_Bos, y_Bos = Bos.fit_resample(X, y)
print('综合采样后，训练集 y_Bos 中的分类情况：{}'.format(Counter(y_Bos)))

Aos = ADASYN(random_state=3)  # 综合采样
X_Aos, y_Aos = Aos.fit_resample(X, y)
print('综合采样后，训练集 y_Aos 中的分类情况：{}'.format(Counter(y_Aos)))


# In[5]:


cv_outer = StratifiedKFold(n_splits=5, random_state=7,shuffle=True)
i=1

if not os.path.exists('train data'):
    os.mkdir('train data')
if not os.path.exists('test data'):
    os.mkdir('test data')#这两行代码检查是否存在名为train data和test data的目录。如果不存在，则创建它们。

for train_idx, test_idx in tqdm(cv_outer.split(X, y)):#对特征集X和标签集y进行分层交叉验证分割。
    train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]#选择特征集X中索引为train_idx的行，这些行是当前训练集的特征。选择特征集X中索引为test_idx的行，这些行是当前测试集的特征。
    train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]
    train = np.column_stack((train_data, train_target))#将训练数据和对应的标签组合成一个二维数组。
    test = np.column_stack((test_data, test_target ))

    train_data = 'train data/train_set{}.csv'.format(i)#这两行代码创建训练集和测试集的文件路径。i是当前迭代的索引，用于区分不同的文件。
    test_data = 'test data/test_set{}.csv'.format(i)
    if os.path.exists(train_data):#这行代码检查train_data指向的文件是否存在。
        os.remove(train_data)
        pd.DataFrame(train).to_csv(train_data, index=False, encoding='utf-8')#创建一个新的DataFrame，其中包含之前合并的训练数据（train变量），并将它保存为CSV文件
    else:
        pd.DataFrame(train).to_csv(train_data, index=False, encoding='utf-8')
        
    if os.path.exists(test_data):
        os.remove(test_data)
        pd.DataFrame(test).to_csv(test_data, index=False, encoding='utf-8')
    else:
        pd.DataFrame(test).to_csv(test_data, index=False, encoding='utf-8')
    i=i+1


# # 模拟缺失值模式

# # 聚类

# In[6]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

train_data = pd.read_csv('train data/train_set1.csv', header=None)
train_data = train_data[1:][:]
train_data_Eclass = train_data.iloc[:, 0:8]
train_data_class = train_data.iloc[:, 8]

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(train_data_Eclass)

score = silhouette_score(train_data_Eclass, kmeans.labels_)
print(score)
print("meta data:\n", pd.value_counts(train_data_class)), pd.value_counts(kmeans.labels_), kmeans.labels_


# In[7]:


from sklearn.cluster import KMeans

X_full_clean = train_data_Eclass.values   # (n, 8)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X_full_clean)

# 以后就用 kmeans 作为 clusterer


# In[8]:


import numpy as np

b = list(range(0, 8))  # 0~7 列允许缺失（最后一列是标签）
missing_rates = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

baseline_mis_datasets = []

for i, rate in enumerate(missing_rates):
    df_new = train_data_Eclass.copy()
    rng = np.random.default_rng(709 + i)

    n = len(df_new)
    row_positions = np.arange(n)  # 位置下标：0 ~ n-1

    for col in b:
        n_miss = round(rate * n)
        miss_pos = rng.choice(row_positions, size=n_miss, replace=False)
        df_new.iloc[miss_pos, col] = np.nan

    baseline_mis_datasets.append(df_new)


# In[9]:


col_std = train_data_Eclass.std(axis=0)
print(col_std)


# In[10]:


# 模拟 --MCAR--缺失模式
def create_mcar_single(df, missing_column, p_missing, random_state=709):
    np.random.seed(random_state)#这行代码设置了NumPy的随机数生成器的种子。使用固定的种子可以确保每次运行代码时生成的随机数序列相同，这对于需要可重复结果的研究或实验是非常重要的。
    indices = [df.sample(n = 1).index[0] for i in range(round(p_missing * df.shape[0]))]#这段代码确保indices列表中的索引是唯一的。set(indices)创建了一个包含所有唯一索引的集合，round(p_missing*df.shape[0])计算MCAR缺失值应占的比例。如果集合中的索引数小于所需的缺失值比例，则循环继续，直到添加足够的唯一索引。

    while len(set(indices)) < round(p_missing * df.shape[0]):#确保随机选择的索引数量至少与缺失数据的比例相符。
        indices.append(df.sample(n = 1).index[0])
    mcar_column = [1 if i in indices else 0 for i in range(df.shape[0])]#这行代码创建了一个名为mcar_column的新列表，其中包含0和1的值。如果DataFrame中的行索引（从0到df.shape[0] - 1）在indices列表中，则对应的值为1（表示缺失），否则为0。
    
    df_new = df.copy()
    for i in range(len(mcar_column)):
        if mcar_column[i] == 1:#这两个循环遍历mcar_column列表，对于每个被标记为缺失的行（即mcar_column[i]为1的情况），将df_new中对应行的missing_column列的值设置为'?'字符。
            df_new[missing_column][i] = '?'      
    df_new = df_new.replace('?', np.nan)#这行代码将所有'?'字符替换为np.nan，这是Pandas中用于表示缺失值的专用字符串。
    return df_new

def create_mcar_mult(df, mising_column, p_missing, random_state):
    df_new = df.copy()
    for i in range(len(mising_column)):#开始一个循环，遍历列表 mising_column，这个列表包含了所有需要创建MCAR缺失数据的列的名称
        tmp = create_mcar_single(df, mising_column[i], p_missing, random_state=random_state+i)
        df_new[mising_column[i]] = tmp[mising_column[i]] #将create_mcar_single 函数返回的DataFrame tmp 中对应列的数据赋值给 df_new 中的同一列。这样做可以保证 df_new 包含了所有列的MCAR缺失数据。
    return df_new

def create_mcar(df, missing_column, p_missing, random_state=709):
    if (type(missing_column) == str):
        df_new = create_mcar_single(df, missing_column, p_missing, random_state=709)
    elif (type(missing_column) == list):
        df_new = create_mcar_mult(df, missing_column, p_missing, random_state=709)
    else:
        raise Exception('Name of the columns should be given as either str or list. Given format was {}'.format(
            type(missing_column)))#如果missing_column既不是字符串也不是列表，代码将抛出一个异常。异常信息指出应将列名提供为字符串或列表，并指出了给定的数据类型。
    return df_new

def test_mcar_single(df, missing_column, p_missing):
    df_new = create_mcar_single(df, missing_column, p_missing, random_state=709)
    if (df_new[missing_column].isna().sum() == round(p_missing * df.shape[0])):#用.isna()方法检查df_new中missing_column列的所有NaN（即缺失）,.sum()方法计算缺失值的总数,计算预期中缺失值的数量
        print('Missingness created for', missing_column ,'succesfully!')
    else:
        print('Something is wrong.')
        
def test_mcar_mult(df, missing_column, p_missing):
    for i in range(len(missing_column)):
        test_mcar_single(df, missing_column[i], p_missing)

def test_mcar(df, missing_column, p_missing):
    if (type(missing_column) == str):
        result = test_mcar_single(df, missing_column, p_missing)
    elif (type(missing_column) == list):
        result = test_mcar_mult(df, missing_column, p_missing)
    return(result)


# In[11]:


import numpy as np
import pandas as pd

def create_mnar(dataset_meta: pd.DataFrame,
                feature_indices: list,
                p_missing: float,
                skew: float = 5.0,
                random_state: int = 0):
    """
    纯净版 MNAR（并行多列版）：
    - 对 feature_indices 中的每一列，都构造 MNAR 缺失；
    - 每一列的缺失概率只依赖该列自己的取值（越极端越容易缺）；
    - 单列缺失率控制在 p_missing 左右；
    - 所有列的缺失掩码在原始完整数据上独立生成，再一次性写入 NaN。
    """
    # 统一成 list，防止传 int
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    df_full = dataset_meta.copy()
    n = len(df_full)
    cols = df_full.columns

    rng_master = np.random.default_rng(random_state)
    masks = {}

    for idx in feature_indices:
        tgt_col = cols[idx]
        x = df_full[tgt_col].to_numpy(dtype=float)

        # 标准化到以中位数为中心
        x_std = (x - np.median(x)) / (np.std(x) + 1e-8)

        # logits：越极端，|x_std| 越大，logits 绝对值越大
        logits = skew * x_std

        # sigmoid → 0~1 的“原始缺失概率”
        prob_raw = 1 / (1 + np.exp(-logits))

        # 调整到整体平均缺失率 ≈ p_missing
        prob = prob_raw * p_missing / (prob_raw.mean() + 1e-8)
        prob = np.clip(prob, 0.01, 0.99)

        # 为这一列生成独立随机数
        rng_col = np.random.default_rng(int(rng_master.integers(1e9)))
        miss_bool = rng_col.random(n) < prob

        # 精确控制：这一列缺失个数 = round(p_missing * n)
        n_need = int(round(p_missing * n))
        curr = miss_bool.sum()

        if curr < n_need:
            add = rng_col.choice(np.where(~miss_bool)[0], n_need - curr, replace=False)
            miss_bool[add] = True
        elif curr > n_need:
            rem = rng_col.choice(np.where(miss_bool)[0], curr - n_need, replace=False)
            miss_bool[rem] = False

        masks[tgt_col] = miss_bool
        print(f"MNAR done: target={tgt_col}, missing={miss_bool.sum()} / {n}")

    # 一次性写回所有列的 NaN
    df_mnar = df_full.copy()
    for col_name, miss_bool in masks.items():
        df_mnar.loc[miss_bool, col_name] = np.nan

    return df_mnar


# In[12]:


from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

def create_mar(dataset_meta: pd.DataFrame,
               feature_indices: list,
               p_missing: float,
               random_state: int = 0):
    """
    纯净版 MAR（并行列）：
    - 对 feature_indices 中的每一列，独立构造 MAR 缺失机制；
    - 每一列的缺失概率仅由“其它列（在原始完整数据中）”预测；
    - 所有列的缺失掩码先分别在完整数据上算好，再一次性写入 NaN。

    参数
    ----
    dataset_meta : 原始完整 DataFrame（不含模拟缺失）
    feature_indices : 要造 MAR 缺失的列索引列表，例如 [0,1,2,3,4,5,6,7]
    p_missing : 单列的目标缺失率（按行），如 0.3
    random_state : 总体随机种子
    """
    # 统一成 list，防止传 int
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    df_full = dataset_meta.copy()
    n = len(df_full)
    cols = df_full.columns

    rng_master = np.random.default_rng(random_state)

    # 保存每一列的缺失掩码
    masks = {}

    for idx in feature_indices:
        tgt_col = cols[idx]

        # === 1. 在“原始完整数据”上构造 X, y ===
        # 特征 X = 除目标列以外的所有列（这里用 df_full，保证其它列是完整的）
        X = df_full.drop(columns=[tgt_col])
        y = df_full[tgt_col]

        # 如果原始数据里已经有 NaN，这里只用完全观测的行来训练
        clean_mask = (~X.isna().any(axis=1)) & (~y.isna())
        X_clean, y_clean = X[clean_mask], y[clean_mask]

        if len(X_clean) == 0:
            raise ValueError(f"列 {tgt_col} 清洗后没有可用样本，无法训练 MAR 模型。")

        # === 2. 用其它列预测目标列（RandomForest 回归） ===
        sub_seed = int(rng_master.integers(1e9))
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=sub_seed,
            n_jobs=-1,            # 并行加速森林内部
        )
        model.fit(X_clean, y_clean)

        # === 3. 预测值 → 概率（在完整数据的“干净行”上） ===
        pred = model.predict(X_clean)  # 只对 clean 行有预测

        pred_min, pred_max = pred.min(), pred.max()
        prob_raw = (pred - pred_min) / (pred_max - pred_min + 1e-8)  # 归一到 [0,1]

        # 调整到整体平均缺失率 ≈ p_missing
        prob = prob_raw * p_missing / (prob_raw.mean() + 1e-8)
        prob = np.clip(prob, 0.01, 0.99)

        # 对齐到所有样本：非 clean 行（X 或 y 原本就缺）缺失概率设为 0
        prob_sr = pd.Series(prob, index=X_clean.index)
        prob_aligned = prob_sr.reindex(df_full.index, fill_value=0.0).to_numpy()

        # === 4. 按概率扔硬币，生成这一列的缺失掩码 ===
        rng_col = np.random.default_rng(int(rng_master.integers(1e9)))
        miss_bool = rng_col.random(n) < prob_aligned

        # 精确对齐：这一列的缺失行数 = round(p_missing * n)
        n_need = int(round(p_missing * n))
        curr = miss_bool.sum()

        if curr < n_need:
            add = rng_col.choice(np.where(~miss_bool)[0], n_need - curr, replace=False)
            miss_bool[add] = True
        elif curr > n_need:
            rem = rng_col.choice(np.where(miss_bool)[0], curr - n_need, replace=False)
            miss_bool[rem] = False

        masks[tgt_col] = miss_bool  # 保存这一列的 mask

        print(f"[MAR-parallel] target={tgt_col}, missing={miss_bool.sum()} / {n}")

    # === 5. 最后一步：把所有列的缺失一次性写入 ===
    df_mar = df_full.copy()
    for col_name, miss_bool in masks.items():
        df_mar.loc[miss_bool, col_name] = np.nan

    return df_mar


# In[13]:


b=list(range(0,8))#排除分类变量再去进行缺失值模拟,这行代码创建了一个包含数字0到7的列表b。列表中的数字代表了数据集中除了分类变量以外的其他变量索引。这里的0到7可能对应于一个8特征的DataFrame，其中第8个特征是分类变量，因此需要排除它。
mis_mcar_005=create_mcar(train_data_Eclass,b,0.05)#b是除了分类变量外的特征索引列表，0.05是缺失数据的比例。
mis_mcar_01=create_mcar(train_data_Eclass,b,0.1)#这行代码调用了一个名为create_mcar的函数来创建一个具有5%缺失数据的MCAR,这些行代码使用了相同的方式创建了具有不同缺失数据率（从10%到80%）的MCAR数据集
mis_mcar_02=create_mcar(train_data_Eclass,b,0.2)
mis_mcar_03=create_mcar(train_data_Eclass,b,0.3)
mis_mcar_04=create_mcar(train_data_Eclass,b,0.4)
mis_mcar_05=create_mcar(train_data_Eclass,b,0.5)
mis_mcar_06=create_mcar(train_data_Eclass,b,0.6)
mis_mcar_07=create_mcar(train_data_Eclass,b,0.7)
mis_mcar_08=create_mcar(train_data_Eclass,b,0.8)

# ===== MNAR：对 0~7 列都做 MNAR 缺失（温和版） =====
mis_mnar_005 = create_mnar(train_data_Eclass, b, 0.05, random_state=0)
mis_mnar_01  = create_mnar(train_data_Eclass, b, 0.10, random_state=1)
mis_mnar_02  = create_mnar(train_data_Eclass, b, 0.20, random_state=2)
mis_mnar_03  = create_mnar(train_data_Eclass, b, 0.30, random_state=3)
mis_mnar_04  = create_mnar(train_data_Eclass, b, 0.40, random_state=4)
mis_mnar_05  = create_mnar(train_data_Eclass, b, 0.50, random_state=5)
mis_mnar_06  = create_mnar(train_data_Eclass, b, 0.60, random_state=6)
mis_mnar_07  = create_mnar(train_data_Eclass, b, 0.70, random_state=7)
mis_mnar_08  = create_mnar(train_data_Eclass, b, 0.80, random_state=8)


# ===== MAR：对 0~7 列都做“温和版 MAR” 缺失 =====
mis_mar_005 = create_mar(train_data_Eclass, b, 0.05, random_state=0)
mis_mar_01  = create_mar(train_data_Eclass, b, 0.10, random_state=1)
mis_mar_02  = create_mar(train_data_Eclass, b, 0.20, random_state=2)
mis_mar_03  = create_mar(train_data_Eclass, b, 0.30, random_state=3)
mis_mar_04  = create_mar(train_data_Eclass, b, 0.40, random_state=4)
mis_mar_05  = create_mar(train_data_Eclass, b, 0.50, random_state=5)
mis_mar_06  = create_mar(train_data_Eclass, b, 0.60, random_state=6)
mis_mar_07  = create_mar(train_data_Eclass, b, 0.70, random_state=7)
mis_mar_08  = create_mar(train_data_Eclass, b, 0.80, random_state=8)


# In[14]:


import numpy as np
import pandas as pd

# ===== 缺失率列表（你之前就有）=====
missing_rates = [0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# ===== 0~7 列都是特征，train_data_Eclass 为完整真值 =====
X_full_df = train_data_Eclass          # DataFrame
X_full = X_full_df.values              # 真值矩阵

# ===== 三种机制：每种机制一个 9 元列表（与 missing_rates 顺序一一对应）=====
baseline_mis_datasets_by_mech = {
    "mcar": [
        mis_mcar_005,
        mis_mcar_01,
        mis_mcar_02,
        mis_mcar_03,
        mis_mcar_04,
        mis_mcar_05,
        mis_mcar_06,
        mis_mcar_07,
        mis_mcar_08,
    ],
    "mnar": [
        mis_mnar_005,
        mis_mnar_01,
        mis_mnar_02,
        mis_mnar_03,
        mis_mnar_04,
        mis_mnar_05,
        mis_mnar_06,
        mis_mnar_07,
        mis_mnar_08,
    ],
    "mar": [
        mis_mar_005,
        mis_mar_01,
        mis_mar_02,
        mis_mar_03,
        mis_mar_04,
        mis_mar_05,
        mis_mar_06,
        mis_mar_07,
        mis_mar_08,
    ],
}
X_full_df = train_data_Eclass              # (n, 8)

missing_rates = [0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# In[ ]:





# # 插补

# In[15]:


# =========================
# 【你需要新增】基回归器 & 自适应 max_iter
# =========================
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, clone

def adaptive_max_iter(rate: float) -> int:
    """根据缺失率自动选择迭代次数（越缺越多，迭代越多）。"""
    if rate <= 0.10:
        return 5
    elif rate <= 0.30:
        return 10
    elif rate <= 0.50:
        return 15
    else:
        return 20

def get_base_regressors_for_rate(rate: float, random_state: int = 0):
    """
    按缺失率选基回归器家族：
    - 小缺失率：只用轻量 LR / BR
    - rate >= 0.2 时再加 RF / ET / GB
    """
    base_regs = {
        "LR": LinearRegression(),
        "BR": BayesianRidge(),
    }

    if rate >= 0.20:
        base_regs.update({
            "RF": RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            ),
            "ET": ExtraTreesRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            ),
            "GB": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            ),
        })

    return base_regs

def build_iterative_imputers_for_rate(rate: float,
                                      random_state: int = 0,
                                      tol: float = 1e-3):
    """
    针对一个缺失率 rate，构造多种 IterativeImputer：
    返回 {name: IterativeImputer}
    """
    base_regs = get_base_regressors_for_rate(rate, random_state=random_state)
    max_iter = adaptive_max_iter(rate)

    imputers = {}
    for name, reg in base_regs.items():
        imputers[name] = IterativeImputer(
            estimator=reg,
            max_iter=max_iter,
            tol=tol,
            sample_posterior=False,
            random_state=random_state
        )
    return imputers


# In[16]:


from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

def eval_one_imputer(rate, df_mis, imp_name, imp, X_full):
    X_mis = df_mis.values
    mask = np.isnan(X_mis)

    X_imp = imp.fit_transform(X_mis)

    recs = []
    n_features = X_mis.shape[1]

    for j in range(n_features):
        miss_idx = np.where(mask[:, j])[0]
        if miss_idx.size == 0:
            continue

        rmse = np.sqrt(np.mean((X_full[miss_idx, j] - X_imp[miss_idx, j]) ** 2))

        recs.append({
            "rate": rate,
            "imp": imp_name,
            "col": j,
            "rmse": rmse
        })

    return recs


def compute_rmse_df_for_baseline(train_data_Eclass, baseline_mis_datasets,
                                 missing_rates, n_jobs=-1):
    """
    对一组“同一个机制”的数据列表 baseline_mis_datasets 计算 rmse_df。
    """
    X_full = train_data_Eclass.values
    records = []

    for rate, df_mis in zip(missing_rates, baseline_mis_datasets):
        print(f"=== 评估缺失率 {rate:.2f} ===")

        imputers = build_iterative_imputers_for_rate(rate, random_state=0)

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_one_imputer)(rate, df_mis, imp_name, imp, X_full)
            for imp_name, imp in imputers.items()
        )

        for rec_list in results:
            records.extend(rec_list)

    rmse_df = pd.DataFrame(records)
    return rmse_df


# In[ ]:


# MCAR
rmse_df_mcar = compute_rmse_df_for_baseline(
    train_data_Eclass,
    baseline_mis_datasets_by_mech["mcar"],
    missing_rates,
    n_jobs=-1
)
rmse_df_mcar["mech"] = "mcar"

# MAR
rmse_df_mar = compute_rmse_df_for_baseline(
    train_data_Eclass,
    baseline_mis_datasets_by_mech["mar"],
    missing_rates,
    n_jobs=-1
)
rmse_df_mar["mech"] = "mar"

# MNAR
rmse_df_mnar = compute_rmse_df_for_baseline(
    train_data_Eclass,
    baseline_mis_datasets_by_mech["mnar"],
    missing_rates,
    n_jobs=-1
)
rmse_df_mnar["mech"] = "mnar"
rmse_df_all = pd.concat([rmse_df_mcar, rmse_df_mar, rmse_df_mnar], ignore_index=True)


# In[ ]:


import pandas as pd

def build_best_imputer_by_rate(rmse_df: pd.DataFrame):
    """
    每个缺失率下，整体平均 RMSE 最小的插补器。
    输入 rmse_df 至少要有列: ['rate', 'imp', 'rmse']
    """
    tmp = (
        rmse_df
        .groupby(["rate", "imp"])["rmse"]
        .mean()
        .reset_index()
    )
    best_imp_by_rate = {}
    for rate, sub in tmp.groupby("rate"):
        best_row = sub.loc[sub["rmse"].idxmin()]
        best_imp_by_rate[float(rate)] = best_row["imp"]
    return best_imp_by_rate


def build_best_imputer_by_rate_and_col(rmse_df: pd.DataFrame):
    """
    每个 (缺失率 × 特征列) 下，RMSE 最小的插补器。
    输入 rmse_df 至少要有列: ['rate', 'col', 'imp', 'rmse']
    """
    tmp = (
        rmse_df
        .groupby(["rate", "col", "imp"])["rmse"]
        .mean()
        .reset_index()
    )
    best_imp_by_rate_col = {}
    for (rate, col), sub in tmp.groupby(["rate", "col"]):
        best_row = sub.loc[sub["rmse"].idxmin()]
        best_imp_by_rate_col[(float(rate), int(col))] = best_row["imp"]
    return best_imp_by_rate_col


# In[ ]:


best_imp_by_rate_mcar = build_best_imputer_by_rate(rmse_df_mcar)
best_imp_by_rate_col_mcar = build_best_imputer_by_rate_and_col(rmse_df_mcar)
best_imp_by_rate_mar = build_best_imputer_by_rate(rmse_df_mar)
best_imp_by_rate_col_mar = build_best_imputer_by_rate_and_col(rmse_df_mar)
best_imp_by_rate_mnar = build_best_imputer_by_rate(rmse_df_mnar)
best_imp_by_rate_col_mnar = build_best_imputer_by_rate_and_col(rmse_df_mnar)


# In[ ]:


# =========================
# 【你需要新增】计算所有基插补器的 RMSE
# =========================
feature_cols = list(range(0, 8))  # 8 个特征
X_full = train_data_Eclass.values  # 真值

records = []

for rate, df_mis in zip(missing_rates, baseline_mis_datasets):
    print(f"=== 评估缺失率 {rate:.2f} ===")

    # 针对这个缺失率，构造一批插补器（链式方程一样）
    imputers = build_iterative_imputers_for_rate(rate, random_state=0)

    # 并行评估所有插补器
    # n_jobs=-1 表示“尽可能占满所有 CPU 核”
    results = Parallel(n_jobs=-1)(
        delayed(eval_one_imputer)(rate, df_mis, imp_name, imp, X_full)
        for imp_name, imp in imputers.items()
    )

    # 把每个插补器返回的 list[dict] 展开
    for rec_list in results:
        records.extend(rec_list)

rmse_df = pd.DataFrame(records)
rmse_df = pd.DataFrame(records)

# ==================【新增】对 RMSE 按列标准化 ==================
# 每一列在完整数据里的标准差（8 个特征）
col_std = X_full.std(axis=0)  # 形状 (8,)

# 把每个 (rate, imp, col) 的 rmse 除以该列自己的 std
rmse_df = rmse_df.copy()
rmse_df["rmse_std"] = rmse_df.apply(
    lambda r: r["rmse"] / col_std[int(r["col"])],
    axis=1
)

# 按列汇总“标准化 RMSE”，看归一化之后哪些列仍然最难补
col_difficulty_std = (
    rmse_df
    .groupby("col")["rmse_std"]
    .agg(["mean", "std", "max"])
    .sort_values("mean", ascending=False)
)

print("\n=== 按列汇总的【标准化 RMSE】（mean 越大越难） ===")
print(col_difficulty_std.round(4))


# ==================【新增】按列整理 RMSE 表，看看哪一列难补 ==================

# 1）做一个大表：索引是列号 col，列是 (缺失率, 插补器)，值是该组合下的 RMSE
rmse_by_col_table = (
    rmse_df
    .groupby(["col", "rate", "imp"])["rmse"]
    .mean()
    .reset_index()
    .pivot_table(
        index="col",              # 行：特征列索引 0~7
        columns=["rate", "imp"],  # 列：多级索引 (缺失率, 插补器名)
        values="rmse"
    )
)

print("\n=== 每个特征列在不同缺失率 / 插补器下的 RMSE 一览 ===")
print(rmse_by_col_table)

# 2）再做一个简单汇总：按列聚合，看“整体最难补的是哪几列”
col_difficulty = (
    rmse_df
    .groupby("col")["rmse"]
    .agg(["mean", "std", "max"])
    .sort_values("mean", ascending=False)   # 平均 RMSE 越大越靠前，越难插补
)

print("\n=== 按列汇总的 RMSE 概览（平均从大到小，越大越难） ===")
print(col_difficulty)


print("\n=== 每个缺失率 × 插补器 的平均 RMSE（对 8 列特征取平均） ===")
print(
    rmse_df
    .groupby(["rate", "imp"])["rmse"]
    .mean()
    .unstack("imp")
    .sort_index()
)
print("=== rate=0.05 各列 RMSE ===")
print(
    rmse_df[rmse_df["rate"] == 0.05]
    .groupby(["imp", "col"])["rmse"]
    .mean()
    .unstack("col")
)
rmse_by_col_table.to_csv("rmse_by_col_table1.csv")
col_difficulty.to_csv("rmse_col_difficulty1.csv")


# In[ ]:


# =========================
# 【你需要新增】学习 best_imp_by_rate & best_imp_by_rate_col
# =========================
def build_best_imputer_by_rate(rmse_df: pd.DataFrame):
    """每个缺失率下，整体平均 RMSE 最小的插补器。"""
    tmp = (
        rmse_df
        .groupby(["rate", "imp"])["rmse"]
        .mean()
        .reset_index()
    )
    best_imp_by_rate = {}
    for rate, sub in tmp.groupby("rate"):
        best_row = sub.loc[sub["rmse"].idxmin()]
        best_imp_by_rate[float(rate)] = best_row["imp"]
    return best_imp_by_rate

def build_best_imputer_by_rate_and_col(rmse_df: pd.DataFrame):
    """
    每个 (缺失率 × 特征列) 下，用【标准化 RMSE】最小的插补器。
    """
    tmp = (
        rmse_df
        .groupby(["rate", "col", "imp"])["rmse_std"]   # 用 rmse_std
        .mean()
        .reset_index()
    )
    best_imp_by_rate_col = {}
    for (rate, col), sub in tmp.groupby(["rate", "col"]):
        best_row = sub.loc[sub["rmse_std"].idxmin()]
        best_imp_by_rate_col[(float(rate), int(col))] = best_row["imp"]
    return best_imp_by_rate_col

best_imp_by_rate_mcar     = build_best_imputer_by_rate(rmse_df_mcar)
best_imp_by_rate_col_mcar = build_best_imputer_by_rate_and_col(rmse_df_mcar)

best_imp_by_rate_mar      = build_best_imputer_by_rate(rmse_df_mar)
best_imp_by_rate_col_mar  = build_best_imputer_by_rate_and_col(rmse_df_mar)

best_imp_by_rate_mnar     = build_best_imputer_by_rate(rmse_df_mnar)
best_imp_by_rate_col_mnar = build_best_imputer_by_rate_and_col(rmse_df_mnar)

best_imp_by_rate_dict = {
    "mcar": best_imp_by_rate_mcar,
    "mar":  best_imp_by_rate_mar,
    "mnar": best_imp_by_rate_mnar,
}
best_imp_by_rate_col_dict = {
    "mcar": best_imp_by_rate_col_mcar,
    "mar":  best_imp_by_rate_col_mar,
    "mnar": best_imp_by_rate_col_mnar,
}


print("\n=== 每个缺失率下的“最优插补器” ===")
print(best_imp_by_rate)


# In[ ]:


# =========================
# 【你需要新增】RateAdaptiveImputer（按缺失率选基器）
# =========================
class RateAdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 best_imp_by_rate: dict,
                 missing_rates: list,
                 random_state: int = 0,
                 tol: float = 1e-3):
        self.best_imp_by_rate = best_imp_by_rate
        self.missing_rates = np.array(missing_rates, dtype=float)
        self.random_state = random_state
        self.tol = tol

    def _closest_rate(self, rate: float) -> float:
        idx = np.argmin(np.abs(self.missing_rates - rate))
        return float(self.missing_rates[idx])

    def fit(self, X, y=None, rate=None):
        X = np.asarray(X, dtype=float)

        if rate is None:
            miss_rate_est = np.isnan(X).mean()
        else:
            miss_rate_est = float(rate)

        chosen_rate = self._closest_rate(miss_rate_est)
        imp_name = self.best_imp_by_rate[chosen_rate]

        self.chosen_rate_ = chosen_rate
        self.chosen_imp_name_ = imp_name

        base_regs = get_base_regressors_for_rate(
            chosen_rate, random_state=self.random_state
        )
        reg = base_regs[imp_name]
        max_iter = adaptive_max_iter(chosen_rate)

        self.imputer_ = IterativeImputer(
            estimator=reg,
            max_iter=max_iter,
            tol=self.tol,
            sample_posterior=False,
            random_state=self.random_state
        )
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self.imputer_.transform(X)

    def fit_transform(self, X, y=None, rate=None):
        self.fit(X, y=y, rate=rate)
        return self.transform(X)


# In[ ]:


# =========================
# 【你需要新增】ColumnAdaptiveImputer（特征级自适应）
# =========================
class ColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 best_imp_by_rate_col: dict,
                 missing_rates: list,
                 random_state: int = 0):
        self.best_imp_by_rate_col = best_imp_by_rate_col
        self.missing_rates = np.array(missing_rates, dtype=float)
        self.random_state = random_state

    def _closest_rate(self, rate: float) -> float:
        idx = np.argmin(np.abs(self.missing_rates - rate))
        return float(self.missing_rates[idx])

    def fit(self, X, y=None, rate=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

                # 全局缺失率（作为兜底，可以不用）
        if rate is None:
            self.global_miss_rate_ = float(np.isnan(X).mean())
        else:
            self.global_miss_rate_ = float(rate)

        # 记录每一列最终用到的“就近缺失率”，方便调试
        self.col_chosen_rate_ = {}


        # 先均值粗插补
        self.simple_imp_ = SimpleImputer(strategy="mean")
        X_simple = self.simple_imp_.fit_transform(X)

        self.col_models_ = {}

        for j in range(n_features):
            mask_j = np.isnan(X[:, j])
            if mask_j.sum() == 0:
                continue
            if (~mask_j).sum() < 5:
                continue

            # ★ 每一列单独估计缺失率
            miss_rate_j = mask_j.mean()              # 这一列自己的缺失率
            chosen_rate_j = self._closest_rate(miss_rate_j)
            self.col_chosen_rate_[j] = chosen_rate_j

            key = (chosen_rate_j, j)
            imp_name = self.best_imp_by_rate_col.get(key, "LR")

            base_regs = get_base_regressors_for_rate(
                chosen_rate_j, random_state=self.random_state
            )
            if imp_name not in base_regs:
                imp_name = "LR"
            reg = clone(base_regs[imp_name])

            X_other = np.delete(X_simple, j, axis=1)
            X_other_train = X_other[~mask_j]
            y_train = X[~mask_j, j]

            reg.fit(X_other_train, y_train)
            self.col_models_[j] = reg

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        X_imp = self.simple_imp_.transform(X)

        for j, reg in self.col_models_.items():
            mask_j = np.isnan(X[:, j])
            if mask_j.sum() == 0:
                continue
            X_other = np.delete(X_imp, j, axis=1)
            X_other_missing = X_other[mask_j]
            y_pred = reg.predict(X_other_missing)
            X_imp[mask_j, j] = y_pred

        return X_imp

    def fit_transform(self, X, y=None, rate=None):
        self.fit(X, y=y, rate=rate)
        return self.transform(X)


# In[ ]:


import numpy as np
import pandas as pd

# ===== 1. 基于完整数据，计算每一列的统计特征 =====
X_full = train_data_Eclass.values   # (n_samples, 8)
n_samples, n_features = X_full.shape

df_full = pd.DataFrame(X_full, columns=[f"col_{j}" for j in range(n_features)])

col_mean  = df_full.mean()
col_std   = df_full.std()
col_min   = df_full.min()
col_max   = df_full.max()
col_range = col_max - col_min

# 相关系数矩阵（绝对值）
corr_mat = df_full.corr().abs().values  # (8, 8)

max_corr = {}
mean_corr = {}
for j in range(n_features):
    others = np.delete(corr_mat[j, :], j)
    max_corr[j]  = others.max()
    mean_corr[j] = others.mean()

# ===== 2. 把 (rate, col) → best_imp_by_rate_col 变成一张“元学习训练集” =====
meta_rows = []

for (rate, col), imp_name in best_imp_by_rate_col.items():
    j = int(col)
    r = float(rate)

    meta_rows.append({
        "rate": r,
        "col_idx": j,
        "col_mean":  float(col_mean[j]),
        "col_std":   float(col_std[j]),
        "col_range": float(col_range[j]),
        "max_corr":  float(max_corr[j]),
        "mean_corr": float(mean_corr[j]),
        "best_imp":  imp_name,   # 标签：最优基器类型（BR/LR/RF/ET/GB）
    })

meta_df = pd.DataFrame(meta_rows)

print("meta_df 预览：")
print(meta_df.head())
print("\n标签分布：")
print(meta_df["best_imp"].value_counts())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# ===== 1. 准备 X / y =====
X_meta = meta_df.drop(columns=["best_imp"])
y_meta = meta_df["best_imp"].copy()

imp_label_encoder = LabelEncoder()
y_meta_enc = imp_label_encoder.fit_transform(y_meta)

# ===== 2. 训练一个简单的随机森林分类器 =====
rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=0
)

# 可以简单看看交叉验证的准确率（可选）
scores = cross_val_score(rf_selector, X_meta, y_meta_enc, cv=3)
print("Meta-selector CV accuracy:", scores.mean().round(3), "+/-", scores.std().round(3))

# 在全数据上拟合，后面直接用
rf_selector.fit(X_meta, y_meta_enc)

# ===== 3. 封装成一个小工具类 =====
class MetaImputerSelector:
    """
    元学习插补器选择器：
    - 输入：列索引 j + 列的缺失率 miss_rate_j
    - 输出：该列推荐的基器名字（'BR' / 'GB' / 'RF' / 'ET' / 'LR'）
    """
    def __init__(self, rf_model, label_encoder,
                 col_mean, col_std, col_range, max_corr, mean_corr,
                 missing_rates):
        self.rf_model = rf_model
        self.label_encoder = label_encoder
        self.col_mean = col_mean
        self.col_std = col_std
        self.col_range = col_range
        self.max_corr = max_corr
        self.mean_corr = mean_corr
        self.missing_rates = np.array(missing_rates, dtype=float)

        # 记录特征列顺序，必须和训练时一致
        self.feature_names_ = ["rate", "col_idx", "col_mean",
                               "col_std", "col_range", "max_corr", "mean_corr"]

    def _closest_rate(self, rate: float) -> float:
        idx = np.argmin(np.abs(self.missing_rates - rate))
        return float(self.missing_rates[idx])

    def _build_meta_feature(self, col_idx: int, miss_rate_j: float) -> np.ndarray:
        # 把 miss_rate_j 映射到最近的训练档位（0.05,0.1,...）
        rate_bin = self._closest_rate(float(miss_rate_j))
        j = int(col_idx)

        feat = {
            "rate":      rate_bin,
            "col_idx":   j,
            "col_mean":  float(self.col_mean[j]),
            "col_std":   float(self.col_std[j]),
            "col_range": float(self.col_range[j]),
            "max_corr":  float(self.max_corr[j]),
            "mean_corr": float(self.mean_corr[j]),
        }
        return np.array([feat[name] for name in self.feature_names_], dtype=float)

    def predict_imp(self, col_idx: int, miss_rate_j: float) -> str:
        x = self._build_meta_feature(col_idx, miss_rate_j).reshape(1, -1)
        y_pred_enc = self.rf_model.predict(x)[0]
        return self.label_encoder.inverse_transform([y_pred_enc])[0]

# ===== 4. 实例化一个 selector，后面给插补器用 =====
meta_selector = MetaImputerSelector(
    rf_model=rf_selector,
    label_encoder=imp_label_encoder,
    col_mean=col_mean,
    col_std=col_std,
    col_range=col_range,
    max_corr=max_corr,
    mean_corr=mean_corr,
    missing_rates=missing_rates  # 你之前的 [0.05,...,0.8]
)


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer

class MetaColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 meta_selector: MetaImputerSelector,
                 missing_rates: list,
                 random_state: int = 0):
        """
        meta_selector: 上面训练好的 MetaImputerSelector 实例
        missing_rates: 缺失率档位列表 [0.05, 0.10, ..., 0.80]
        """
        self.meta_selector = meta_selector
        self.missing_rates = np.array(missing_rates, dtype=float)
        self.random_state = random_state

    def fit(self, X, y=None, rate=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 记录一个全局缺失率（备用，不一定用得到）
        if rate is None:
            self.global_miss_rate_ = float(np.isnan(X).mean())
        else:
            self.global_miss_rate_ = float(rate)

        # 1）先做一次均值粗插补
        self.simple_imp_ = SimpleImputer(strategy="mean")
        X_simple = self.simple_imp_.fit_transform(X)

        self.col_models_ = {}
        self.col_chosen_imp_ = {}   # 记录每列选了哪个基器
        self.col_miss_rate_ = {}    # 记录每列缺失率

        # 2）对每一列单独选择基器 & 训练回归模型
        for j in range(n_features):
            mask_j = np.isnan(X[:, j])
            if mask_j.sum() == 0:
                continue
            if (~mask_j).sum() < 5:
                continue

            miss_rate_j = mask_j.mean()
            self.col_miss_rate_[j] = miss_rate_j

            # ★ 核心：用元学习选择该列的基器类型
            imp_name = self.meta_selector.predict_imp(col_idx=j, miss_rate_j=miss_rate_j)
            self.col_chosen_imp_[j] = imp_name

            # 按缺失率档位选择一套基器族（和之前一致）
            chosen_rate_j = self.meta_selector._closest_rate(miss_rate_j)
            base_regs = get_base_regressors_for_rate_for_column_adp(
                chosen_rate_j, random_state=self.random_state
            ) if "get_base_regressors_for_rate_for_column_adp" in globals() else \
                get_base_regressors_for_rate(
                    chosen_rate_j, random_state=self.random_state
                )

            if imp_name not in base_regs:
                imp_name = "LR"
            reg = clone(base_regs[imp_name])

            X_other = np.delete(X_simple, j, axis=1)
            X_other_train = X_other[~mask_j]
            y_train = X[~mask_j, j]

            reg.fit(X_other_train, y_train)
            self.col_models_[j] = reg

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        X_imp = self.simple_imp_.transform(X)

        for j, reg in self.col_models_.items():
            mask_j = np.isnan(X[:, j])
            if mask_j.sum() == 0:
                continue
            X_other = np.delete(X_imp, j, axis=1)
            X_other_missing = X_other[mask_j]
            y_pred = reg.predict(X_other_missing)
            X_imp[mask_j, j] = y_pred

        return X_imp

    def fit_transform(self, X, y=None, rate=None):
        self.fit(X, y=y, rate=rate)
        return self.transform(X)


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np

class ClusterMetaColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    """
    簇级 + 特征级 + 元学习 三合一插补器：
    - 外层：先用 clusterer(KMeans/FCM) 把样本分簇
    - 内层：
        * 一个全局 MetaColumnAdaptiveImputer
        * 每个簇（样本数够多）一个簇内 MetaColumnAdaptiveImputer
    """
    def __init__(self,
                 clusterer,            # 已在完整数据上 fit 好的 kmeans / fcm
                 meta_selector,        # 之前训练好的 MetaImputerSelector
                 missing_rates: list,
                 random_state: int = 0,
                 min_cluster_size: int = 30):
        self.clusterer = clusterer
        self.meta_selector = meta_selector
        self.missing_rates = np.array(missing_rates, dtype=float)
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 1) 用均值先粗插补一遍，用这版数据来做聚类预测
        self.cluster_simple_imp_ = SimpleImputer(strategy="mean")
        X_simple = self.cluster_simple_imp_.fit_transform(X)

        clusters = self.clusterer.predict(X_simple)
        self.train_clusters_ = clusters
        unique_clusters = np.unique(clusters)

        # 2) 先训练一个「全局」MetaColumnAdaptiveImputer，兜底用
        self.global_meta_imp_ = MetaColumnAdaptiveImputer(
            meta_selector=self.meta_selector,
            missing_rates=self.missing_rates,
            random_state=self.random_state
        )
        self.global_meta_imp_.fit(X)

        # 3) 针对每个簇，单独训练一个簇内 MetaColumnAdaptiveImputer
        self.cluster_imputers_ = {}
        self.cluster_sizes_ = {}

        for c in unique_clusters:
            idx_c = np.where(clusters == c)[0]
            X_c = X[idx_c, :]
            self.cluster_sizes_[int(c)] = len(idx_c)

            if len(idx_c) < self.min_cluster_size:
                # 簇太小，后面 transform 时用「全局」兜底
                print(f"[ClusterMeta] 簇 {c} 样本数 {len(idx_c)} < {self.min_cluster_size}，使用全局模型兜底。")
                continue

            meta_col_imp = MetaColumnAdaptiveImputer(
                meta_selector=self.meta_selector,
                missing_rates=self.missing_rates,
                random_state=self.random_state
            )
            meta_col_imp.fit(X_c)

            self.cluster_imputers_[int(c)] = meta_col_imp
            print(f"[ClusterMeta] 簇 {c} 训练完成，样本数 {len(idx_c)}。")

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 1) 用同一个均值插补器粗插补，用于聚类
        X_simple = self.cluster_simple_imp_.transform(X)
        clusters = self.clusterer.predict(X_simple)

        # 先用「全局」MetaColumnAdaptive 大致补一版
        X_imp = self.global_meta_imp_.transform(X)

        unique_clusters = np.unique(clusters)

        # 2) 对有簇内模型的簇，再用簇内模型细化替换
        for c in unique_clusters:
            idx_c = np.where(clusters == c)[0]
            X_c = X[idx_c, :]

            if int(c) not in self.cluster_imputers_:
                # 这个簇没有单独模型（样本太少），保持全局结果即可
                continue

            imp_c = self.cluster_imputers_[int(c)]
            X_c_imp = imp_c.transform(X_c)
            X_imp[idx_c, :] = X_c_imp

        return X_imp

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


# In[ ]:


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge

class UltimateAdaptiveImputer(BaseEstimator, TransformerMixin):
    """
    终极大杂烩自适应插补器：
    - rate 自适应（缺失率分档）
    - column 自适应（col 级别选基器）
    - cluster 自适应（簇内单独建模）
    - meta_selector（可选，用元学习器选基器）
    - 不确定性感知（每个被插补位置给出 std，保存在 self.U_）

    注意：这里采用“均值粗插补 + 回归插补”的列级框架，
    不是 sklearn IterativeImputer 的多轮 MICE，但思路一致：其它列预测这一列。
    """

    def __init__(self,
                 clusterer=None,
                 best_imp_by_rate=None,
                 best_imp_by_rate_col=None,
                 meta_selector=None,
                 missing_rates=None,
                 random_state=0,
                 n_ensemble=5,
                 use_bayes_std=True,
                 initial_strategy="mean",
                 min_cluster_size=30,
                 min_non_missing=5):
        # 这些都是“配置”，直接存起来
        self.clusterer = clusterer
        self.best_imp_by_rate = best_imp_by_rate
        self.best_imp_by_rate_col = best_imp_by_rate_col
        self.meta_selector = meta_selector
        self.missing_rates = np.array(missing_rates, dtype=float) if missing_rates is not None else None
        self.random_state = random_state
        self.n_ensemble = n_ensemble
        self.use_bayes_std = use_bayes_std
        self.initial_strategy = initial_strategy
        self.min_cluster_size = min_cluster_size
        self.min_non_missing = min_non_missing

    # ---------- 工具：找最近缺失率档位 ----------
    def _closest_rate(self, rate):
        if self.missing_rates is None:
            return float(rate)
        idx = np.argmin(np.abs(self.missing_rates - rate))
        return float(self.missing_rates[idx])

    # ---------- 工具：根据 meta / rmse 表 / rate 选基器名 ----------
    def _select_imp_name(self, rate_bin, col_idx, miss_rate_j):
        col_idx = int(col_idx)
        miss_rate_j = float(miss_rate_j)

        # 1) 优先用 meta_selector.predict_imp(col_idx, miss_rate_j)
        if (self.meta_selector is not None and
                hasattr(self.meta_selector, "predict_imp")):
            return self.meta_selector.predict_imp(col_idx, miss_rate_j)

        # 2) 退回列级 rmse 表
        if self.best_imp_by_rate_col is not None:
            key = (rate_bin, col_idx)
            if key in self.best_imp_by_rate_col:
                return self.best_imp_by_rate_col[key]

        # 3) 再退回 rate 级 rmse 表
        if self.best_imp_by_rate is not None:
            if rate_bin in self.best_imp_by_rate:
                return self.best_imp_by_rate[rate_bin]

        # 4) 都没有就 LR
        return "LR"

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 记录原始缺失
        self.mask_missing_ = np.isnan(X)

        # 先做一次粗插补
        self.simple_imp_ = SimpleImputer(strategy=self.initial_strategy)
        X_simple = self.simple_imp_.fit_transform(X)

        # 簇标签：如果没有 clusterer，就全放到簇 0
        if self.clusterer is not None:
            cluster_labels = self.clusterer.predict(X_simple)
        else:
            cluster_labels = np.zeros(n_samples, dtype=int)

        self.cluster_labels_ = cluster_labels
        self.unique_clusters_ = np.unique(cluster_labels)

        # 存 (cluster, col) -> 模型 + 信息
        self.models_ = {}

        rng = np.random.default_rng(self.random_state)

        for c in self.unique_clusters_:
            idx_c = np.where(cluster_labels == c)[0]
            if len(idx_c) < self.min_cluster_size:
                # 簇太小，不在簇内建模，后面用粗插补
                continue

            X_c = X[idx_c]
            X_simple_c = X_simple[idx_c]

            for j in range(n_features):
                # 这一簇里这一列的缺失 / 非缺
                mask_j_global = np.isnan(X[:, j])
                mask_j_c = mask_j_global[idx_c]

                n_miss_c = mask_j_c.sum()
                n_obs_c = (~mask_j_c).sum()
                if n_miss_c == 0 or n_obs_c < self.min_non_missing:
                    continue

                # 簇内这一列的缺失率，映射到 rate 档位
                miss_rate_cj = float(mask_j_c.mean())
                rate_bin = self._closest_rate(miss_rate_cj)

                # 选基器名（融合 meta + rate + col）
                imp_name = self._select_imp_name(rate_bin, j, miss_rate_cj)

                # 拿到这一档 rate 下的基回归器家族
                base_regs = get_base_regressors_for_rate(
                    rate_bin, random_state=self.random_state
                )
                if imp_name not in base_regs:
                    imp_name = "LR"
                base_reg = base_regs[imp_name]

                # 构造簇内训练数据：其它列 -> 当前列
                X_other_c = np.delete(X_simple_c, j, axis=1)
                X_other_train = X_other_c[~mask_j_c]
                y_train = X_c[~mask_j_c, j]

                # ========== 1) BayesianRidge: 自带方差 ==========
                if isinstance(base_reg, BayesianRidge) and self.use_bayes_std:
                    reg = clone(base_reg)
                    reg.fit(X_other_train, y_train)
                    self.models_[(c, j)] = {
                        "mode": "bayes",
                        "models": [reg],
                        "imp_name": imp_name,
                        "rate_bin": rate_bin,
                    }
                else:
                    # ========== 2) 其它基器：小 ensemble 近似不确定性 ==========
                    models = []
                    for k in range(self.n_ensemble):
                        reg_k = clone(base_reg)
                        if hasattr(reg_k, "random_state"):
                            reg_k.random_state = self.random_state + k
                        reg_k.fit(X_other_train, y_train)
                        models.append(reg_k)

                    self.models_[(c, j)] = {
                        "mode": "ensemble",
                        "models": models,
                        "imp_name": imp_name,
                        "rate_bin": rate_bin,
                    }

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 粗插补
        X_imp = self.simple_imp_.transform(X)

        # 初始化不确定性矩阵
        U = np.zeros_like(X_imp, dtype=float)

        # 用当前插补后的 X_imp 再算簇标签（模拟真实应用：测试集也是先粗插补再聚类）
        if self.clusterer is not None:
            cluster_labels = self.clusterer.predict(X_imp)
        else:
            cluster_labels = np.zeros(n_samples, dtype=int)

        for (c, j), info in self.models_.items():
            idx_c = np.where(cluster_labels == c)[0]
            if len(idx_c) == 0:
                continue

            # 只对“原始 X 里缺失”的位置做精细插补
            mask_j_global = np.isnan(X[:, j])
            mask_j_c = mask_j_global[idx_c]
            if mask_j_c.sum() == 0:
                continue

            X_other = np.delete(X_imp[idx_c], j, axis=1)
            X_other_missing = X_other[mask_j_c]

            mode = info["mode"]
            models = info["models"]

            if mode == "bayes":
                reg = models[0]
                y_mean, y_std = reg.predict(X_other_missing, return_std=True)
            else:
                preds = []
                for reg in models:
                    preds.append(reg.predict(X_other_missing))
                preds = np.vstack(preds)         # (n_models, n_missing)
                y_mean = preds.mean(axis=0)
                y_std = preds.std(axis=0, ddof=1)

            global_idx_missing = idx_c[mask_j_c]
            X_imp[global_idx_missing, j] = y_mean
            U[global_idx_missing, j] = y_std

        self.U_ = U
        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

# =========================
# 工具：统一的评估函数
# =========================
def eval_imputer(imputer, X_mis, X_full, miss_pos, name):
    """
    给定一个插补器：
    - 在 X_mis 上 fit_transform
    - 只在原本缺失的位置上计算 RMSE
    - 打印时间和 RMSE，并返回 (time, rmse)
    """
    t0 = time.time()
    X_imp = imputer.fit_transform(X_mis)
    t1 = time.time()

    r_idx, c_idx = miss_pos
    mse = mean_squared_error(X_full[r_idx, c_idx], X_imp[r_idx, c_idx])
    rmse = mse ** 0.5
    print(f"{name:30s} | 时间: {t1 - t0:7.3f} s | RMSE: {rmse:.4f}")
    return t1 - t0, rmse


# =========================
# 三种缺失机制的数据集列表（每个长度 = 9）
# =========================
mech2datasets = {
    "mcar": baseline_mis_mcar_datasets,  # [df_005, df_01, ..., df_08]
    "mar":  baseline_mis_mar_datasets,
    "mnar": baseline_mis_mnar_datasets,
}

# =========================
# 三种机制各自的“最佳插补器策略表”
# （你前面已经用 rmse_df_* 算好了）
# =========================
best_imp_by_rate_dict = {
    "mcar": best_imp_by_rate_mcar,
    "mar":  best_imp_by_rate_mar,
    "mnar": best_imp_by_rate_mnar,
}

best_imp_by_rate_col_dict = {
    "mcar": best_imp_by_rate_col_mcar,
    "mar":  best_imp_by_rate_col_mar,
    "mnar": best_imp_by_rate_col_mnar,
}

# =========================
# 统一真值
# =========================
X_full = train_data_Eclass.values

# =========================
# 主循环：三种机制 × 九种缺失率 × 多种插补方法
# =========================
all_results = []

for mech_name, ds_list in mech2datasets.items():
    print(f"\n######## 机制: {mech_name} ########")

    # 这一机制专用的策略
    best_imp_by_rate     = best_imp_by_rate_dict[mech_name]
    best_imp_by_rate_col = best_imp_by_rate_col_dict[mech_name]

    for rate, df_mis in zip(missing_rates, ds_list):
        X_mis = df_mis.values
        mask  = np.isnan(X_mis)
        miss_pos = np.where(mask)

        print(f"\n=== 机制 {mech_name} | 缺失率 {rate:.2f} ===")
        print(f"估计缺失率: {mask.mean():.3f}")

        # ---------- 1) Baseline：RF + 20 次迭代的 MICE ----------
        rf_reg = RandomForestRegressor(
            n_estimators=300, random_state=0, n_jobs=-1
        )
        rf_imp_baseline = IterativeImputer(
            estimator=rf_reg,
            max_iter=20,
            tol=1e-3,
            random_state=0
        )
        t, rmse = eval_imputer(
            rf_imp_baseline, X_mis, X_full, miss_pos,
            "Baseline RF(20 iter)"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "Baseline_RF20",
            "time": t,
            "rmse": rmse
        })

        # ---------- 2) 数据集级自适应：RateAdaptiveImputer ----------
        adp_imp = RateAdaptiveImputer(
            best_imp_by_rate=best_imp_by_rate,   # 这一机制自己的策略
            missing_rates=missing_rates,
            random_state=0,
            tol=1e-3
        )
        t, rmse = eval_imputer(
            adp_imp, X_mis, X_full, miss_pos,
            "RateAdaptiveImputer"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "RateAdaptive",
            "time": t,
            "rmse": rmse
        })

        # ---------- 3) 特征级自适应：ColumnAdaptiveImputer ----------
        col_adp_imp = ColumnAdaptiveImputer(
            best_imp_by_rate_col=best_imp_by_rate_col,  # 这一机制自己的特征级策略
            missing_rates=missing_rates,
            random_state=0
        )
        t, rmse = eval_imputer(
            col_adp_imp, X_mis, X_full, miss_pos,
            "ColumnAdaptiveImputer"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "ColumnAdaptive",
            "time": t,
            "rmse": rmse
        })

        # ---------- 4) 元学习特征级自适应：MetaColumnAdaptiveImputer ----------
        meta_col_adp_imp = MetaColumnAdaptiveImputer(
            meta_selector=meta_selector,    # 之前训练好的 meta 选择器
            missing_rates=missing_rates,
            random_state=0
        )
        t, rmse = eval_imputer(
            meta_col_adp_imp, X_mis, X_full, miss_pos,
            "MetaColumnAdaptive"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "MetaColumnAdaptive",
            "time": t,
            "rmse": rmse
        })

        # ---------- 5) 簇 + 元学习：ClusterMetaColumnAdaptiveImputer ----------
        cluster_meta_imp = ClusterMetaColumnAdaptiveImputer(
            clusterer=kmeans,              # 在完整 train_data_Eclass 上 fit 好的 KMeans/FCM
            meta_selector=meta_selector,   # 同一个 meta_selector
            missing_rates=missing_rates,
            random_state=0,
            min_cluster_size=30
        )
        t, rmse = eval_imputer(
            cluster_meta_imp, X_mis, X_full, miss_pos,
            "ClusterMetaColumnAdaptive"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "ClusterMetaColumnAdaptive",
            "time": t,
            "rmse": rmse
        })

        # ---------- 6) 不确定性感知特征级自适应 ----------
        ua_col_imp = UncertaintyAwareColumnAdaptiveImputer(
            best_imp_by_rate_col=best_imp_by_rate_col,
            missing_rates=missing_rates,
            random_state=0,
            n_ensemble=5,        # 可调：越大越稳、越慢
            use_bayes_std=True,  # BR 用自身方差，其它用 ensemble 方差
            initial_strategy="mean"
        )
        t, rmse = eval_imputer(
            ua_col_imp, X_mis, X_full, miss_pos,
            "UncertaintyAwareColumnAdaptive"
        )
        all_results.append({
            "mech": mech_name,
            "rate": rate,
            "method": "UncertaintyAwareColumnAdaptive",
            "time": t,
            "rmse": rmse
        })


# =========================
# 汇总结果：三机制 × 九缺失率 × 多方法
# =========================
results_df = pd.DataFrame(all_results)

# 1）按机制分别打印 RMSE 表
for mech_name, sub in results_df.groupby("mech"):
    print(f"\n=== 机制 {mech_name} | 各缺失率下，不同方法的 RMSE ===")
    print(
        sub
        .pivot(index="rate", columns="method", values="rmse")
        .sort_index()
    )

    print(f"\n=== 机制 {mech_name} | 各缺失率下，不同方法的运行时间（秒） ===")
    print(
        sub
        .pivot(index="rate", columns="method", values="time")
        .sort_index()
    )

# 2）保存总表（后面想怎么画图都可以）
results_df.to_csv("adaptive_imputer_compare_all_mechs.csv",
                  index=False, encoding="utf-8-sig")
print("\n[已保存] 三种机制总结果到 adaptive_imputer_compare_all_mechs.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




