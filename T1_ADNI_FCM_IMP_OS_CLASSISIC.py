#!/usr/bin/env python
# coding: utf-8

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


ori_data = pd.read_csv('COM ADNI DATA.csv',header = None)
data_meta=ori_data.copy()

# 标准化
data_meta=data_meta[1:][:]
data_meta_nor=preprocessing.scale(data_meta.iloc[:,2:10])
data=data_meta_nor.copy()
data = pd.DataFrame(data_meta_nor)
data[8]=data_meta.iloc[:,1].reset_index(drop=True).astype('int')

#相关矩阵
data= pd.DataFrame(data)
data.corr()
plt.figure(figsize=(25,13))
sns.heatmap(data.corr(), annot = True)
data=pd.DataFrame(data)
data


ori_data


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE,ADASYN,SMOTE

X= data.iloc[:,0:8]  # 所有行都要 从1开始 到最后 （如果是取到最后 冒号后面的可以省略）
y=data.iloc[:,8]
print('不经过任何采样处理的原始 y_train 中的分类情况：{}'.format(Counter(y)))

# 综合采样（先过采样再欠采样）
kos = SMOTETomek(random_state=0)  # 综合采样
X_kos, y_kos = kos.fit_resample(X,y)
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


cv_outer = StratifiedKFold(n_splits=5, random_state=7,shuffle=True)
i=1

if not os.path.exists('train data'):
    os.mkdir('train data')
if not os.path.exists('test data'):
    os.mkdir('test data')

for train_idx, test_idx in tqdm(cv_outer.split(X, y)):
    train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
    train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]
    
    train = np.column_stack((train_data, train_target))
    test = np.column_stack((test_data, test_target ))

    train_data = 'train data/train_set{}.csv'.format(i)
    test_data = 'test data/test_set{}.csv'.format(i)
    if os.path.exists(train_data):
        os.remove(train_data)
        pd.DataFrame(train).to_csv(train_data, index=False, encoding='utf-8')
    else:
        pd.DataFrame(train).to_csv(train_data, index=False, encoding='utf-8')
        
    if os.path.exists(test_data):
        os.remove(test_data)
        pd.DataFrame(test).to_csv(test_data, index=False, encoding='utf-8')
    else:
        pd.DataFrame(test).to_csv(test_data, index=False, encoding='utf-8')
    i=i+1


# 模拟 --MCAR--缺失模式
def create_mcar_single(df, missing_column, p_missing, random_state=709):
    np.random.seed(random_state)
    indices = [df.sample(n = 1).index[0] for i in range(round(p_missing * df.shape[0]))]
    while len(set(indices)) < round(p_missing * df.shape[0]):
        indices.append(df.sample(n = 1).index[0])
    mcar_column = [1 if i in indices else 0 for i in range(df.shape[0])]
    
    df_new = df.copy()
    for i in range(len(mcar_column)):
        if mcar_column[i] == 1:
            df_new[missing_column][i] = '?'      
    df_new = df_new.replace('?', np.nan)
    return df_new

def create_mcar_mult(df, mising_column, p_missing, random_state):
    df_new = df.copy()
    for i in range(len(mising_column)):
        tmp = create_mcar_single(df, mising_column[i], p_missing, random_state=random_state+i)
        df_new[mising_column[i]] = tmp[mising_column[i]] 
    return df_new

def create_mcar(df, missing_column, p_missing, random_state=709):
    if (type(missing_column) == str):
        df_new = create_mcar_single(df, missing_column, p_missing, random_state=709)
    elif (type(missing_column) == list):
        df_new = create_mcar_mult(df, missing_column, p_missing, random_state=709)
    else:
        raise Exception('Name of the columns should be given as either str or list. Given format was {}'.format(
            type(missing_column)))
    return df_new

def test_mcar_single(df, missing_column, p_missing):
    df_new = create_mcar_single(df, missing_column, p_missing, random_state=709)
    if (df_new[missing_column].isna().sum() == round(p_missing * df.shape[0])):
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


#模拟MNAR方法
def create_mnar(dataset_meta, p_missing):
    dataset=dataset_meta.copy()
    low = p_missing/2
    high = 1 - p_missing/2
    print("p_missing:",p_missing)
    for i in range(0,8):
        feature=dataset[i]
        low_val=np.quantile(feature,low)
        high_val=np.quantile(feature,high)
        min=feature.min()
        max=feature.max()
#         print("i:", i)
#         print("low_val:", low_val) 
#         print("high_val:",high_val)
#         print("min_value:",min)
#         print("max_value:",max)
#         missing=feature[(feature<low_val) & (feature>high_val)]
        missing_low=feature[(feature<low_val)]
        missing_high=feature[(feature>high_val)]
        missing_low_index=feature[(feature<low_val)].index.tolist()
        missing_high_index=feature[(feature>high_val)].index.tolist()
#         print("missing_low:\n",missing_low)
#         print("missing_high:\n",missing_high)
        missing_low_index.extend(missing_high_index)
        list_sort=sorted(missing_low_index)
#         print("list:\n",list_sort)
#         for j in range(len(list_sort)):
#             print("list_sort:\n",list_sort[j])    
        feature[list_sort]='?'
#         print("feature:\n",feature)
    dataset = dataset.replace('?', np.nan)
    return dataset


#模拟MAR方法
# missing_low_index=[]
# missing_high_index=[]
def create_mar(dataset_meta, p_missing):
    
    low = p_missing/14
    high = 1 - p_missing/14
    print("p_missing:",p_missing)
    dataset1=dataset_meta.copy()
    for i in range(0,8):
        dataset=dataset_meta.copy()
#         dataset1=dataset_meta.copy()
        feature=dataset1[i]
#         print("i=",i)
        list_low=[]
        list_high=[]
        for j in [x for x in range(0,8) if x!=i]:
#             print(j)
            low_val=np.quantile(dataset[j],low)
            high_val=np.quantile(dataset[j],high)
            missing_low=dataset[j][(dataset[j]<low_val)]
            missing_high=dataset[j][(dataset[j]>high_val)]
            missing_low_index=missing_low.index
            missing_high_index=missing_high.index
            list_low.extend(missing_low_index)
            list_high.extend(missing_high_index)
#             print("missing_low:\n",missing_low)
#             print("missing_high:\n",missing_high)
#             print("list_low:\n",list_low)
#             print("list_high:\n",list_high)
#  ##       missing_low_index.extend(missing_high_index)
        list_low.extend(list_high)
        list_sort=set(list_low)
        list_sort=sorted(list_sort)
#         print("list_sort:\n",list_sort)
        feature[list_sort]='?'
#         print("feature:\n",feature)
    dataset1 = dataset1.replace('?', np.nan)
    return dataset1


train_data = pd.read_csv('train data/train_set1.csv',header = None)
train_data=train_data[1:][:]
train_data_Eclass= train_data.iloc[:,0:8] 
train_data_class=train_data.iloc[:,8]
#KMeans聚类
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(train_data_Eclass) 
score = silhouette_score(train_data_Eclass,kmeans.labels_)
print(score)
print("meta data:\n",pd.value_counts(train_data_class)),pd.value_counts(kmeans.labels_),kmeans.labels_


train_data = pd.read_csv('train data/train_set1.csv',header = None)
train_data=train_data[1:][:]
train_data_Eclass= train_data.iloc[:,0:8] 
train_data_class=train_data.iloc[:,8]
#C 均值聚类
fcm = FCM(n_clusters=3,random_state=0)
fcm.fit(np.array(train_data_Eclass))
fcm_labels = fcm.predict(np.array(train_data_Eclass))
score = silhouette_score(train_data_Eclass,fcm_labels)
print(score)
print("meta data:\n",pd.value_counts(train_data_class)),pd.value_counts(fcm_labels),fcm_labels


# ONE-HOT编码
from sklearn.preprocessing import OneHotEncoder

integer_encoded=fcm_labels
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded_DF=pd.DataFrame(onehot_encoded)
onehot_encoded_DF


# MCMV
data_kmeans=train_data_Eclass.copy()
data_kmeans[8]=fcm_labels
k0=np.dot(data_kmeans.groupby(8).mean().values[0][:],data_kmeans.groupby(8).mean().values[0][:])
k1=np.dot(data_kmeans.groupby(8).mean().values[1][:],data_kmeans.groupby(8).mean().values[1][:])
k2=np.dot(data_kmeans.groupby(8).mean().values[2][:],data_kmeans.groupby(8).mean().values[2][:])

print(k0)
print(k1)
print(k2)
data_kmeans[8]=data_kmeans[8].map({0:k0,1:k1,2:k2})
data_kmeans


b=list(range(0,8))#排除分类变量再去进行缺失值模拟
mis_mcar_005=create_mcar(train_data_Eclass,b,0.05)
mis_mcar_01=create_mcar(train_data_Eclass,b,0.1)
mis_mcar_02=create_mcar(train_data_Eclass,b,0.2)
mis_mcar_03=create_mcar(train_data_Eclass,b,0.3)
mis_mcar_04=create_mcar(train_data_Eclass,b,0.4)
mis_mcar_05=create_mcar(train_data_Eclass,b,0.5)
mis_mcar_06=create_mcar(train_data_Eclass,b,0.6)
mis_mcar_07=create_mcar(train_data_Eclass,b,0.7)
mis_mcar_08=create_mcar(train_data_Eclass,b,0.8)

#MNAR
mis_mnar_005=create_mnar(train_data_Eclass,0.05)
mis_mnar_01=create_mnar(train_data_Eclass,0.1)
mis_mnar_02=create_mnar(train_data_Eclass,0.2)
mis_mnar_03=create_mnar(train_data_Eclass,0.3)
mis_mnar_04=create_mnar(train_data_Eclass,0.4)
mis_mnar_05=create_mnar(train_data_Eclass,0.5)
mis_mnar_06=create_mnar(train_data_Eclass,0.6)
mis_mnar_07=create_mnar(train_data_Eclass,0.7)
mis_mnar_08=create_mnar(train_data_Eclass,0.8)

#MAR
mis_mar_005=create_mar(train_data_Eclass,0.05)
mis_mar_01=create_mar(train_data_Eclass,0.1)
mis_mar_02=create_mar(train_data_Eclass,0.2)
mis_mar_03=create_mar(train_data_Eclass,0.3)
mis_mar_04=create_mar(train_data_Eclass,0.4)
mis_mar_05=create_mar(train_data_Eclass,0.5)
mis_mar_06=create_mar(train_data_Eclass,0.6)
mis_mar_07=create_mar(train_data_Eclass,0.7)
mis_mar_08=create_mar(train_data_Eclass,0.8)


#baseline datasets
def baseline(dataset):
    bl_dataset=dataset.copy()
    bl_dataset[8]=train_data_class
    return bl_dataset

bl_mis_mcar_005=baseline(mis_mcar_005)
bl_mis_mcar_01=baseline(mis_mcar_01)
bl_mis_mcar_02=baseline(mis_mcar_02)
bl_mis_mcar_03=baseline(mis_mcar_03)
bl_mis_mcar_04=baseline(mis_mcar_04)
bl_mis_mcar_05=baseline(mis_mcar_05)
bl_mis_mcar_06=baseline(mis_mcar_06)
bl_mis_mcar_07=baseline(mis_mcar_07)
bl_mis_mcar_08=baseline(mis_mcar_08)


bl_mis_mnar_005=baseline(mis_mnar_005)
bl_mis_mnar_01=baseline(mis_mnar_01)
bl_mis_mnar_02=baseline(mis_mnar_02)
bl_mis_mnar_03=baseline(mis_mnar_03)
bl_mis_mnar_04=baseline(mis_mnar_04)
bl_mis_mnar_05=baseline(mis_mnar_05)
bl_mis_mnar_06=baseline(mis_mnar_06)
bl_mis_mnar_07=baseline(mis_mnar_07)
bl_mis_mnar_08=baseline(mis_mnar_08)

bl_mis_mar_005=baseline(mis_mar_005)
bl_mis_mar_01=baseline(mis_mar_01)
bl_mis_mar_02=baseline(mis_mar_02)
bl_mis_mar_03=baseline(mis_mar_03)
bl_mis_mar_04=baseline(mis_mar_04)
bl_mis_mar_05=baseline(mis_mar_05)
bl_mis_mar_06=baseline(mis_mar_06)
bl_mis_mar_07=baseline(mis_mar_07)
bl_mis_mar_08=baseline(mis_mar_08)


#删除有缺失值的行-complete train data'
def  mean_imp(dataset):
#     remove_rows_dataset=dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    mean_imp_dataset=SimpleFill(fill_method='mean').fit_transform(dataset)
    return mean_imp_dataset
mean_imp_mis_mcar_005=mean_imp(bl_mis_mcar_005)
mean_imp_mis_mcar_01=mean_imp(bl_mis_mcar_01)
mean_imp_mis_mcar_02=mean_imp(bl_mis_mcar_02)
mean_imp_mis_mcar_03=mean_imp(bl_mis_mcar_03)
mean_imp_mis_mcar_04=mean_imp(bl_mis_mcar_04)
mean_imp_mis_mcar_05=mean_imp(bl_mis_mcar_05)
mean_imp_mis_mcar_06=mean_imp(bl_mis_mcar_06)
mean_imp_mis_mcar_07=mean_imp(bl_mis_mcar_07)
mean_imp_mis_mcar_08=mean_imp(bl_mis_mcar_08)

mean_imp_mis_mnar_005=mean_imp(bl_mis_mnar_005)
mean_imp_mis_mnar_01=mean_imp(bl_mis_mnar_01)
mean_imp_mis_mnar_02=mean_imp(bl_mis_mnar_02)
mean_imp_mis_mnar_03=mean_imp(bl_mis_mnar_03)
mean_imp_mis_mnar_04=mean_imp(bl_mis_mnar_04)
mean_imp_mis_mnar_05=mean_imp(bl_mis_mnar_05)
mean_imp_mis_mnar_06=mean_imp(bl_mis_mnar_06)
mean_imp_mis_mnar_07=mean_imp(bl_mis_mnar_07)
mean_imp_mis_mnar_08=mean_imp(bl_mis_mnar_08)

mean_imp_mis_mar_005=mean_imp(bl_mis_mar_005)
mean_imp_mis_mar_01=mean_imp(bl_mis_mar_01)
mean_imp_mis_mar_02=mean_imp(bl_mis_mar_02)
mean_imp_mis_mar_03=mean_imp(bl_mis_mar_03)
mean_imp_mis_mar_04=mean_imp(bl_mis_mar_04)
mean_imp_mis_mar_05=mean_imp(bl_mis_mar_05)
mean_imp_mis_mar_06=mean_imp(bl_mis_mar_06)
mean_imp_mis_mar_07=mean_imp(bl_mis_mar_07)
mean_imp_mis_mar_08=mean_imp(bl_mis_mar_08)


#删除有缺失值的行-complete train data'
def  median_imp(dataset):
#     remove_rows_dataset=dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    median_imp_dataset=SimpleFill(fill_method='median').fit_transform(dataset)
    return median_imp_dataset
median_imp_mis_mcar_005=median_imp(bl_mis_mcar_005)
median_imp_mis_mcar_01=median_imp(bl_mis_mcar_01)
median_imp_mis_mcar_02=median_imp(bl_mis_mcar_02)
median_imp_mis_mcar_03=median_imp(bl_mis_mcar_03)
median_imp_mis_mcar_04=median_imp(bl_mis_mcar_04)
median_imp_mis_mcar_05=median_imp(bl_mis_mcar_05)
median_imp_mis_mcar_06=median_imp(bl_mis_mcar_06)
median_imp_mis_mcar_07=median_imp(bl_mis_mcar_07)
median_imp_mis_mcar_08=median_imp(bl_mis_mcar_08)

median_imp_mis_mnar_005=median_imp(bl_mis_mnar_005)
median_imp_mis_mnar_01=median_imp(bl_mis_mnar_01)
median_imp_mis_mnar_02=median_imp(bl_mis_mnar_02)
median_imp_mis_mnar_03=median_imp(bl_mis_mnar_03)
median_imp_mis_mnar_04=median_imp(bl_mis_mnar_04)
median_imp_mis_mnar_05=median_imp(bl_mis_mnar_05)
median_imp_mis_mnar_06=median_imp(bl_mis_mnar_06)
median_imp_mis_mnar_07=median_imp(bl_mis_mnar_07)
median_imp_mis_mnar_08=median_imp(bl_mis_mnar_08)

median_imp_mis_mar_005=median_imp(bl_mis_mar_005)
median_imp_mis_mar_01=median_imp(bl_mis_mar_01)
median_imp_mis_mar_02=median_imp(bl_mis_mar_02)
median_imp_mis_mar_03=median_imp(bl_mis_mar_03)
median_imp_mis_mar_04=median_imp(bl_mis_mar_04)
median_imp_mis_mar_05=median_imp(bl_mis_mar_05)
median_imp_mis_mar_06=median_imp(bl_mis_mar_06)
median_imp_mis_mar_07=median_imp(bl_mis_mar_07)
median_imp_mis_mar_08=median_imp(bl_mis_mar_08)


pd.DataFrame(mean_imp_mis_mar_04)


#Add the cluster information,basedata
def add_CISCL(dataset):
    data_add_CISCL=dataset.copy()
    data_add_CISCL[9]=fcm_labels
    return data_add_CISCL

def add_CISCL_onehot(dataset):
    data_add_CISCL_onehot=dataset.copy()
    data_add_CISCL_onehot[9]=onehot_encoded_DF[0]
    data_add_CISCL_onehot[10]=onehot_encoded_DF[1]
    data_add_CISCL_onehot[11]=onehot_encoded_DF[2]
    return data_add_CISCL_onehot

def add_CISCL_MCMV(dataset):
    data_add_CISCL_MCMV=dataset.copy()
    data_add_CISCL_MCMV[9]=data_kmeans[7]
    return data_add_CISCL_MCMV

add_mis_mcar_005=add_CISCL(bl_mis_mcar_005)
add_mis_mcar_01=add_CISCL(bl_mis_mcar_01)
add_mis_mcar_02=add_CISCL(bl_mis_mcar_02)
add_mis_mcar_03=add_CISCL(bl_mis_mcar_03)
add_mis_mcar_04=add_CISCL(bl_mis_mcar_04)
add_mis_mcar_05=add_CISCL(bl_mis_mcar_05)
add_mis_mcar_06=add_CISCL(bl_mis_mcar_06)
add_mis_mcar_07=add_CISCL(bl_mis_mcar_07)
add_mis_mcar_08=add_CISCL(bl_mis_mcar_08)

add_onehot_mis_mcar_005=add_CISCL_onehot(bl_mis_mcar_005)
add_onehot_mis_mcar_01=add_CISCL_onehot(bl_mis_mcar_01)
add_onehot_mis_mcar_02=add_CISCL_onehot(bl_mis_mcar_02)
add_onehot_mis_mcar_03=add_CISCL_onehot(bl_mis_mcar_03)
add_onehot_mis_mcar_04=add_CISCL_onehot(bl_mis_mcar_04)
add_onehot_mis_mcar_05=add_CISCL_onehot(bl_mis_mcar_05)
add_onehot_mis_mcar_06=add_CISCL_onehot(bl_mis_mcar_06)
add_onehot_mis_mcar_07=add_CISCL_onehot(bl_mis_mcar_07)
add_onehot_mis_mcar_08=add_CISCL_onehot(bl_mis_mcar_08)

add_MCMV_mis_mcar_005=add_CISCL_MCMV(bl_mis_mcar_005)
add_MCMV_mis_mcar_01=add_CISCL_MCMV(bl_mis_mcar_01)
add_MCMV_mis_mcar_02=add_CISCL_MCMV(bl_mis_mcar_02)
add_MCMV_mis_mcar_03=add_CISCL_MCMV(bl_mis_mcar_03)
add_MCMV_mis_mcar_04=add_CISCL_MCMV(bl_mis_mcar_04)
add_MCMV_mis_mcar_05=add_CISCL_MCMV(bl_mis_mcar_05)
add_MCMV_mis_mcar_06=add_CISCL_MCMV(bl_mis_mcar_06)
add_MCMV_mis_mcar_07=add_CISCL_MCMV(bl_mis_mcar_07)
add_MCMV_mis_mcar_08=add_CISCL_MCMV(bl_mis_mcar_08)



add_mis_mnar_005=add_CISCL(bl_mis_mnar_005)
add_mis_mnar_01=add_CISCL(bl_mis_mnar_01)
add_mis_mnar_02=add_CISCL(bl_mis_mnar_02)
add_mis_mnar_03=add_CISCL(bl_mis_mnar_03)
add_mis_mnar_04=add_CISCL(bl_mis_mnar_04)
add_mis_mnar_05=add_CISCL(bl_mis_mnar_05)
add_mis_mnar_06=add_CISCL(bl_mis_mnar_06)
add_mis_mnar_07=add_CISCL(bl_mis_mnar_07)
add_mis_mnar_08=add_CISCL(bl_mis_mnar_08)

add_onehot_mis_mnar_005=add_CISCL_onehot(bl_mis_mnar_005)
add_onehot_mis_mnar_01=add_CISCL_onehot(bl_mis_mnar_01)
add_onehot_mis_mnar_02=add_CISCL_onehot(bl_mis_mnar_02)
add_onehot_mis_mnar_03=add_CISCL_onehot(bl_mis_mnar_03)
add_onehot_mis_mnar_04=add_CISCL_onehot(bl_mis_mnar_04)
add_onehot_mis_mnar_05=add_CISCL_onehot(bl_mis_mnar_05)
add_onehot_mis_mnar_06=add_CISCL_onehot(bl_mis_mnar_06)
add_onehot_mis_mnar_07=add_CISCL_onehot(bl_mis_mnar_07)
add_onehot_mis_mnar_08=add_CISCL_onehot(bl_mis_mnar_08)

add_MCMV_mis_mnar_005=add_CISCL_MCMV(bl_mis_mnar_005)
add_MCMV_mis_mnar_01=add_CISCL_MCMV(bl_mis_mnar_01)
add_MCMV_mis_mnar_02=add_CISCL_MCMV(bl_mis_mnar_02)
add_MCMV_mis_mnar_03=add_CISCL_MCMV(bl_mis_mnar_03)
add_MCMV_mis_mnar_04=add_CISCL_MCMV(bl_mis_mnar_04)
add_MCMV_mis_mnar_05=add_CISCL_MCMV(bl_mis_mnar_05)
add_MCMV_mis_mnar_06=add_CISCL_MCMV(bl_mis_mnar_06)
add_MCMV_mis_mnar_07=add_CISCL_MCMV(bl_mis_mnar_07)
add_MCMV_mis_mnar_08=add_CISCL_MCMV(bl_mis_mnar_08)



add_mis_mar_005=add_CISCL(bl_mis_mar_005)
add_mis_mar_01=add_CISCL(bl_mis_mar_01)
add_mis_mar_02=add_CISCL(bl_mis_mar_02)
add_mis_mar_03=add_CISCL(bl_mis_mar_03)
add_mis_mar_04=add_CISCL(bl_mis_mar_04)
add_mis_mar_05=add_CISCL(bl_mis_mar_05)
add_mis_mar_06=add_CISCL(bl_mis_mar_06)
add_mis_mar_07=add_CISCL(bl_mis_mar_07)
add_mis_mar_08=add_CISCL(bl_mis_mar_08)

add_onehot_mis_mar_005=add_CISCL_onehot(bl_mis_mar_005)
add_onehot_mis_mar_01=add_CISCL_onehot(bl_mis_mar_01)
add_onehot_mis_mar_02=add_CISCL_onehot(bl_mis_mar_02)
add_onehot_mis_mar_03=add_CISCL_onehot(bl_mis_mar_03)
add_onehot_mis_mar_04=add_CISCL_onehot(bl_mis_mar_04)
add_onehot_mis_mar_05=add_CISCL_onehot(bl_mis_mar_05)
add_onehot_mis_mar_06=add_CISCL_onehot(bl_mis_mar_06)
add_onehot_mis_mar_07=add_CISCL_onehot(bl_mis_mar_07)
add_onehot_mis_mar_08=add_CISCL_onehot(bl_mis_mar_08)

add_MCMV_mis_mar_005=add_CISCL_MCMV(bl_mis_mar_005)
add_MCMV_mis_mar_01=add_CISCL_MCMV(bl_mis_mar_01)
add_MCMV_mis_mar_02=add_CISCL_MCMV(bl_mis_mar_02)
add_MCMV_mis_mar_03=add_CISCL_MCMV(bl_mis_mar_03)
add_MCMV_mis_mar_04=add_CISCL_MCMV(bl_mis_mar_04)
add_MCMV_mis_mar_05=add_CISCL_MCMV(bl_mis_mar_05)
add_MCMV_mis_mar_06=add_CISCL_MCMV(bl_mis_mar_06)
add_MCMV_mis_mar_07=add_CISCL_MCMV(bl_mis_mar_07)
add_MCMV_mis_mar_08=add_CISCL_MCMV(bl_mis_mar_08)


#MCAR
# baseline impution(LR-MICE,iterative SVD, matrix factorization, KNN,GB-MICE,RF-MICE)

LR = LinearRegression()
# DT=DecisionTreeRegressor()
RF=RandomForestRegressor()
#梯度提升树
GB=GradientBoostingRegressor()

imp_LR = IterativeImputer(estimator=LR,missing_values=np.nan,max_iter=5, verbose=1, imputation_order='roman',random_state=0)
# imp_DT = IterativeImputer(estimator=DT,missing_values=np.nan, sample_posterior=True,max_iter=20, verbose=2, imputation_order='roman',random_state=1)
imp_RF = IterativeImputer(estimator=RF,missing_values=np.nan,max_iter=5, verbose=1, imputation_order='roman',random_state=2)
imp_GB = IterativeImputer(estimator=GB,missing_values=np.nan,max_iter=5, verbose=1, imputation_order='roman',random_state=3)

bl_mis_mcar_005_imp_LR=imp_LR.fit_transform(bl_mis_mcar_005)
bl_mis_mcar_01_imp_LR=imp_LR.fit_transform(bl_mis_mcar_01)
bl_mis_mcar_02_imp_LR=imp_LR.fit_transform(bl_mis_mcar_02)
bl_mis_mcar_03_imp_LR=imp_LR.fit_transform(bl_mis_mcar_03)
bl_mis_mcar_04_imp_LR=imp_LR.fit_transform(bl_mis_mcar_04)
bl_mis_mcar_05_imp_LR=imp_LR.fit_transform(bl_mis_mcar_05)
bl_mis_mcar_06_imp_LR=imp_LR.fit_transform(bl_mis_mcar_06)
bl_mis_mcar_07_imp_LR=imp_LR.fit_transform(bl_mis_mcar_07)
bl_mis_mcar_08_imp_LR=imp_LR.fit_transform(bl_mis_mcar_08)

mis_mcar_005_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_005)
mis_mcar_01_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_01)
mis_mcar_02_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_02)
mis_mcar_03_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_03)
mis_mcar_04_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_04)
mis_mcar_05_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_05)
mis_mcar_06_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_06)
mis_mcar_07_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_07)
mis_mcar_08_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mcar_08)

mis_mcar_005_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_005)
mis_mcar_01_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_01)
mis_mcar_02_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_02)
mis_mcar_03_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_03)
mis_mcar_04_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_04)
mis_mcar_05_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_05)
mis_mcar_06_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_06)
mis_mcar_07_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_07)
mis_mcar_08_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mcar_08)

mis_mcar_005_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_005)
mis_mcar_01_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_01)
mis_mcar_02_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_02)
mis_mcar_03_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_03)
mis_mcar_04_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_04)
mis_mcar_05_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_05)
mis_mcar_06_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_06)
mis_mcar_07_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_07)
mis_mcar_08_filled_knn = KNN(k=3).fit_transform(bl_mis_mcar_08)

bl_mis_mcar_005_imp_GB =imp_GB.fit_transform(bl_mis_mcar_005)
bl_mis_mcar_01_imp_GB =imp_GB.fit_transform(bl_mis_mcar_01)
bl_mis_mcar_02_imp_GB =imp_GB.fit_transform(bl_mis_mcar_02)
bl_mis_mcar_03_imp_GB =imp_GB.fit_transform(bl_mis_mcar_03)
bl_mis_mcar_04_imp_GB =imp_GB.fit_transform(bl_mis_mcar_04)
bl_mis_mcar_05_imp_GB =imp_GB.fit_transform(bl_mis_mcar_05)
bl_mis_mcar_06_imp_GB =imp_GB.fit_transform(bl_mis_mcar_06)
bl_mis_mcar_07_imp_GB =imp_GB.fit_transform(bl_mis_mcar_07)
bl_mis_mcar_08_imp_GB =imp_GB.fit_transform(bl_mis_mcar_08)
                                            
bl_mis_mcar_005_imp_RF =imp_RF.fit_transform(bl_mis_mcar_005)
bl_mis_mcar_01_imp_RF =imp_RF.fit_transform(bl_mis_mcar_01)
bl_mis_mcar_02_imp_RF =imp_RF.fit_transform(bl_mis_mcar_02)
bl_mis_mcar_03_imp_RF =imp_RF.fit_transform(bl_mis_mcar_03)
bl_mis_mcar_04_imp_RF =imp_RF.fit_transform(bl_mis_mcar_04)
bl_mis_mcar_05_imp_RF =imp_RF.fit_transform(bl_mis_mcar_05)
bl_mis_mcar_06_imp_RF =imp_RF.fit_transform(bl_mis_mcar_06)
bl_mis_mcar_07_imp_RF =imp_RF.fit_transform(bl_mis_mcar_07)
bl_mis_mcar_08_imp_RF =imp_RF.fit_transform(bl_mis_mcar_08)


#Add cluster dataset
#add lable
add_mis_mcar_005_imp_LR=imp_LR.fit_transform(add_mis_mcar_005)
add_mis_mcar_01_imp_LR=imp_LR.fit_transform(add_mis_mcar_01)
add_mis_mcar_02_imp_LR=imp_LR.fit_transform(add_mis_mcar_02)
add_mis_mcar_03_imp_LR=imp_LR.fit_transform(add_mis_mcar_03)
add_mis_mcar_04_imp_LR=imp_LR.fit_transform(add_mis_mcar_04)
add_mis_mcar_05_imp_LR=imp_LR.fit_transform(add_mis_mcar_05)
add_mis_mcar_06_imp_LR=imp_LR.fit_transform(add_mis_mcar_06)
add_mis_mcar_07_imp_LR=imp_LR.fit_transform(add_mis_mcar_07)
add_mis_mcar_08_imp_LR=imp_LR.fit_transform(add_mis_mcar_08)

add_mis_mcar_005_imp_GB =imp_GB.fit_transform(add_mis_mcar_005)
add_mis_mcar_01_imp_GB =imp_GB.fit_transform(add_mis_mcar_01)
add_mis_mcar_02_imp_GB =imp_GB.fit_transform(add_mis_mcar_02)
add_mis_mcar_03_imp_GB =imp_GB.fit_transform(add_mis_mcar_03)
add_mis_mcar_04_imp_GB =imp_GB.fit_transform(add_mis_mcar_04)
add_mis_mcar_05_imp_GB =imp_GB.fit_transform(add_mis_mcar_05)
add_mis_mcar_06_imp_GB =imp_GB.fit_transform(add_mis_mcar_06)
add_mis_mcar_07_imp_GB =imp_GB.fit_transform(add_mis_mcar_07)
add_mis_mcar_08_imp_GB =imp_GB.fit_transform(add_mis_mcar_08)
                                       
add_mis_mcar_005_imp_RF =imp_RF.fit_transform(add_mis_mcar_005)
add_mis_mcar_01_imp_RF =imp_RF.fit_transform(add_mis_mcar_01)
add_mis_mcar_02_imp_RF =imp_RF.fit_transform(add_mis_mcar_02)
add_mis_mcar_03_imp_RF =imp_RF.fit_transform(add_mis_mcar_03)
add_mis_mcar_04_imp_RF =imp_RF.fit_transform(add_mis_mcar_04)
add_mis_mcar_05_imp_RF =imp_RF.fit_transform(add_mis_mcar_05)
add_mis_mcar_06_imp_RF =imp_RF.fit_transform(add_mis_mcar_06)
add_mis_mcar_07_imp_RF =imp_RF.fit_transform(add_mis_mcar_07)
add_mis_mcar_08_imp_RF =imp_RF.fit_transform(add_mis_mcar_08)

#add onehot
add_onehot_mis_mcar_005_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_005)
add_onehot_mis_mcar_01_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_01)
add_onehot_mis_mcar_02_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_02)
add_onehot_mis_mcar_03_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_03)
add_onehot_mis_mcar_04_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_04)
add_onehot_mis_mcar_05_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_05)
add_onehot_mis_mcar_06_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_06)
add_onehot_mis_mcar_07_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_07)
add_onehot_mis_mcar_08_imp_LR=imp_LR.fit_transform(add_onehot_mis_mcar_08)

add_onehot_mis_mcar_005_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_005)
add_onehot_mis_mcar_01_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_01)
add_onehot_mis_mcar_02_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_02)
add_onehot_mis_mcar_03_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_03)
add_onehot_mis_mcar_04_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_04)
add_onehot_mis_mcar_05_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_05)
add_onehot_mis_mcar_06_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_06)
add_onehot_mis_mcar_07_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_07)
add_onehot_mis_mcar_08_imp_GB =imp_GB.fit_transform(add_onehot_mis_mcar_08)
                                 
add_onehot_mis_mcar_005_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_005)
add_onehot_mis_mcar_01_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_01)
add_onehot_mis_mcar_02_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_02)
add_onehot_mis_mcar_03_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_03)
add_onehot_mis_mcar_04_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_04)
add_onehot_mis_mcar_05_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_05)
add_onehot_mis_mcar_06_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_06)
add_onehot_mis_mcar_07_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_07)
add_onehot_mis_mcar_08_imp_RF =imp_RF.fit_transform(add_onehot_mis_mcar_08)

# add MCMV
add_MCMV_mis_mcar_005_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_005)
add_MCMV_mis_mcar_01_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_01)
add_MCMV_mis_mcar_02_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_02)
add_MCMV_mis_mcar_03_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_03)
add_MCMV_mis_mcar_04_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_04)
add_MCMV_mis_mcar_05_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_05)
add_MCMV_mis_mcar_06_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_06)
add_MCMV_mis_mcar_07_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_07)
add_MCMV_mis_mcar_08_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mcar_08)

add_MCMV_mis_mcar_005_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_005)
add_MCMV_mis_mcar_01_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_01)
add_MCMV_mis_mcar_02_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_02)
add_MCMV_mis_mcar_03_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_03)
add_MCMV_mis_mcar_04_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_04)
add_MCMV_mis_mcar_05_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_05)
add_MCMV_mis_mcar_06_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_06)
add_MCMV_mis_mcar_07_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_07)
add_MCMV_mis_mcar_08_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mcar_08)
                                 
add_MCMV_mis_mcar_005_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_005)
add_MCMV_mis_mcar_01_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_01)
add_MCMV_mis_mcar_02_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_02)
add_MCMV_mis_mcar_03_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_03)
add_MCMV_mis_mcar_04_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_04)
add_MCMV_mis_mcar_05_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_05)
add_MCMV_mis_mcar_06_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_06)
add_MCMV_mis_mcar_07_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_07)
add_MCMV_mis_mcar_08_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mcar_08)


#MNAR
# baseline impution(LR-MICE,iterative SVD, matrix factorization, KNN,GB-MICE,RF-MICE)
print("imp_LR")
bl_mis_mnar_005_imp_LR=imp_LR.fit_transform(bl_mis_mnar_005)
bl_mis_mnar_01_imp_LR=imp_LR.fit_transform(bl_mis_mnar_01)
bl_mis_mnar_02_imp_LR=imp_LR.fit_transform(bl_mis_mnar_02)
bl_mis_mnar_03_imp_LR=imp_LR.fit_transform(bl_mis_mnar_03)
bl_mis_mnar_04_imp_LR=imp_LR.fit_transform(bl_mis_mnar_04)
bl_mis_mnar_05_imp_LR=imp_LR.fit_transform(bl_mis_mnar_05)
bl_mis_mnar_06_imp_LR=imp_LR.fit_transform(bl_mis_mnar_06)
bl_mis_mnar_07_imp_LR=imp_LR.fit_transform(bl_mis_mnar_07)
bl_mis_mnar_08_imp_LR=imp_LR.fit_transform(bl_mis_mnar_08)

mis_mnar_005_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_005)
mis_mnar_01_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_01)
mis_mnar_02_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_02)
mis_mnar_03_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_03)
mis_mnar_04_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_04)
mis_mnar_05_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_05)
mis_mnar_06_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_06)
mis_mnar_07_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_07)
mis_mnar_08_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mnar_08)
print("MatrixFactorization")
mis_mnar_005_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_005)
mis_mnar_01_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_01)
mis_mnar_02_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_02)
mis_mnar_03_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_03)
mis_mnar_04_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_04)
mis_mnar_05_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_05)
mis_mnar_06_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_06)
mis_mnar_07_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_07)
mis_mnar_08_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mnar_08)
print("KNN")
mis_mnar_005_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_005)
mis_mnar_01_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_01)
mis_mnar_02_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_02)
mis_mnar_03_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_03)
mis_mnar_04_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_04)
mis_mnar_05_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_05)
mis_mnar_06_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_06)
mis_mnar_07_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_07)
mis_mnar_08_filled_knn = KNN(k=3).fit_transform(bl_mis_mnar_08)
print("imp_GB")
bl_mis_mnar_005_imp_GB =imp_GB.fit_transform(bl_mis_mnar_005)
bl_mis_mnar_01_imp_GB =imp_GB.fit_transform(bl_mis_mnar_01)
bl_mis_mnar_02_imp_GB =imp_GB.fit_transform(bl_mis_mnar_02)
bl_mis_mnar_03_imp_GB =imp_GB.fit_transform(bl_mis_mnar_03)
bl_mis_mnar_04_imp_GB =imp_GB.fit_transform(bl_mis_mnar_04)
bl_mis_mnar_05_imp_GB =imp_GB.fit_transform(bl_mis_mnar_05)
bl_mis_mnar_06_imp_GB =imp_GB.fit_transform(bl_mis_mnar_06)
bl_mis_mnar_07_imp_GB =imp_GB.fit_transform(bl_mis_mnar_07)
bl_mis_mnar_08_imp_GB =imp_GB.fit_transform(bl_mis_mnar_08)
print("imp_RF")                              
bl_mis_mnar_005_imp_RF =imp_RF.fit_transform(bl_mis_mnar_005)
bl_mis_mnar_01_imp_RF =imp_RF.fit_transform(bl_mis_mnar_01)
bl_mis_mnar_02_imp_RF =imp_RF.fit_transform(bl_mis_mnar_02)
bl_mis_mnar_03_imp_RF =imp_RF.fit_transform(bl_mis_mnar_03)
bl_mis_mnar_04_imp_RF =imp_RF.fit_transform(bl_mis_mnar_04)
bl_mis_mnar_05_imp_RF =imp_RF.fit_transform(bl_mis_mnar_05)
bl_mis_mnar_06_imp_RF =imp_RF.fit_transform(bl_mis_mnar_06)
bl_mis_mnar_07_imp_RF =imp_RF.fit_transform(bl_mis_mnar_07)
bl_mis_mnar_08_imp_RF =imp_RF.fit_transform(bl_mis_mnar_08)


#Add cluster dataset
#add lable
print("imp_LR")
add_mis_mnar_005_imp_LR=imp_LR.fit_transform(add_mis_mnar_005)
add_mis_mnar_01_imp_LR=imp_LR.fit_transform(add_mis_mnar_01)
add_mis_mnar_02_imp_LR=imp_LR.fit_transform(add_mis_mnar_02)
add_mis_mnar_03_imp_LR=imp_LR.fit_transform(add_mis_mnar_03)
add_mis_mnar_04_imp_LR=imp_LR.fit_transform(add_mis_mnar_04)
add_mis_mnar_05_imp_LR=imp_LR.fit_transform(add_mis_mnar_05)
add_mis_mnar_06_imp_LR=imp_LR.fit_transform(add_mis_mnar_06)
add_mis_mnar_07_imp_LR=imp_LR.fit_transform(add_mis_mnar_07)
add_mis_mnar_08_imp_LR=imp_LR.fit_transform(add_mis_mnar_08)

print("imp_GB")
add_mis_mnar_005_imp_GB =imp_GB.fit_transform(add_mis_mnar_005)
add_mis_mnar_01_imp_GB =imp_GB.fit_transform(add_mis_mnar_01)
add_mis_mnar_02_imp_GB =imp_GB.fit_transform(add_mis_mnar_02)
add_mis_mnar_03_imp_GB =imp_GB.fit_transform(add_mis_mnar_03)
add_mis_mnar_04_imp_GB =imp_GB.fit_transform(add_mis_mnar_04)
add_mis_mnar_05_imp_GB =imp_GB.fit_transform(add_mis_mnar_05)
add_mis_mnar_06_imp_GB =imp_GB.fit_transform(add_mis_mnar_06)
add_mis_mnar_07_imp_GB =imp_GB.fit_transform(add_mis_mnar_07)
add_mis_mnar_08_imp_GB =imp_GB.fit_transform(add_mis_mnar_08)

print("imp_RF")                
add_mis_mnar_005_imp_RF =imp_RF.fit_transform(add_mis_mnar_005)
add_mis_mnar_01_imp_RF =imp_RF.fit_transform(add_mis_mnar_01)
add_mis_mnar_02_imp_RF =imp_RF.fit_transform(add_mis_mnar_02)
add_mis_mnar_03_imp_RF =imp_RF.fit_transform(add_mis_mnar_03)
add_mis_mnar_04_imp_RF =imp_RF.fit_transform(add_mis_mnar_04)
add_mis_mnar_05_imp_RF =imp_RF.fit_transform(add_mis_mnar_05)
add_mis_mnar_06_imp_RF =imp_RF.fit_transform(add_mis_mnar_06)
add_mis_mnar_07_imp_RF =imp_RF.fit_transform(add_mis_mnar_07)
add_mis_mnar_08_imp_RF =imp_RF.fit_transform(add_mis_mnar_08)

#add onehot
print("imp_LR")          
add_onehot_mis_mnar_005_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_005)
add_onehot_mis_mnar_01_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_01)
add_onehot_mis_mnar_02_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_02)
add_onehot_mis_mnar_03_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_03)
add_onehot_mis_mnar_04_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_04)
add_onehot_mis_mnar_05_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_05)
add_onehot_mis_mnar_06_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_06)
add_onehot_mis_mnar_07_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_07)
add_onehot_mis_mnar_08_imp_LR=imp_LR.fit_transform(add_onehot_mis_mnar_08)

print("imp_GB")
add_onehot_mis_mnar_005_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_005)
add_onehot_mis_mnar_01_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_01)
add_onehot_mis_mnar_02_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_02)
add_onehot_mis_mnar_03_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_03)
add_onehot_mis_mnar_04_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_04)
add_onehot_mis_mnar_05_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_05)
add_onehot_mis_mnar_06_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_06)
add_onehot_mis_mnar_07_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_07)
add_onehot_mis_mnar_08_imp_GB =imp_GB.fit_transform(add_onehot_mis_mnar_08)

print("imp_RF")                               
add_onehot_mis_mnar_005_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_005)
add_onehot_mis_mnar_01_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_01)
add_onehot_mis_mnar_02_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_02)
add_onehot_mis_mnar_03_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_03)
add_onehot_mis_mnar_04_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_04)
add_onehot_mis_mnar_05_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_05)
add_onehot_mis_mnar_06_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_06)
add_onehot_mis_mnar_07_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_07)
add_onehot_mis_mnar_08_imp_RF =imp_RF.fit_transform(add_onehot_mis_mnar_08)

# add MCMV
print("imp_LR")
add_MCMV_mis_mnar_005_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_005)
add_MCMV_mis_mnar_01_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_01)
add_MCMV_mis_mnar_02_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_02)
add_MCMV_mis_mnar_03_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_03)
add_MCMV_mis_mnar_04_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_04)
add_MCMV_mis_mnar_05_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_05)
add_MCMV_mis_mnar_06_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_06)
add_MCMV_mis_mnar_07_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_07)
add_MCMV_mis_mnar_08_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mnar_08)

print("imp_GB")
add_MCMV_mis_mnar_005_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_005)
add_MCMV_mis_mnar_01_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_01)
add_MCMV_mis_mnar_02_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_02)
add_MCMV_mis_mnar_03_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_03)
add_MCMV_mis_mnar_04_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_04)
add_MCMV_mis_mnar_05_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_05)
add_MCMV_mis_mnar_06_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_06)
add_MCMV_mis_mnar_07_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_07)
add_MCMV_mis_mnar_08_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mnar_08)
print("imp_RF")                                 
add_MCMV_mis_mnar_005_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_005)
add_MCMV_mis_mnar_01_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_01)
add_MCMV_mis_mnar_02_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_02)
add_MCMV_mis_mnar_03_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_03)
add_MCMV_mis_mnar_04_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_04)
add_MCMV_mis_mnar_05_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_05)
add_MCMV_mis_mnar_06_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_06)
add_MCMV_mis_mnar_07_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_07)
add_MCMV_mis_mnar_08_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mnar_08)


#MAR
# baseline impution(LR-MICE,iterative SVD, matrix factorization, KNN,GB-MICE,RF-MICE)
bl_mis_mar_005_imp_LR=imp_LR.fit_transform(bl_mis_mar_005)
bl_mis_mar_01_imp_LR=imp_LR.fit_transform( bl_mis_mar_01)
bl_mis_mar_02_imp_LR=imp_LR.fit_transform( bl_mis_mar_02)
bl_mis_mar_03_imp_LR=imp_LR.fit_transform( bl_mis_mar_03)
bl_mis_mar_04_imp_LR=imp_LR.fit_transform( bl_mis_mar_04)
bl_mis_mar_05_imp_LR=imp_LR.fit_transform( bl_mis_mar_05)
bl_mis_mar_06_imp_LR=imp_LR.fit_transform( bl_mis_mar_06)
bl_mis_mar_07_imp_LR=imp_LR.fit_transform( bl_mis_mar_07)
bl_mis_mar_08_imp_LR=imp_LR.fit_transform( bl_mis_mar_08)
 
mis_mar_005_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_005)
mis_mar_01_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_01)
mis_mar_02_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_02)
mis_mar_03_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_03)
mis_mar_04_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_04)
mis_mar_05_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_05)
mis_mar_06_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_06)
mis_mar_07_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_07)
mis_mar_08_filled_SVD  = IterativeSVD(rank=8).fit_transform(bl_mis_mar_08)

mis_mar_005_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_005)
mis_mar_01_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_01)
mis_mar_02_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_02)
mis_mar_03_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_03)
mis_mar_04_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_04)
mis_mar_05_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_05)
mis_mar_06_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_06)
mis_mar_07_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_07)
mis_mar_08_filled_Matrix = MatrixFactorization().fit_transform(bl_mis_mar_08)

mis_mar_005_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_005)
mis_mar_01_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_01)
mis_mar_02_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_02)
mis_mar_03_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_03)
mis_mar_04_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_04)
mis_mar_05_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_05)
mis_mar_06_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_06)
mis_mar_07_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_07)
mis_mar_08_filled_knn = KNN(k=3).fit_transform(bl_mis_mar_08)

bl_mis_mar_005_imp_GB =imp_GB.fit_transform(bl_mis_mar_005)
bl_mis_mar_01_imp_GB =imp_GB.fit_transform(bl_mis_mar_01)
bl_mis_mar_02_imp_GB =imp_GB.fit_transform(bl_mis_mar_02)
bl_mis_mar_03_imp_GB =imp_GB.fit_transform(bl_mis_mar_03)
bl_mis_mar_04_imp_GB =imp_GB.fit_transform(bl_mis_mar_04)
bl_mis_mar_05_imp_GB =imp_GB.fit_transform(bl_mis_mar_05)
bl_mis_mar_06_imp_GB =imp_GB.fit_transform(bl_mis_mar_06)
bl_mis_mar_07_imp_GB =imp_GB.fit_transform(bl_mis_mar_07)
bl_mis_mar_08_imp_GB =imp_GB.fit_transform(bl_mis_mar_08)
                                  
bl_mis_mar_005_imp_RF =imp_RF.fit_transform(bl_mis_mar_005)
bl_mis_mar_01_imp_RF =imp_RF.fit_transform(bl_mis_mar_01)
bl_mis_mar_02_imp_RF =imp_RF.fit_transform(bl_mis_mar_02)
bl_mis_mar_03_imp_RF =imp_RF.fit_transform(bl_mis_mar_03)
bl_mis_mar_04_imp_RF =imp_RF.fit_transform(bl_mis_mar_04)
bl_mis_mar_05_imp_RF =imp_RF.fit_transform(bl_mis_mar_05)
bl_mis_mar_06_imp_RF =imp_RF.fit_transform(bl_mis_mar_06)
bl_mis_mar_07_imp_RF =imp_RF.fit_transform(bl_mis_mar_07)
bl_mis_mar_08_imp_RF =imp_RF.fit_transform(bl_mis_mar_08)


#Add cluster dataset
#add lable
add_mis_mar_005_imp_LR=imp_LR.fit_transform(add_mis_mar_005)
add_mis_mar_01_imp_LR=imp_LR.fit_transform(add_mis_mar_01)
add_mis_mar_02_imp_LR=imp_LR.fit_transform(add_mis_mar_02)
add_mis_mar_03_imp_LR=imp_LR.fit_transform(add_mis_mar_03)
add_mis_mar_04_imp_LR=imp_LR.fit_transform(add_mis_mar_04)
add_mis_mar_05_imp_LR=imp_LR.fit_transform(add_mis_mar_05)
add_mis_mar_06_imp_LR=imp_LR.fit_transform(add_mis_mar_06)
add_mis_mar_07_imp_LR=imp_LR.fit_transform(add_mis_mar_07)
add_mis_mar_08_imp_LR=imp_LR.fit_transform(add_mis_mar_08)

add_mis_mar_005_imp_GB =imp_GB.fit_transform(add_mis_mar_005)
add_mis_mar_01_imp_GB =imp_GB.fit_transform(add_mis_mar_01)
add_mis_mar_02_imp_GB =imp_GB.fit_transform(add_mis_mar_02)
add_mis_mar_03_imp_GB =imp_GB.fit_transform(add_mis_mar_03)
add_mis_mar_04_imp_GB =imp_GB.fit_transform(add_mis_mar_04)
add_mis_mar_05_imp_GB =imp_GB.fit_transform(add_mis_mar_05)
add_mis_mar_06_imp_GB =imp_GB.fit_transform(add_mis_mar_06)
add_mis_mar_07_imp_GB =imp_GB.fit_transform(add_mis_mar_07)
add_mis_mar_08_imp_GB =imp_GB.fit_transform(add_mis_mar_08)
                           
add_mis_mar_005_imp_RF =imp_RF.fit_transform(add_mis_mar_005)
add_mis_mar_01_imp_RF =imp_RF.fit_transform(add_mis_mar_01)
add_mis_mar_02_imp_RF =imp_RF.fit_transform(add_mis_mar_02)
add_mis_mar_03_imp_RF =imp_RF.fit_transform(add_mis_mar_03)
add_mis_mar_04_imp_RF =imp_RF.fit_transform(add_mis_mar_04)
add_mis_mar_05_imp_RF =imp_RF.fit_transform(add_mis_mar_05)
add_mis_mar_06_imp_RF =imp_RF.fit_transform(add_mis_mar_06)
add_mis_mar_07_imp_RF =imp_RF.fit_transform(add_mis_mar_07)
add_mis_mar_08_imp_RF =imp_RF.fit_transform(add_mis_mar_08)

#add onehot
add_onehot_mis_mar_005_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_005)
add_onehot_mis_mar_01_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_01)
add_onehot_mis_mar_02_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_02)
add_onehot_mis_mar_03_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_03)
add_onehot_mis_mar_04_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_04)
add_onehot_mis_mar_05_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_05)
add_onehot_mis_mar_06_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_06)
add_onehot_mis_mar_07_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_07)
add_onehot_mis_mar_08_imp_LR=imp_LR.fit_transform(add_onehot_mis_mar_08)

add_onehot_mis_mar_005_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_005)
add_onehot_mis_mar_01_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_01)
add_onehot_mis_mar_02_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_02)
add_onehot_mis_mar_03_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_03)
add_onehot_mis_mar_04_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_04)
add_onehot_mis_mar_05_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_05)
add_onehot_mis_mar_06_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_06)
add_onehot_mis_mar_07_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_07)
add_onehot_mis_mar_08_imp_GB =imp_GB.fit_transform(add_onehot_mis_mar_08)
                               
add_onehot_mis_mar_005_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_005)
add_onehot_mis_mar_01_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_01)
add_onehot_mis_mar_02_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_02)
add_onehot_mis_mar_03_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_03)
add_onehot_mis_mar_04_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_04)
add_onehot_mis_mar_05_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_05)
add_onehot_mis_mar_06_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_06)
add_onehot_mis_mar_07_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_07)
add_onehot_mis_mar_08_imp_RF =imp_RF.fit_transform(add_onehot_mis_mar_08)

# add MCMV
add_MCMV_mis_mar_005_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_005)
add_MCMV_mis_mar_01_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_01)
add_MCMV_mis_mar_02_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_02)
add_MCMV_mis_mar_03_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_03)
add_MCMV_mis_mar_04_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_04)
add_MCMV_mis_mar_05_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_05)
add_MCMV_mis_mar_06_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_06)
add_MCMV_mis_mar_07_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_07)
add_MCMV_mis_mar_08_imp_LR=imp_LR.fit_transform(add_MCMV_mis_mar_08)

add_MCMV_mis_mar_005_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_005)
add_MCMV_mis_mar_01_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_01)
add_MCMV_mis_mar_02_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_02)
add_MCMV_mis_mar_03_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_03)
add_MCMV_mis_mar_04_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_04)
add_MCMV_mis_mar_05_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_05)
add_MCMV_mis_mar_06_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_06)
add_MCMV_mis_mar_07_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_07)
add_MCMV_mis_mar_08_imp_GB =imp_GB.fit_transform(add_MCMV_mis_mar_08)
                                
add_MCMV_mis_mar_005_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_005)
add_MCMV_mis_mar_01_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_01)
add_MCMV_mis_mar_02_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_02)
add_MCMV_mis_mar_03_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_03)
add_MCMV_mis_mar_04_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_04)
add_MCMV_mis_mar_05_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_05)
add_MCMV_mis_mar_06_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_06)
add_MCMV_mis_mar_07_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_07)
add_MCMV_mis_mar_08_imp_RF =imp_RF.fit_transform(add_MCMV_mis_mar_08)


dataset_bl_mis_mar_LR=[bl_mis_mar_005_imp_LR,
                      bl_mis_mar_01_imp_LR,
                      bl_mis_mar_02_imp_LR,
                      bl_mis_mar_03_imp_LR,
                      bl_mis_mar_04_imp_LR,
                      bl_mis_mar_05_imp_LR,
                      bl_mis_mar_06_imp_LR,
                      bl_mis_mar_07_imp_LR,
                      bl_mis_mar_08_imp_LR]

dataset_add_mis_mar_LR=[add_mis_mar_005_imp_LR,
                      add_mis_mar_01_imp_LR,
                      add_mis_mar_02_imp_LR,
                      add_mis_mar_03_imp_LR,
                      add_mis_mar_04_imp_LR,
                      add_mis_mar_05_imp_LR,
                      add_mis_mar_06_imp_LR,
                      add_mis_mar_07_imp_LR,
                      add_mis_mar_08_imp_LR]

dataset_add_onehot_mis_mar_LR=[add_onehot_mis_mar_005_imp_LR,
                              add_onehot_mis_mar_01_imp_LR,
                              add_onehot_mis_mar_02_imp_LR,
                              add_onehot_mis_mar_03_imp_LR,
                              add_onehot_mis_mar_04_imp_LR,
                              add_onehot_mis_mar_05_imp_LR,
                              add_onehot_mis_mar_06_imp_LR,
                              add_onehot_mis_mar_07_imp_LR,
                              add_onehot_mis_mar_08_imp_LR]


dataset_add_MCMV_mis_mar_LR=[add_MCMV_mis_mar_005_imp_LR,
                              add_MCMV_mis_mar_01_imp_LR,
                              add_MCMV_mis_mar_02_imp_LR,
                              add_MCMV_mis_mar_03_imp_LR,
                              add_MCMV_mis_mar_04_imp_LR,
                              add_MCMV_mis_mar_05_imp_LR,
                              add_MCMV_mis_mar_06_imp_LR,
                              add_MCMV_mis_mar_07_imp_LR,
                              add_MCMV_mis_mar_08_imp_LR]
dataset_bl_mis_mar_GB=[bl_mis_mar_005_imp_GB,
                        bl_mis_mar_01_imp_GB,
                        bl_mis_mar_02_imp_GB,
                        bl_mis_mar_03_imp_GB,
                        bl_mis_mar_04_imp_GB,
                        bl_mis_mar_05_imp_GB,
                        bl_mis_mar_06_imp_GB,
                        bl_mis_mar_07_imp_GB,
                        bl_mis_mar_08_imp_GB]

dataset_add_mis_mar_GB=[add_mis_mar_005_imp_GB,
                         add_mis_mar_01_imp_GB,
                         add_mis_mar_02_imp_GB,
                         add_mis_mar_03_imp_GB,
                         add_mis_mar_04_imp_GB,
                         add_mis_mar_05_imp_GB,
                         add_mis_mar_06_imp_GB,
                         add_mis_mar_07_imp_GB,
                         add_mis_mar_08_imp_GB]

dataset_add_onehot_mis_mar_GB=[add_onehot_mis_mar_005_imp_GB,
                                add_onehot_mis_mar_01_imp_GB,
                                add_onehot_mis_mar_02_imp_GB,
                                add_onehot_mis_mar_03_imp_GB,
                                add_onehot_mis_mar_04_imp_GB,
                                add_onehot_mis_mar_05_imp_GB,
                                add_onehot_mis_mar_06_imp_GB,
                                add_onehot_mis_mar_07_imp_GB,
                                add_onehot_mis_mar_08_imp_GB]

dataset_add_MCMV_mis_mar_GB=[add_MCMV_mis_mar_005_imp_GB,
                              add_MCMV_mis_mar_01_imp_GB,
                              add_MCMV_mis_mar_02_imp_GB,
                              add_MCMV_mis_mar_03_imp_GB,
                              add_MCMV_mis_mar_04_imp_GB,
                              add_MCMV_mis_mar_05_imp_GB,
                              add_MCMV_mis_mar_06_imp_GB,
                              add_MCMV_mis_mar_07_imp_GB,
                              add_MCMV_mis_mar_08_imp_GB]
dataset_bl_mis_mar_RF=[bl_mis_mar_005_imp_RF,
                        bl_mis_mar_01_imp_RF,
                        bl_mis_mar_02_imp_RF,
                        bl_mis_mar_03_imp_RF,
                        bl_mis_mar_04_imp_RF,
                        bl_mis_mar_05_imp_RF,
                        bl_mis_mar_06_imp_RF,
                        bl_mis_mar_07_imp_RF,
                        bl_mis_mar_08_imp_RF]

dataset_add_mis_mar_RF=[add_mis_mar_005_imp_RF,
                         add_mis_mar_01_imp_RF,
                         add_mis_mar_02_imp_RF,
                         add_mis_mar_03_imp_RF,
                         add_mis_mar_04_imp_RF,
                         add_mis_mar_05_imp_RF,
                         add_mis_mar_06_imp_RF,
                         add_mis_mar_07_imp_RF,
                         add_mis_mar_08_imp_RF]

dataset_add_onehot_mis_mar_RF=[add_onehot_mis_mar_005_imp_RF,
                                add_onehot_mis_mar_01_imp_RF,
                                add_onehot_mis_mar_02_imp_RF,
                                add_onehot_mis_mar_03_imp_RF,
                                add_onehot_mis_mar_04_imp_RF,
                                add_onehot_mis_mar_05_imp_RF,
                                add_onehot_mis_mar_06_imp_RF,
                                add_onehot_mis_mar_07_imp_RF,
                                add_onehot_mis_mar_08_imp_RF]


dataset_add_MCMV_mis_mar_RF=[add_MCMV_mis_mar_005_imp_RF,
                              add_MCMV_mis_mar_01_imp_RF,
                              add_MCMV_mis_mar_02_imp_RF,
                              add_MCMV_mis_mar_03_imp_RF,
                              add_MCMV_mis_mar_04_imp_RF,
                              add_MCMV_mis_mar_05_imp_RF,
                              add_MCMV_mis_mar_06_imp_RF,
                              add_MCMV_mis_mar_07_imp_RF,
                              add_MCMV_mis_mar_08_imp_RF]

dataset_bl_mis_mcar_LR=[bl_mis_mcar_005_imp_LR,
                        bl_mis_mcar_01_imp_LR,
                        bl_mis_mcar_02_imp_LR,
                        bl_mis_mcar_03_imp_LR,
                        bl_mis_mcar_04_imp_LR,
                        bl_mis_mcar_05_imp_LR,
                        bl_mis_mcar_06_imp_LR,
                        bl_mis_mcar_07_imp_LR,
                        bl_mis_mcar_08_imp_LR]

dataset_add_mis_mcar_LR=[add_mis_mcar_005_imp_LR,
                        add_mis_mcar_01_imp_LR,
                        add_mis_mcar_02_imp_LR,
                        add_mis_mcar_03_imp_LR,
                        add_mis_mcar_04_imp_LR,
                        add_mis_mcar_05_imp_LR,
                        add_mis_mcar_06_imp_LR,
                        add_mis_mcar_07_imp_LR,
                        add_mis_mcar_08_imp_LR]

dataset_add_onehot_mis_mcar_LR=[add_onehot_mis_mcar_005_imp_LR,
                                add_onehot_mis_mcar_01_imp_LR,
                                add_onehot_mis_mcar_02_imp_LR,
                                add_onehot_mis_mcar_03_imp_LR,
                                add_onehot_mis_mcar_04_imp_LR,
                                add_onehot_mis_mcar_05_imp_LR,
                                add_onehot_mis_mcar_06_imp_LR,
                                add_onehot_mis_mcar_07_imp_LR,
                                add_onehot_mis_mcar_08_imp_LR]


dataset_add_MCMV_mis_mcar_LR=[add_MCMV_mis_mcar_005_imp_LR,
                             add_MCMV_mis_mcar_01_imp_LR,
                             add_MCMV_mis_mcar_02_imp_LR,
                             add_MCMV_mis_mcar_03_imp_LR,
                             add_MCMV_mis_mcar_04_imp_LR,
                             add_MCMV_mis_mcar_05_imp_LR,
                             add_MCMV_mis_mcar_06_imp_LR,
                             add_MCMV_mis_mcar_07_imp_LR,
                             add_MCMV_mis_mcar_08_imp_LR]
dataset_bl_mis_mcar_GB=[bl_mis_mcar_005_imp_GB,
                        bl_mis_mcar_01_imp_GB,
                        bl_mis_mcar_02_imp_GB,
                        bl_mis_mcar_03_imp_GB,
                        bl_mis_mcar_04_imp_GB,
                        bl_mis_mcar_05_imp_GB,
                        bl_mis_mcar_06_imp_GB,
                        bl_mis_mcar_07_imp_GB,
                        bl_mis_mcar_08_imp_GB]

dataset_add_mis_mcar_GB=[add_mis_mcar_005_imp_GB,
                        add_mis_mcar_01_imp_GB,
                        add_mis_mcar_02_imp_GB,
                        add_mis_mcar_03_imp_GB,
                        add_mis_mcar_04_imp_GB,
                        add_mis_mcar_05_imp_GB,
                        add_mis_mcar_06_imp_GB,
                        add_mis_mcar_07_imp_GB,
                        add_mis_mcar_08_imp_GB]

dataset_add_onehot_mis_mcar_GB=[add_onehot_mis_mcar_005_imp_GB,
                                add_onehot_mis_mcar_01_imp_GB,
                                add_onehot_mis_mcar_02_imp_GB,
                                add_onehot_mis_mcar_03_imp_GB,
                                add_onehot_mis_mcar_04_imp_GB,
                                add_onehot_mis_mcar_05_imp_GB,
                                add_onehot_mis_mcar_06_imp_GB,
                                add_onehot_mis_mcar_07_imp_GB,
                                add_onehot_mis_mcar_08_imp_GB]


dataset_add_MCMV_mis_mcar_GB=[add_MCMV_mis_mcar_005_imp_GB,
                             add_MCMV_mis_mcar_01_imp_GB,
                             add_MCMV_mis_mcar_02_imp_GB,
                             add_MCMV_mis_mcar_03_imp_GB,
                             add_MCMV_mis_mcar_04_imp_GB,
                             add_MCMV_mis_mcar_05_imp_GB,
                             add_MCMV_mis_mcar_06_imp_GB,
                             add_MCMV_mis_mcar_07_imp_GB,
                             add_MCMV_mis_mcar_08_imp_GB]
dataset_bl_mis_mcar_RF=[bl_mis_mcar_005_imp_RF,
                        bl_mis_mcar_01_imp_RF,
                        bl_mis_mcar_02_imp_RF,
                        bl_mis_mcar_03_imp_RF,
                        bl_mis_mcar_04_imp_RF,
                        bl_mis_mcar_05_imp_RF,
                        bl_mis_mcar_06_imp_RF,
                        bl_mis_mcar_07_imp_RF,
                        bl_mis_mcar_08_imp_RF]

dataset_add_mis_mcar_RF=[add_mis_mcar_005_imp_RF,
                        add_mis_mcar_01_imp_RF,
                        add_mis_mcar_02_imp_RF,
                        add_mis_mcar_03_imp_RF,
                        add_mis_mcar_04_imp_RF,
                        add_mis_mcar_05_imp_RF,
                        add_mis_mcar_06_imp_RF,
                        add_mis_mcar_07_imp_RF,
                        add_mis_mcar_08_imp_RF]


dataset_add_onehot_mis_mcar_RF=[add_onehot_mis_mcar_005_imp_RF,
                                add_onehot_mis_mcar_01_imp_RF,
                                add_onehot_mis_mcar_02_imp_RF,
                                add_onehot_mis_mcar_03_imp_RF,
                                add_onehot_mis_mcar_04_imp_RF,
                                add_onehot_mis_mcar_05_imp_RF,
                                add_onehot_mis_mcar_06_imp_RF,
                                add_onehot_mis_mcar_07_imp_RF,
                                add_onehot_mis_mcar_08_imp_RF]


dataset_add_MCMV_mis_mcar_RF=[add_MCMV_mis_mcar_005_imp_RF,
                             add_MCMV_mis_mcar_01_imp_RF,
                             add_MCMV_mis_mcar_02_imp_RF,
                             add_MCMV_mis_mcar_03_imp_RF,
                             add_MCMV_mis_mcar_04_imp_RF,
                             add_MCMV_mis_mcar_05_imp_RF,
                             add_MCMV_mis_mcar_06_imp_RF,
                             add_MCMV_mis_mcar_07_imp_RF,
                             add_MCMV_mis_mcar_08_imp_RF]
dataset_bl_mis_mnar_LR=[bl_mis_mnar_005_imp_LR,
                        bl_mis_mnar_01_imp_LR,
                        bl_mis_mnar_02_imp_LR,
                        bl_mis_mnar_03_imp_LR,
                        bl_mis_mnar_04_imp_LR,
                        bl_mis_mnar_05_imp_LR,
                        bl_mis_mnar_06_imp_LR,
                        bl_mis_mnar_07_imp_LR,
                        bl_mis_mnar_08_imp_LR]
dataset_add_mis_mnar_LR=[add_mis_mnar_005_imp_LR,
                        add_mis_mnar_01_imp_LR,
                        add_mis_mnar_02_imp_LR,
                        add_mis_mnar_03_imp_LR,
                        add_mis_mnar_04_imp_LR,
                        add_mis_mnar_05_imp_LR,
                        add_mis_mnar_06_imp_LR,
                        add_mis_mnar_07_imp_LR,
                        add_mis_mnar_08_imp_LR]

dataset_add_onehot_mis_mnar_LR=[add_onehot_mis_mnar_005_imp_LR,
                                add_onehot_mis_mnar_01_imp_LR,
                                add_onehot_mis_mnar_02_imp_LR,
                                add_onehot_mis_mnar_03_imp_LR,
                                add_onehot_mis_mnar_04_imp_LR,
                                add_onehot_mis_mnar_05_imp_LR,
                                add_onehot_mis_mnar_06_imp_LR,
                                add_onehot_mis_mnar_07_imp_LR,
                                add_onehot_mis_mnar_08_imp_LR]


dataset_add_MCMV_mis_mnar_LR=[add_MCMV_mis_mnar_005_imp_LR,
                             add_MCMV_mis_mnar_01_imp_LR,
                             add_MCMV_mis_mnar_02_imp_LR,
                             add_MCMV_mis_mnar_03_imp_LR,
                             add_MCMV_mis_mnar_04_imp_LR,
                             add_MCMV_mis_mnar_05_imp_LR,
                             add_MCMV_mis_mnar_06_imp_LR,
                             add_MCMV_mis_mnar_07_imp_LR,
                             add_MCMV_mis_mnar_08_imp_LR]
dataset_bl_mis_mnar_GB=[bl_mis_mnar_005_imp_GB,
                        bl_mis_mnar_01_imp_GB,
                        bl_mis_mnar_02_imp_GB,
                        bl_mis_mnar_03_imp_GB,
                        bl_mis_mnar_04_imp_GB,
                        bl_mis_mnar_05_imp_GB,
                        bl_mis_mnar_06_imp_GB,
                        bl_mis_mnar_07_imp_GB,
                        bl_mis_mnar_08_imp_GB]

dataset_add_mis_mnar_GB=[add_mis_mnar_005_imp_GB,
                        add_mis_mnar_01_imp_GB,
                        add_mis_mnar_02_imp_GB,
                        add_mis_mnar_03_imp_GB,
                        add_mis_mnar_04_imp_GB,
                        add_mis_mnar_05_imp_GB,
                        add_mis_mnar_06_imp_GB,
                        add_mis_mnar_07_imp_GB,
                        add_mis_mnar_08_imp_GB]

dataset_add_onehot_mis_mnar_GB=[add_onehot_mis_mnar_005_imp_GB,
                                add_onehot_mis_mnar_01_imp_GB,
                                add_onehot_mis_mnar_02_imp_GB,
                                add_onehot_mis_mnar_03_imp_GB,
                                add_onehot_mis_mnar_04_imp_GB,
                                add_onehot_mis_mnar_05_imp_GB,
                                add_onehot_mis_mnar_06_imp_GB,
                                add_onehot_mis_mnar_07_imp_GB,
                                add_onehot_mis_mnar_08_imp_GB]


dataset_add_MCMV_mis_mnar_GB=[add_MCMV_mis_mnar_005_imp_GB,
                             add_MCMV_mis_mnar_01_imp_GB,
                             add_MCMV_mis_mnar_02_imp_GB,
                             add_MCMV_mis_mnar_03_imp_GB,
                             add_MCMV_mis_mnar_04_imp_GB,
                             add_MCMV_mis_mnar_05_imp_GB,
                             add_MCMV_mis_mnar_06_imp_GB,
                             add_MCMV_mis_mnar_07_imp_GB,
                             add_MCMV_mis_mnar_08_imp_GB]
dataset_bl_mis_mnar_RF=[bl_mis_mnar_005_imp_RF,
                        bl_mis_mnar_01_imp_RF,
                        bl_mis_mnar_02_imp_RF,
                        bl_mis_mnar_03_imp_RF,
                        bl_mis_mnar_04_imp_RF,
                        bl_mis_mnar_05_imp_RF,
                        bl_mis_mnar_06_imp_RF,
                        bl_mis_mnar_07_imp_RF,
                        bl_mis_mnar_08_imp_RF]

dataset_add_mis_mnar_RF=[add_mis_mnar_005_imp_RF,
                        add_mis_mnar_01_imp_RF,
                        add_mis_mnar_02_imp_RF,
                        add_mis_mnar_03_imp_RF,
                        add_mis_mnar_04_imp_RF,
                        add_mis_mnar_05_imp_RF,
                        add_mis_mnar_06_imp_RF,
                        add_mis_mnar_07_imp_RF,
                        add_mis_mnar_08_imp_RF]

dataset_add_onehot_mis_mnar_RF=[add_onehot_mis_mnar_005_imp_RF,
                                add_onehot_mis_mnar_01_imp_RF,
                                add_onehot_mis_mnar_02_imp_RF,
                                add_onehot_mis_mnar_03_imp_RF,
                                add_onehot_mis_mnar_04_imp_RF,
                                add_onehot_mis_mnar_05_imp_RF,
                                add_onehot_mis_mnar_06_imp_RF,
                                add_onehot_mis_mnar_07_imp_RF,
                                add_onehot_mis_mnar_08_imp_RF]


dataset_add_MCMV_mis_mnar_RF=[add_MCMV_mis_mnar_005_imp_RF,
                             add_MCMV_mis_mnar_01_imp_RF,
                             add_MCMV_mis_mnar_02_imp_RF,
                             add_MCMV_mis_mnar_03_imp_RF,
                             add_MCMV_mis_mnar_04_imp_RF,
                             add_MCMV_mis_mnar_05_imp_RF,
                             add_MCMV_mis_mnar_06_imp_RF,
                             add_MCMV_mis_mnar_07_imp_RF,
                             add_MCMV_mis_mnar_08_imp_RF]



dataset_mean_imp_mar=[mean_imp_mis_mar_005,
                        mean_imp_mis_mar_01,
                        mean_imp_mis_mar_02,
                        mean_imp_mis_mar_03,
                        mean_imp_mis_mar_04,
                        mean_imp_mis_mar_05,
                        mean_imp_mis_mar_06,
                        mean_imp_mis_mar_07,
                        mean_imp_mis_mar_08]

dataset_mean_imp_mcar=[mean_imp_mis_mcar_005,
                        mean_imp_mis_mcar_01,
                        mean_imp_mis_mcar_02,
                        mean_imp_mis_mcar_03,
                        mean_imp_mis_mcar_04,
                        mean_imp_mis_mcar_05,
                        mean_imp_mis_mcar_06,
                        mean_imp_mis_mcar_07,
                        mean_imp_mis_mcar_08]


dataset_mean_imp_mnar=[mean_imp_mis_mnar_005,
                        mean_imp_mis_mnar_01,
                        mean_imp_mis_mnar_02,
                        mean_imp_mis_mnar_03,
                        mean_imp_mis_mnar_04,
                        mean_imp_mis_mnar_05,
                        mean_imp_mis_mnar_06,
                        mean_imp_mis_mnar_07,
                        mean_imp_mis_mnar_08]

dataset_median_imp_mar=[median_imp_mis_mar_005,
                        median_imp_mis_mar_01,
                        median_imp_mis_mar_02,
                        median_imp_mis_mar_03,
                        median_imp_mis_mar_04,
                        median_imp_mis_mar_05,
                        median_imp_mis_mar_06,
                        median_imp_mis_mar_07,
                        median_imp_mis_mar_08]

dataset_median_imp_mcar=[median_imp_mis_mcar_005,
                        median_imp_mis_mcar_01,
                        median_imp_mis_mcar_02,
                        median_imp_mis_mcar_03,
                        median_imp_mis_mcar_04,
                        median_imp_mis_mcar_05,
                        median_imp_mis_mcar_06,
                        median_imp_mis_mcar_07,
                        median_imp_mis_mcar_08]


dataset_median_imp_mnar=[median_imp_mis_mnar_005,
                        median_imp_mis_mnar_01,
                        median_imp_mis_mnar_02,
                        median_imp_mis_mnar_03,
                        median_imp_mis_mnar_04,
                        median_imp_mis_mnar_05,
                        median_imp_mis_mnar_06,
                        median_imp_mis_mnar_07,
                        median_imp_mis_mnar_08]


def rmse(datasets):
    i=0
    list= np.zeros(9)
    for dataset in (datasets): 
        rmse_stats=mean_squared_error(train_data, dataset[:,0:9], squared=False)
        list[i]=rmse_stats
        i=i+1
    path  = "RMSE_FCM/train_data_1_RMSE.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list
        csv_write.writerow(data_row)
    print("rmse_stats:",pd.DataFrame(list.tolist()))


print("RMSE_MAR:\n")
rmse(dataset_mean_imp_mar)
rmse(dataset_median_imp_mar)

print("mar_LR:\n")
rmse(dataset_bl_mis_mar_LR)
rmse(dataset_add_mis_mar_LR)
rmse(dataset_add_onehot_mis_mar_LR)
rmse(dataset_add_MCMV_mis_mar_LR)
print("\n")
print("mar_GB:")
rmse(dataset_bl_mis_mar_GB)
rmse(dataset_add_mis_mar_GB)
rmse(dataset_add_onehot_mis_mar_GB)
rmse(dataset_add_MCMV_mis_mar_GB)
print("\n")
print("mar_RF:\n")
rmse(dataset_bl_mis_mar_RF)
rmse(dataset_add_mis_mar_RF)
rmse(dataset_add_onehot_mis_mar_RF)
rmse(dataset_add_MCMV_mis_mar_RF)


print("RMSE_MCAR:\n")

rmse(dataset_mean_imp_mcar)
rmse(dataset_median_imp_mcar)
print("mcar_LR:\n")
rmse(dataset_bl_mis_mcar_LR)
rmse(dataset_add_mis_mcar_LR)
rmse(dataset_add_onehot_mis_mcar_LR)
rmse(dataset_add_MCMV_mis_mcar_LR)
print("\n")
print("mcar_GB:\n")
rmse(dataset_bl_mis_mcar_GB)
rmse(dataset_add_mis_mcar_GB)
rmse(dataset_add_onehot_mis_mcar_GB)
rmse(dataset_add_MCMV_mis_mcar_GB)
print("\n")
print("mcar_RF:\n")
rmse(dataset_bl_mis_mcar_RF)
rmse(dataset_add_mis_mcar_RF)
rmse(dataset_add_onehot_mis_mcar_RF)
rmse(dataset_add_MCMV_mis_mcar_RF)


print("RMSE_MNAR:\n")

rmse(dataset_mean_imp_mnar)
rmse(dataset_median_imp_mnar)
print("mnar_LR:\n")
rmse(dataset_bl_mis_mnar_LR)
rmse(dataset_add_mis_mnar_LR)
rmse(dataset_add_onehot_mis_mnar_LR)
rmse(dataset_add_MCMV_mis_mnar_LR)
print("\n")
print("mnar_GB:\n")
rmse(dataset_bl_mis_mnar_GB)
rmse(dataset_add_mis_mnar_GB)
rmse(dataset_add_onehot_mis_mnar_GB)
rmse(dataset_add_MCMV_mis_mnar_GB)
print("\n")
print("mnar_RF:\n")
rmse(dataset_bl_mis_mnar_RF)
rmse(dataset_add_mis_mnar_RF)
rmse(dataset_add_onehot_mis_mnar_RF)
rmse(dataset_add_MCMV_mis_mnar_RF)


from sklearn.ensemble import RandomForestClassifier as RF
def classify_RF_no_miss(datasets):
    i=0
    list= np.zeros(9)
    list_AUC=np.zeros(9)
    for dataset in tqdm(datasets): 
        dataset=pd.DataFrame(dataset)
#         print(dataset.info())
        X=dataset.iloc[:,0:8]
        y=dataset.iloc[:,8] 
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7,stratify = y)
        cv_outer = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)

        ACC_test=[]
        AUC_test=[]
        for train_idx, test_idx in tqdm(cv_outer.split(X, y)):
            train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
            train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]

            model = RF(n_jobs=-1)
            cv_inner = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)
            p_grid = {'n_estimators':range(50,151,10),
                      'max_depth':range(1,16,1), }
#                       'min_samples_split':range(10,101,10),
#                      'min_samples_leaf':range(5,51,5)}
            gd_search = GridSearchCV(model,p_grid, n_jobs=-1, cv=cv_inner,scoring = 'accuracy').fit(train_data, train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model
            print("Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
            scores =classifier.score(test_data,test_target)
            y_pred_prob = classifier.predict_proba(test_data)
            AUC= metrics.roc_auc_score(test_target, y_pred_prob,multi_class='ovo')
            ACC_test.append(scores)
            AUC_test.append(AUC)
        print("ACC_test:",ACC_test)
        print("AUC_test:",AUC_test)
        ACC=sum(ACC_test)/len(ACC_test)
        AUC=sum(AUC_test)/len(AUC_test)
        print("ACC：",ACC)
        print("AUC：",AUC)
        list[i]=ACC
        list_AUC[i]=AUC
        i=i+1
    path  = "ACC/train1_no_miss_ACC_RF.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list
        csv_write.writerow(data_row)
    path1  = "ACU/train1_no_miss_AUC_RF.csv"
    with open(path1,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list_AUC
        csv_write.writerow(data_row)
    print("AUC:", pd.DataFrame(list_AUC.tolist()))
    print("scores_mean:",pd.DataFrame(list.tolist()))








def classify_RF_no_miss(datasets):
    i=0
    list= np.zeros(9)
    list_AUC=np.zeros(9)
    for dataset in tqdm(datasets): 
        dataset=pd.DataFrame(dataset)
#         print(dataset.info())
        X=dataset.iloc[:,0:8]
        y=dataset.iloc[:,8] 
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7,stratify = y)
        cv_outer = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)

        ACC_test=[]
        AUC_test=[]
#         for train_idx, test_idx in tqdm(cv_outer.split(X, y)):
#             train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
#             train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]

        model = RF(n_jobs=-1)
        cv_inner = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)
        p_grid = {'n_estimators':range(50,151,10),
                  'max_depth':range(1,16,1), }
#                       'min_samples_split':range(10,101,10),
#                      'min_samples_leaf':range(5,51,5)}
        gd_search = GridSearchCV(model,p_grid, n_jobs=-1, cv=cv_inner,scoring = 'accuracy').fit(X, y)
        best_model = gd_search.best_estimator_
        classifier = best_model
        print("Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
        scores =classifier.score(test_X,test_y)
        y_pred_prob = classifier.predict_proba(test_X)
        AUC= metrics.roc_auc_score(test_y, y_pred_prob,multi_class='ovo')
        ACC_test.append(scores)
        AUC_test.append(AUC)
#         print("ACC_test:",ACC_test)
#         print("AUC_test:",AUC_test)
        ACC=sum(ACC_test)/len(ACC_test)
        AUC=sum(AUC_test)/len(AUC_test)
        print("ACC：",ACC)
        print("AUC：",AUC)
        list[i]=ACC
        list_AUC[i]=AUC
        i=i+1
    path  = "ACC_FCM/train1_mean_median_imp_ACC_RF.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list
        csv_write.writerow(data_row)
    path1  = "AUC_FCM/train1_mean_median_imp_AUC_RF.csv"
    with open(path1,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list_AUC
        csv_write.writerow(data_row)
    print("AUC:", pd.DataFrame(list_AUC.tolist()))
    print("scores_mean:",pd.DataFrame(list.tolist()))


test_data = pd.read_csv('test data/test_set1.csv',header = None)
test_data=test_data[1:][:]
test_X=test_data.iloc[:,0:8]
test_y=test_data.iloc[:,8]


classify_RF_no_miss(dataset_mean_imp_mar)
classify_RF_no_miss(dataset_mean_imp_mcar)
classify_RF_no_miss(dataset_mean_imp_mnar)
classify_RF_no_miss(dataset_median_imp_mar)
classify_RF_no_miss(dataset_median_imp_mcar)
classify_RF_no_miss(dataset_median_imp_mnar)


def SMOTE_classify_RF(datasets):
    i=0
    list= np.zeros(9)
    list_AUC=np.zeros(9)
    for dataset in tqdm(datasets): 
        dataset=pd.DataFrame(dataset)
#         print(dataset.info())
        X=dataset.iloc[:,0:8]
        y=dataset.iloc[:,8] 
        nos = ADASYN(random_state=i)  # 综合采样
        X, y = nos.fit_resample(X, y)
        #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7,stratify = y)

        cv_outer = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)

        ACC_test=[]
        AUC_test=[]
#         for train_idx, test_idx in tqdm(cv_outer.split(X, y)):
#             train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
#             train_target, test_target = y.iloc[train_idx], y.iloc[test_idx]

        model = RF(n_jobs=-1)
        cv_inner = StratifiedKFold(n_splits=5, random_state=i,shuffle=True)
        p_grid = {'n_estimators':range(50,201,10),
                  'max_depth':range(10,26,1), }
#                       'min_samples_split':range(10,101,10),
#                      'min_samples_leaf':range(5,51,5)}
        gd_search = GridSearchCV(model,p_grid, n_jobs= -1, cv=cv_inner,scoring = 'accuracy').fit(X, y)
        best_model = gd_search.best_estimator_
        classifier = best_model
        print("Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
        scores =classifier.score(test_X,test_y)
        y_pred_prob = classifier.predict_proba(test_X)
        AUC= metrics.roc_auc_score(test_y, y_pred_prob,multi_class='ovo')
        ACC_test.append(scores)
        AUC_test.append(AUC)
        print("ACC_test:",ACC_test)
        print("AUC_test:",AUC_test)
        ACC=sum(ACC_test)/len(ACC_test)
        AUC=sum(AUC_test)/len(AUC_test)
        print("ACC：",ACC)
        print("AUC：",AUC)
        list[i]=ACC
        list_AUC[i]=AUC
        i=i+1
    path  = "ACC_FCM/train1_ADASYN_ACC_RF_test.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list
        csv_write.writerow(data_row)
    path1  = "AUC_FCM/train1_ADASYN_AUC_RF_test.csv"
    with open(path1,'a+') as f:
        csv_write = csv.writer(f)
        data_row = list_AUC
        csv_write.writerow(data_row)
    print("AUC:", pd.DataFrame(list_AUC.tolist()))
    print("scores_mean:",pd.DataFrame(list.tolist()))



print("dataset_bl_mis_mar_LR")
SMOTE_classify_RF(dataset_bl_mis_mar_LR)

print("\n")
print("dataset_add_mis_mar_LR")
SMOTE_classify_RF(dataset_add_mis_mar_LR)

print("\n")
print("dataset_add_onehot_mis_mar_LR")
SMOTE_classify_RF(dataset_add_onehot_mis_mar_LR)

print("\n")
print("dataset_add_MCMV_mis_mar_LR")
SMOTE_classify_RF(dataset_add_MCMV_mis_mar_LR)



print("dataset_bl_mis_mar_GB")
SMOTE_classify_RF(dataset_bl_mis_mar_GB)

print("dataset_add_mis_mar_GB")
SMOTE_classify_RF(dataset_add_mis_mar_GB)

print("dataset_add_onehot_mis_mar_GB")
SMOTE_classify_RF(dataset_add_onehot_mis_mar_GB)

print("dataset_add_MCMV_mis_mar_GB")
SMOTE_classify_RF(dataset_add_MCMV_mis_mar_GB)



print("dataset_bl_mis_mar_RF")
SMOTE_classify_RF(dataset_bl_mis_mar_RF)

print("dataset_add_mis_mar_RF")
SMOTE_classify_RF(dataset_add_mis_mar_RF)

print("dataset_add_onehot_mis_mar_RF")
SMOTE_classify_RF(dataset_add_onehot_mis_mar_RF)

print("dataset_add_MCMV_mis_mar_RF")
SMOTE_classify_RF(dataset_add_MCMV_mis_mar_RF)



print("dataset_bl_mis_mcar_LR")
SMOTE_classify_RF(dataset_bl_mis_mcar_LR)

print("dataset_add_mis_mcar_LR")
SMOTE_classify_RF(dataset_add_mis_mcar_LR)

print("dataset_add_onehot_mis_mcar_LR")
SMOTE_classify_RF(dataset_add_onehot_mis_mcar_LR)

print("dataset_add_MCMV_mis_mcar_LR")
SMOTE_classify_RF(dataset_add_MCMV_mis_mcar_LR)



print("dataset_bl_mis_mcar_GB")
SMOTE_classify_RF(dataset_bl_mis_mcar_GB)

print("dataset_add_mis_mcar_GB")
SMOTE_classify_RF(dataset_add_mis_mcar_GB)

print("dataset_add_onehot_mis_mcar_GB")
SMOTE_classify_RF(dataset_add_onehot_mis_mcar_GB)

print("dataset_add_MCMV_mis_mcar_GB")
SMOTE_classify_RF(dataset_add_MCMV_mis_mcar_GB)



print("dataset_bl_mis_mcar_RF")
SMOTE_classify_RF(dataset_bl_mis_mcar_RF)

print("dataset_add_mis_mcar_RF")
SMOTE_classify_RF(dataset_add_mis_mcar_RF)

print("dataset_add_onehot_mis_mcar_RF")
SMOTE_classify_RF(dataset_add_onehot_mis_mcar_RF)

print("dataset_add_MCMV_mis_mcar_RF")
SMOTE_classify_RF(dataset_add_MCMV_mis_mcar_RF)



print("dataset_bl_mis_mnar_LR")
SMOTE_classify_RF(dataset_bl_mis_mnar_LR)

print("dataset_add_mis_mnar_LR")
SMOTE_classify_RF(dataset_add_mis_mnar_LR)

print("dataset_add_onehot_mis_mnar_LR")
SMOTE_classify_RF(dataset_add_onehot_mis_mnar_LR)

print("dataset_add_MCMV_mis_mnar_LR")
SMOTE_classify_RF(dataset_add_MCMV_mis_mnar_LR)



print("dataset_bl_mis_mnar_GB")
SMOTE_classify_RF(dataset_bl_mis_mnar_GB)

print("dataset_add_mis_mnar_GB")
SMOTE_classify_RF(dataset_add_mis_mnar_GB)

print("dataset_add_onehot_mis_mnar_GB")
SMOTE_classify_RF(dataset_add_onehot_mis_mnar_GB)

print("dataset_add_MCMV_mis_mnar_GB")
SMOTE_classify_RF(dataset_add_MCMV_mis_mnar_GB)



print("dataset_bl_mis_mnar_RF")
SMOTE_classify_RF(dataset_bl_mis_mnar_RF)

print("dataset_add_mis_mnar_RF")
SMOTE_classify_RF(dataset_add_mis_mnar_RF)

print("dataset_add_onehot_mis_mnar_RF")
SMOTE_classify_RF(dataset_add_onehot_mis_mnar_RF)
 
print("dataset_add_MCMV_mis_mnar_RF")
SMOTE_classify_RF(dataset_add_MCMV_mis_mnar_RF)













