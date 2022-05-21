# Clustring practice

# 1 군집화 실습



### 군집화(Clustering) 환경 설정

import os 
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import collections
import mglearn
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings(action='ignore')

random.seed(2020)

## 1.1 데이터 수집

Obesity = pd.read_csv('C:/Users/Soonchan Kwon/Desktop/ObesityDataSet_raw_and_data_sinthetic.csv')

Obesity

#### 데이터 설명
<br>
Obesity 데이터는 14 ~ 61세 사이의 중남미 국가 사람들의 다양한 식습관 및 신체 조건을 활용한 비만 수준 측정 데이터이다. <br>
속성들은 다음을 의미한다.<br>
* FAVC : 고열량 식품의 잦은 섭취<br>
* FCVC : 채소 섭취 빈도<br>
* NCP : 주요 식사 횟수<br>
* CAEC : 식사 사이의 음식 섭취<br>
* CH2O : 매일 물 섭취<br>
* SCC : 칼로리 소비 모니터링<br>
* FAF : 신체 활동 빈도<br>
또한, NObeyesdad는 lable 값으로 Normal_Weight 부터 Obesity까지 존재한다.

Obesity.info()

## 데이터 전처리 <br>

범주형 변수 변환 

Obesity = Obesity.replace({'Male':0, 'Female':1}, inplace=False)
Obesity.family_history_with_overweight = Obesity.family_history_with_overweight.replace('no', 0)
Obesity.family_history_with_overweight = Obesity.family_history_with_overweight.replace('yes', 1)
Obesity.FAVC = Obesity.FAVC.replace('no', 0)
Obesity.FAVC = Obesity.FAVC.replace('yes', 1)
Obesity.SMOKE = Obesity.SMOKE.replace('no', 0)
Obesity.SMOKE = Obesity.SMOKE.replace('yes', 1)
Obesity.SCC = Obesity.SMOKE.replace('no', 1)
Obesity.SCC = Obesity.SMOKE.replace('yes', 0)
Obesity.CAEC = Obesity.CAEC.replace('no', 0)
Obesity.CAEC = Obesity.CAEC.replace('Sometimes', 1)
Obesity.CAEC = Obesity.CAEC.replace('Frequently', 2)
Obesity.CAEC = Obesity.CAEC.replace('Always', 3)
Obesity.CALC = Obesity.CALC.replace('no', 0)
Obesity.CALC = Obesity.CALC.replace('Sometimes', 1)
Obesity.CALC = Obesity.CALC.replace('Frequently', 2)
Obesity.CALC = Obesity.CALC.replace('Always', 3)
Obesity.MTRANS = Obesity.MTRANS.replace('Walking', 0)
Obesity.MTRANS = Obesity.MTRANS.replace('Bike', 1)
Obesity.MTRANS = Obesity.MTRANS.replace('Public_Transportation', 2)
Obesity.MTRANS = Obesity.MTRANS.replace('Motorbike', 3)
Obesity.MTRANS = Obesity.MTRANS.replace('Automobile', 4)

NObey|esdad를 제외한 범주형 변수들을 모두 적절한 수치형으로 변환시켜 주었다.<br>

print(Obesity.head(10))

Obesity.isna().sum()

결측치가 존재하는 지 확인 하였다. 

<br>

### 전처리 2

<br>

군집화 분석에 들어가기 앞서 특정 변수를 선정하는 작업을 거쳤다.<br>
비만의 경우 체중과 신장이 큰 영향을 지닌다는 사실을 인지하고 있기에 두 변수를 포함하였다.<br>
또한, heatmap과 pairplot을 통해 상관도를 확인 후 나머지 변수를 선정하였다.

g = sns.heatmap(Obesity.corr(), annot=True, linewidths=.16)
bottom, top = g.get_ylim() # heatmap plot이 잘리는 것 방지하기 위한 방법
g.set_ylim(bottom+0.5, top-0.5)
plt.show()

#import seaborn as sns
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
#sns.pairplot(Obesity)


pairplot 결과는 출력값이 너무 커 범주처리 해두었다.

Obesity_value=Obesity[['Weight', 'Height', 'CH2O']]

선정한 변수는 위와 같다.<br>
군집화를 위해 다음 변수들을 포함하여 새롭게 데이터 프레임을 정의하였다.<br>

Obesity_1=Obesity[['Weight', 'Height','NObeyesdad']]
Obesity_1.rename(columns={'Weight':'X'}, inplace = True)
Obesity_1.rename(columns={'Height':'Y'}, inplace = True)
Obesity_1

Obesity_2=Obesity[['Weight','CH2O','NObeyesdad']]
Obesity_2.rename(columns={'Weight':'X'}, inplace = True)
Obesity_2.rename(columns={'CH2O':'Y'}, inplace= True)
Obesity_2

Obesity_3=Obesity[['Height','CH2O','NObeyesdad']]
Obesity_3.rename(columns={'Height':'X'}, inplace = True)
Obesity_3.rename(columns={'CH2O':'Y'}, inplace = True)
Obesity_3

총 3가지 형태의 df를 새로 정의하였고 각각 다음과 같은 과정을 통해 정규화를 진행하였다.

def standardization(Data):
    Input = ((Data[['X', 'Y']] - np.mean(Data[['X', 'Y']], axis=0)) / np.std(Data[['X', 'Y']], axis=0))
    return(pd.concat([Input, Data['NObeyesdad']], axis=1))

Obesity_st1 = standardization(Obesity_1)
Obesity_st2 = standardization(Obesity_2)
Obesity_st3 = standardization(Obesity_3)

Artificial_Obesity={'Obesity_st1':Obesity_st1,  'Obesity_st2': Obesity_st2,'Obesity_st3': Obesity_st3}

세 df를 하나로 묶었다.<br>

Data = Artificial_Obesity[list(Artificial_Obesity.keys())[1]]
Data['X']

## 1-3 군집화 분석

### K-Means 알고리즘

def k_Means_Plot(Data, Select_k, NAME, Init_Method = 'k-means++', Num_Init=10):
    Data2 = Data[['X', 'Y']]    
    fig, axes = plt.subplots(1, (np.max(list(Select_k))-np.min(list(Select_k)))+1, figsize=(15, 4))
    for i in Select_k:
        Kmeans_Clustering = KMeans(n_clusters=i, init=Init_Method, random_state=2020, n_init=Num_Init)
        Kmeans_Clustering.fit(Data2)
        mglearn.discrete_scatter(Data2['X'], Data2['Y'], Kmeans_Clustering.labels_, ax=axes[i - 2], s=5)
        mglearn.discrete_scatter(Kmeans_Clustering.cluster_centers_[:, 0], 
                                 Kmeans_Clustering.cluster_centers_[:, 1],
                                 list(range(i)), 
                                 markeredgewidth=3, 
                                 ax=axes[i - 2], s=10)
        Score = np.round(silhouette_score(Data2, Kmeans_Clustering.labels_),3)
        axes[i - 2].set_title( NAME + ' / k = ' + str(i)+' / S_Score:'+str(Score))

먼저 K-Means 함수를 정의하였다.<br>

def Simple_Scatter(i, Name):
    Data = Artificial_Obesity[list(Artificial_Obesity.keys())[i]]
    G = sns.jointplot(x = 'X', y = 'Y', data = Data, kind='scatter', hue = "NObeyesdad")
    G.fig.suptitle("Data : " + Name, fontsize = 10, color = 'black', alpha = 0.9)

산점도를 가시화 하는 함수도 정의한 후 K-Means 분석을 실행 하였다. 

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    
    k_Means_Plot(Data = Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                 Select_k = range(2, 6),
                 NAME = list(Artificial_Obesity.keys())[i],
                 Init_Method='random',Num_Init=1)

Num_init 값을 1로 설정한 후 K-Means 분석을 진행하였다.<br>
Obesity_st1 같은 경우, k = 4 일 때,<br>
Obesity_st2 같은 경우, k = 5 일 때,<br>
Obesity_st3 같은 경우, k = 2 일 때,<br>
Silhouette score가 가장 높은 것을 확인 할 수 있다. 

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    
    k_Means_Plot(Data = Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                 Select_k = range(2, 6),
                 NAME = list(Artificial_Obesity.keys())[i],
                 Init_Method='random',Num_Init=10)

Num_init 값을 10으로 설정한 후 K-Means 분석을 진행할 경우엔<br>
Obesity_st1 같은 경우, k = 2 일 때,<br>
Obesity_st2 같은 경우, k = 5 일 때,<br>
Obesity_st3 같은 경우, k = 5 일 때,<br>
Silhouette score가 가장 높은 것을 확인 할 수 있다. 

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    
    k_Means_Plot(Data = Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                 Select_k = range(2, 6),
                 NAME = list(Artificial_Obesity.keys())[i],
                 Init_Method='k-means++',Num_Init=1)

마지막으로 K-Means++도 사용하여 군집 분석을 실행하였다.<br>

### Herarchical clustering


Simple_Scatter(2, list(Artificial_Obesity.keys())[2])

def Fixed_Dendrogram(Data, Num_of_p, Full_Use):
    Linkage_Matrix = linkage(Data,'complete')
    if(Full_Use == True):
        Num_of_p = np.shape(Data)[0]
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
    else:
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(
        Linkage_Matrix,
        truncate_mode = 'lastp',  
        p = Num_of_p, 
        leaf_rotation = 90.,
        leaf_font_size = 12.,
        color_threshold = 'default'
    )
    plt.show()

Fixed_Dendrogram(Obesity_st1[['X','Y']], -1, True) # -1: nomeaningful value

clustr의 개수가 5개 ~ 8개 일 때 거리가 눈에띄게 줄어드는 것처럼 보인다.

a = Fixed_Dendrogram(Obesity_st1[['X','Y']], 8, False)

cluster의 개수를 8개로 leaf node를 생성해 보았다.

def Hclust_Plot(Data,Select_k,NAME):
    Data2 = Data[['X', 'Y']]    
    fig, axes = plt.subplots(1, (np.max(list(Select_k))-np.min(list(Select_k)))+1, figsize=(15, 4))
    for i in Select_k:
        H_Clustering = AgglomerativeClustering(n_clusters=i+3,linkage="complete")
        P_Labels = H_Clustering.fit_predict(Data2)
        mglearn.discrete_scatter(Data2['X'], Data2['Y'], P_Labels, ax=axes[i - 2], s=5)        
        axes[i - 2].set_title("Data:" + NAME + ' / k = ' + str(i+3))
        Score=np.round(silhouette_score(Data2,P_Labels),3)
        axes[i - 2].set_title( NAME + ' / k = ' + str(i+3)+' / S_Score:'+str(Score))

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    Hclust_Plot(Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                range(2, 6), list(Artificial_Obesity.keys())[i])

cluster의 개수를 증가시킨 후 HC 분석을 진행하고 가시화를 하였다.<br>
Obesity_st1 같은 경우, k = 8 일 때,<br>
Obesity_st2 같은 경우, k = 6 일 때,<br>
Obesity_st3 같은 경우, k = 5 일 때,<br>
Silhouette score가 가장 높은 것을 확인 할 수 있다. 

### DBSCAN clustering

def DBSCAN_Plot(Data,NAME,min_samples=10,eps=0.5):
    Data2 = Data[['X', 'Y']]
    Append_k_Means_Results = list()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    Set_DBSCAN_Hyperparameter=DBSCAN(min_samples=min_samples,eps=eps)
    Results = Set_DBSCAN_Hyperparameter.fit_predict(Data2)
    Score=np.round(silhouette_score(Data2,Results),3)
    mglearn.discrete_scatter(Data2['X'], Data2['Y'], Data['NObeyesdad'], ax=axes[0], s=5)
    axes[0].set_title("Data:" + NAME + 'GroundTruth')    
    mglearn.discrete_scatter(Data2['X'], Data2['Y'], Results, ax=axes[1], s=5)
    axes[1].set_title("Data:" + NAME + ' DBSCAN/ eps:'+str(eps)+' / min_sample:'+str(min_samples)+'/ S_score:'+str(Score))


DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1")
DBSCAN_Plot(Data=Obesity_st2,NAME="Obesity_st2")
DBSCAN_Plot(Data=Obesity_st3,NAME="Obesity_st3")

최소 샘플수를 15, eps를 0.4로 했을 때 원하는 대로 군집이 이루어지지 않는 것을 알 수 있다.<br>
가장 Silhouette score가 낮은 Obesity_st1 을 가지고 hyperparmeters를 튜닝해보았다.<br>

DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=5,eps=0.10)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=5,eps=0.15)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=5,eps=0.20)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=5,eps=0.25)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=10,eps=0.10)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=10,eps=0.15)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=10,eps=0.20)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=10,eps=0.25)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=15,eps=0.10)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=15,eps=0.15)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=15,eps=0.20)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=15,eps=0.25)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=15,eps=0.30)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=30,eps=0.20)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=30,eps=0.25)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=30,eps=0.30)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=30,eps=0.35)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=30,eps=0.40)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=50,eps=0.30)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=50,eps=0.35)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=50,eps=0.40)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=50,eps=0.45)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=100,eps=0.40)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=100,eps=0.45)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=100,eps=0.50)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=100,eps=0.55)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=100,eps=0.60)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=200,eps=0.50)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=200,eps=0.60)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=200,eps=0.70)
DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=200,eps=0.80)


Obesity_st1의 경우, Silhouette score는 eps가 0.7, 0.8 와 같이 높을 때 증가하지만<br>
오히려 낮은 Silhouette score가 도출되게끔 hyper-parameter를 선정했을 시 기존 lable과 가장 비슷하게 군집되는 것을 알 수 있다.<br>
원하는 best hyper-parameter를 찾지는 못했다.

<br>

## 1.3 군집 결과 분석 및 가시화

<br>

앞서 진행한 군집화 분석 결과를 다시 한번 살펴보며<br>
실제 lable을 기준으로 평가해 보았다.

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    
    k_Means_Plot(Data = Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                 Select_k = range(2, 6),
                 NAME = list(Artificial_Obesity.keys())[i],
                 Init_Method='random',Num_Init=10)

for i in range(0,3):
    Simple_Scatter(i, list(Artificial_Obesity.keys())[i])
    Hclust_Plot(Artificial_Obesity[list(Artificial_Obesity.keys())[i]],
                range(2, 6), list(Artificial_Obesity.keys())[i])

DBSCAN_Plot(Data=Obesity_st1,NAME="Obesity_st1",min_samples=10,eps=0.5)
DBSCAN_Plot(Data=Obesity_st2,NAME="Obesity_st2",min_samples=10,eps=0.5)
DBSCAN_Plot(Data=Obesity_st3,NAME="Obesity_st3",min_samples=10,eps=0.5)

* 3가지 분석 방법을 비교해 보았을 때, DBSCAN 분석 방법이 가장 높은 Silhouette score를 도출했다는 것을 알 수 있다.<br>
* 하지만 기존 lable과 비교했을 때 가장 군집이 비슷하지 않게 되었다는 것도 알 수 있다.<br> <br>
* 3가지 분석 방법 모두 Obesity 데이터 셋의 비만도에 따른 기존 군집과 비슷하게 군집화 할 수 없었다.<br> 1. 특정 변수로만 진행했던 점 <br> 2. 기존 lable 군집 형태가 복잡했던 점 <br> 3. 적절한 Hyper parameter를 선정하지 못했던 점 <br> 등의 문제로 인해 완벽한 군집이 진행되지는 못했다. 

### PCA 차원 축소 진행

차원축소를 진행하기 전 방법을 쉽게하기 위해 lable 값을 변환해주었다.

Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Insufficient_Weight', 0)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Normal_Weight', 1)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Overweight_Level_I', 2)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Overweight_Level_II', 3)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Obesity_Type_I', 4)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Obesity_Type_II', 5)
Obesity.NObeyesdad = Obesity.NObeyesdad.replace('Obesity_Type_III', 6)

from sklearn.decomposition import PCA

# 2차원으로 차원 축소, target 정보는 제외
pca = PCA(n_components = 2)
pca.fit(Obesity.iloc[:,:-1])
 
# pca transform 후 데이터프레임으로 자료형 변경
df_pca = pca.transform(Obesity.iloc[:,:-1])
df_pca = pd.DataFrame(df_pca, columns = ['component 0', 'component 1'])

df_pca

lable값이 feature에 포함되어지지 않게 2개의 주성분으로 차원축소를 진행하였다.<br>

# PCA 주성분 설명력 출력

print(pca.explained_variance_ratio_)

주성분 설명력을 출력한 결과 <br>
첫 주성분이 대략 94% 두번 째 주성분이 대략 5퍼의 분산 설명력을 가져 <br>
두 주성분이 총 약 99%를 설명하고 있다는 것을 알 수 있다.

import matplotlib.pyplot as plt

# class target 정보 불러오기 
df_pca['target'] = Obesity['NObeyesdad']

# target 별 분리
df_pca_0 = df_pca[df_pca['target'] == 0]
df_pca_1 = df_pca[df_pca['target'] == 1]
df_pca_2 = df_pca[df_pca['target'] == 2]
df_pca_3 = df_pca[df_pca['target'] == 3]
df_pca_4 = df_pca[df_pca['target'] == 4]
df_pca_5 = df_pca[df_pca['target'] == 5]
df_pca_6 = df_pca[df_pca['target'] == 6]

# target 별 시각화
plt.scatter(df_pca_0['component 0'], df_pca_0['component 1'], color = 'orange', alpha = 0.7, label = 'Insufficient_Weight')
plt.scatter(df_pca_1['component 0'], df_pca_1['component 1'], color = 'red', alpha = 0.7, label = 'Normal_Weight')
plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'green', alpha = 0.7, label = 'Overweight_Level_I')
plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'yellow', alpha = 0.7, label = 'Overweight_Level_II')
plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'blue', alpha = 0.7, label = 'Obesity_Type_I')
plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'violet', alpha = 0.7, label = 'Obesity_Type_II')
plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'black', alpha = 0.7, label = 'Obesity_Type_III')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
plt.show()



<br>
PCA 차원 축소를 진행 후 시각화 하였다.<br>
하지만 무슨 영문인지 중간 몇 가지 lable값을 인식하지 못하여 그래프로 표현하지 못하였다.<br>
나머지 lable값을 통해 두 가지 주성분이 어느정도 알맞게 군집을 형성하고 있다는 것을 알 수 있다.
