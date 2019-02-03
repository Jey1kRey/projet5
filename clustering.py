# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:37:57 2018

@author: Jérôme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


"""-----------------------------------------------------------------------------------------"""
"""    Programme regroupant d'une part les nouvelles variables RFM et autres, d'autre part
    les variables mots avec ACP, puis création du clustering. 
    Deux méthodes sont effectuées : le K-mean et l'arbre hiérarchique ascendant
    
    Le partitionnement est vérifié via la méthode du coude, la stabilité, l'indice
    de silhouette                                                                          """
"""----------------------------------------------------------------------------------------"""






df_origine=pd.read_excel('vente.xlsx')




df_rfm=pd.read_csv("base_donnees.csv", sep=',', low_memory=False)
df_rfm=df_rfm.set_index(['CustomerID'])



df_acp=pd.read_csv('df_tt_vocab.csv',sep=',')
df_acp=df_acp.drop(['Unnamed: 0'], axis=1)



groupe=df_acp.groupby(['CustomerID']).sum()



df=pd.concat([df_rfm,groupe], axis=1)

df=df.fillna(0)

indices=df.index


matrice=np.array(df)

matrice_norme=normalize(matrice)




'''----- test du nombre de clusters, méthode du coude et indice de silhouette pour retenir le bon nombre de clusters-- '''
inertie=[]

for i in range(2,11):
    k_moyenne=KMeans(n_clusters=i,init='k-means++',random_state=0)
    k_moyenne.fit(matrice_norme)
    inertie.append(k_moyenne.inertia_)
    
    clust=k_moyenne.predict(matrice_norme)
    silhouette=silhouette_score(matrice_norme,clust)
    
    print(' pour ', i, ' clusters, on obtient un score silhouette de :', silhouette)

plt.plot(range(2,11),inertie)
plt.xlabel('nbr clusters')
plt.ylabel('inertie')
plt.show()





'''------- mise en place de la méthode k-means -----------------'''


kmeans=KMeans(n_clusters=7,init='k-means++', max_iter=3000,random_state=0)
kmeans.fit(matrice_norme)

x=kmeans.labels_

print(len(x))

'''------- intégration dans le dataframe d'une colonne contenant le numéro du cluster---'''

label_kmeans=pd.Series(x, name='label_cluster')

cluster_kmean=pd.DataFrame(x, index=indices, columns=['label_cluster'])

df_kmean=pd.concat([df,cluster_kmean],axis=1)

print(" répartition clusters avec K mean : ", df_kmean['label_cluster'].value_counts())

df_kmean.to_csv('fichier_kmean_7clust.csv',sep=',')




'''---------------- Mise en place de l'arbre hiérarchique ascendant--------------- '''



for i in range(2,11):
    
    clust_ah=AgglomerativeClustering(n_clusters=i,affinity='euclidean',linkage='ward')
    
    clust=clust_ah.predict(matrice_norme)
    silhouette=silhouette_score(matrice_norme,clust)
    
    print(' pour ', i, ' clusters, on obtient un score silhouette de :', silhouette)



cah=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='ward')

cah.fit_predict(matrice_norme)

    
''' test du tracé du dendrogramme'''
z=linkage(matrice, method='ward',metric='euclidean')
plt.figure(figsize=(10, 10))  
plt.title("Dendrogramme clustering clients")  
dend = dendrogram(z) 


x=cah.labels_


''' ---------------récupération des numéros de cluster dans une colonne et insertion dans le dataframe------- '''

cluster_cah=pd.DataFrame(x, index=indices, columns=['label_cluster'])

df_cah=pd.concat([df,cluster_cah],axis=1)

print(" répartition clusters avec CAH ", df_cah['label_cluster'].value_counts())

df_cah.to_csv('fichier_cah_7clust.csv',sep=',')





'''-------------- interprétation des clusters : récupération de certaines informations en vue de faciliter
l'interprétation de la répartition des clusters ----------------------------------------------------------'''


print(' La moyenne Recency sur la base est de :', df['Dep_der_commande'].mean())
print(' La moyenne Frequency sur la base est de :', df['nbr_commande'].mean())
print(' La moyenne Monetary sur la base est de :', df['total_depense'].mean())
print(' la moyenne du temps écoulé entre la première et la dernière commande est de :', df['temps_commande'].mean())

def stats_clusters(cluster):
    
    #df_clust_cah=df_cah[df_cah['label_cluster']==cluster]
    
    df_clust_kmean=df_kmean[df_kmean['label_cluster']==cluster]
    print(' le nombre de clients inclus dans le cluster ', cluster, ' est de :', len(df_clust_kmean))
    print(' La moyenne Recency pour le cluster', cluster, ' est de :', df_clust_kmean['Dep_der_commande'].mean())
    print(' La moyenne Frequency pour le cluster', cluster, ' est de :', df_clust_kmean['nbr_commande'].mean())
    print(' La moyenne Monetary pour le cluster', cluster, ' est de :', df_clust_kmean['total_depense'].mean())
    print(' la moyenne du temps écoulé entre la première et la dernière commande pour le cluster', cluster, ' est de :', df_clust_kmean['temps_commande'].mean())
    print(' la moyenne du minimum dépensé pour le cluster', cluster, 'est de ', df_clust_kmean['min_depense'].mean())
    print(' la moyenne du maximum dépensé pour le cluster ',cluster, ' est de ', df_clust_kmean['max_depense'].mean())
    
    
    '''
    print(' le nombre de clients inclus dans le cluster ', cluster, ' est de :', len(df_clust_cah))
    print(' La moyenne Recency pour le cluster', cluster, ' est de :', df_clust_cah['Dep_der_commande'].mean())
    print(' La moyenne Frequency pour le cluster', cluster, ' est de :', df_clust_cah['nbr_commande'].mean())
    print(' La moyenne Monetary pour le cluster', cluster, ' est de :', df_clust_cah['total_depense'].mean())
    print(' la moyenne du temps écoulé entre la première et la dernière commande pour le cluster', cluster, ' est de :', df_clust_cah['temps_commande'].mean())
    '''
for i in range(0,6):
    stats_clusters(i)



