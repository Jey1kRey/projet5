# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:47:51 2018

@author: Jérôme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from yellowbrick.classifier import ROCAUC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle


"""-------------------------------------------------------------------------------"""
"""    Programme permettant le tracé des graphes de la courbe ROC avec l'affichage
    des aires sous la courbe pour les différentes méthodes de classification      """
"""-------------------------------------------------------------------------------"""



df_kmean=pd.read_csv('fichier_kmean_7clust.csv',sep=',')


df_kmean=df_kmean.set_index(['CustomerID'])



df_test=df_kmean[['min_depense','moy_depense']]
df_var=df_kmean.iloc[:,9:]
df_var=df_var.drop(['label_cluster'], axis=1)

df_valeur=df_kmean.drop(['label_cluster'], axis=1)



df_valeur=df_valeur.iloc[:,:9]

         

y=df_kmean['label_cluster']


kplis=KFold(n_splits=5)
kplis_strat=StratifiedKFold(n_splits=5, random_state=0)
shuffle=ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=5)

y_bin=label_binarize(y, classes=[0,1,2,3,4,5,6])
n_classes=y_bin.shape[1]


x_train,x_test, y_train, y_test = train_test_split(df_valeur,y_bin,test_size=0.3)






''' établissement de chacun des algorithmes de classification ''' 


arbre=DecisionTreeClassifier(max_depth=8)   
arbre.fit(x_train,y_train)
y_theo=arbre.predict(x_test)


foret=RandomForestClassifier(n_estimators=120,max_features='sqrt',n_jobs=-1,random_state=0)
foret.fit(x_train, y_train)
y_theo=foret.predict(x_test)



neurone=MLPClassifier()
neurone.fit(x_train,y_train)
y_theo=neurone.predict(x_test)


gbc=GradientBoostingClassifier(n_estimators=130,max_features='sqrt',random_state=0)
gbc.fit(x_train,y_train)
y_theo=gbc.predict(x_test)

adab=AdaBoostClassifier(n_estimators=50,random_state=0)
adab.fit(x_train,y_train)
y_theo=adab.predict(x_test)


kppv=KNeighborsClassifier(n_neighbors=6, metric='euclidean')
kppv.fit(x_train,y_train)
y_theo=kppv.predict(x_test)




''' -------------------calcul de la courbe ROC pour chaque classe------------------- '''

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_theo[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_theo.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





'''-------------------- graphe des courbes ROC pour chaque classe ------------------------'''

# récupération des faux positifs
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# calcul de AUC
mean_tpr /= n_classes


fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# graphe de toutes les courbes ROC avec les AUC en légendes

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','grey','blue','yellow'])

for i, color in zip(range(n_classes), colors):
    
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC et surface sous la courbe')
plt.legend(loc="inférieur droit")
plt.show()

