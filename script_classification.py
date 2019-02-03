# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:47:01 2018

@author: Jérôme
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.model_selection import LearningCurve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from yellowbrick.classifier import ClassPredictionError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, ShuffleSplit
from yellowbrick.features.importances import FeatureImportances




"""---------------------------------------------------------------------------------------"""
"""    Programme de test des différentes méthodes de classification : chaque méthode
    renvoie des tests de validation croisée différentes, avec affichage des scores R²,
    puis scores moyens, affichage des matrices de confusion et affichage de la courbe
    d'apprentissage permettant de visualiser l'impact de la base d'apprentissage et test
    sur la méthode
    affichage du graphe d'erreur de prédiction par cluster                               """
"""--------------------------------------------------------------------------------------"""




''' fichier essai test avec modif equilibrage clusters '''
#df_kmean=pd.read_csv('fichier_modif_clust.csv', sep=',')


''' fichier normal '''
df_kmean=pd.read_csv('fichier_kmean_7clust.csv',sep=',')
df_kmean=pd.read_csv('fichier_kmean.csv',sep=',')

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

x_train,x_test, y_train, y_test = train_test_split(df_valeur,y,test_size=0.3)



''' ------------- Arbre de décision ---------------------------'''



arbre=DecisionTreeClassifier(max_depth=8)   
arbre.fit(x_train,y_train)

print(arbre.score(x_train,y_train))

result=cross_val_score(arbre, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(arbre, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(arbre, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(arbre, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))



cm=ConfusionMatrix(arbre,classes=[0,1,2,3,4,5,6], percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()


size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
lc=LearningCurve(DecisionTreeClassifier(),train_sizes=size,score='r2')
lc.fit(x_train,y_train)
lc.poof()





''' ---------------------- Forêt aléatoire ------------------------'''




foret=RandomForestClassifier(n_estimators=120,max_features='sqrt',n_jobs=-1,random_state=0)
foret.fit(x_train, y_train)


result=cross_val_score(foret, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(foret, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(foret, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(foret, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))



cm=ConfusionMatrix(foret,classes=[0,1,2,3,4,6], percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()

size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]


lc=LearningCurve(RandomForestClassifier(),train_sizes=size,score='r2')
lc.fit(x_train,y_train)
lc.poof()

viz=ClassPredictionError(RandomForestClassifier(),classes=["0","1","2","3","4","5","6"])
viz.fit(x_train, y_train)
viz.score(x_test,y_test)
viz.poof


fig=plt.figure()
ax=fig.add_subplot()

feat=FeatureImportances(RandomForestClassifier(), ax=ax)
feat.fit(x_train, y_train)
feat.poof()





'''--------------------- Réseau de neurones --------------------- '''



neurone=MLPClassifier()
neurone.fit(x_train,y_train)

print(neurone.score(x_test,y_test))

result=cross_val_score(neurone, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(neurone, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(neurone, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(neurone, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))



cm=ConfusionMatrix(neurone,classes=[0,1,2,3,4,5,6], percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()


size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
lc=LearningCurve(MLPClassifier(),train_sizes=size,score='r2')
lc.fit(x_train,y_train)
lc.poof()

vizualiser=ClassPredictionError(MLPClassifier(),classes=[0,1,2,3,4,5,6])
vizualiser.fit(x_train, y_train)
vizualiser.score(x_test,y_test)
vizualiser.poof



''' -------------- AdaBoost ----------------'''



adab=AdaBoostClassifier(n_estimators=50,random_state=0)
adab.fit(x_train, y_train)

print(adab.score(x_test,y_test))


result=cross_val_score(adab, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(adab, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(adab, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(adab, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))

cm=ConfusionMatrix(adab,classes=[0,1,2,3,4,5,6],percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()



''' ---------------- Gradient Boosting ---------------------'''


gbc=GradientBoostingClassifier(n_estimators=90,max_features='sqrt', learning_rate=0.2,random_state=0)
gbc.fit(x_train,y_train)

print(gbc.score(x_test,y_test))


result=cross_val_score(gbc, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(gbc, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(gbc, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(gbc, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))



cm=ConfusionMatrix(gbc,classes=[0,1,2,3,4,5,6], percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()


size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

lc=LearningCurve(GradientBoostingClassifier(),train_sizes=size,score='r2')
lc.fit(x_train,y_train)
lc.poof()

vizualiser=ClassPredictionError(GradientBoostingClassifier(),classes=[0,1,2,3,4,5,6])
vizualiser.fit(x_train, y_train)
vizualiser.score(x_test,y_test)
vizualiser.poof





''' --------------- kppv ------------------------'''


kppv=KNeighborsClassifier(n_neighbors=6, metric='euclidean')
   
kppv.fit(x_train,y_train)

print(kppv.score(x_train,y_train))
  
result=cross_val_score(kppv, x_train,y_train, cv=5)
print(' score cross validation :{}'.format(result))
print('moyenne score cross validation : {:.2f}'.format(result.mean()))

result1=cross_val_score(kppv, x_train,y_train, cv=kplis)
print(' score kplis cross validation :{}'.format(result1))
print('moyenne score cross validation : {:.2f}'.format(result1.mean()))

result2=cross_val_score(kppv, x_train,y_train, cv=kplis_strat)
print(' score kplis strat cross validation :{}'.format(result2))
print('moyenne score cross validation : {:.2f}'.format(result2.mean()))

result3=cross_val_score(kppv, x_train,y_train, cv=shuffle)
print(' score shuffle split cross validation :{}'.format(result3))
print('moyenne score cross validation : {:.2f}'.format(result3.mean()))




cm=ConfusionMatrix(kppv,classes=[0,1,2,3,4,5,6], percent=True)
cm.fit(x_train,y_train)
cm.score(x_test,y_test)
cm.poof()



size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
lc=LearningCurve(KNeighborsClassifier(),train_sizes=size,score='r2')
lc.fit(x_train,y_train)
lc.poof()

visual=ClassPredictionError(KNeighborsClassifier(),classes=[0,1,2,3,4,5,6])
visual.fit(x_train, y_train)
visual.score(x_test,y_test)
visual.poof
