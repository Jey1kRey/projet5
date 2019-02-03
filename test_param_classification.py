# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:31:49 2018

@author: Jérôme
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold



"""---------------------------------------------------------------------"""
"""    Programme permettant de tester les hyperparamètres des méthodes
    de classification
    
    Attention : les modules de tests des méthodes ont été écrits pour
    être lancés séparément. Ajouter des guillements de commentaires pour
    lancer les modules                                                   """
"""----------------------------------------------------------------------"""




df_kmean=pd.read_csv('fichier_kmean_7clust.csv',sep=',')


df_kmean=df_kmean.set_index(['CustomerID'])



df_test=df_kmean[['min_depense','moy_depense']]
df_var=df_kmean.iloc[:,9:]
df_var=df_var.drop(['label_cluster'], axis=1)

df_valeur=df_kmean.drop(['label_cluster'], axis=1)

                

y=df_kmean['label_cluster']


kplis=KFold(n_splits=5)
kplis_strat=StratifiedKFold(n_splits=5, random_state=0)
shuffle=ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=5)

x_train,x_test, y_train, y_test = train_test_split(df_valeur,y,test_size=0.3)









''' ----------------- Arbre de décision -----------------'''

meilleur_score=0
profondeur=[5,7,10,12,15]

for k in profondeur : 

    arbre=DecisionTreeClassifier(max_depth=k)   
    arbre.fit(x_train,y_train)
    score=arbre.score(x_test,y_test)
    
    if score > meilleur_score :
        meilleur_score=score
        meilleur_param={'profondeur': k}
        
    cm=ConfusionMatrix(arbre,classes=[0,1,2,3,4,5,6], percent=True)
    cm.fit(x_train,y_train)
    cm.score(x_test,y_test)
    cm.poof()

print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param))




''' ----------- Forêt Aléatoire ----------------------'''

meilleur_score=0

nbr_arbre=[20,40,60,80,100,120,140]
features=['sqrt','log2']
critere=['gini','entropy']

for arbre_choix in nbr_arbre:
    
    for choix_features in features:
        
        for crit in critere :
        
            foret=RandomForestClassifier(n_estimators=arbre_choix, max_features=choix_features, criterion=crit, n_jobs=-1, random_state=0)
            foret.fit(x_train, y_train)

            score=foret.score(x_test,y_test)
    
            if score > meilleur_score :
                meilleur_score=score
                meilleur_param={'arbre_choix': arbre_choix, 'features':choix_features, 'critere': crit }

            cm=ConfusionMatrix(foret,classes=[0,1,2,3,4,5,6], percent=True)
            cm.fit(x_train,y_train)
            cm.score(x_test,y_test)
            cm.poof()

print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param))




''' ------------------------ GradientBoosting --------------------'''



meilleur_score = 0

nbr_estimateur=[80,90,100,110,120,130,140]
learning=[0.05,0.1,0.15,0.2]
features=['auto','sqrt','log2']

for estim in nbr_estimateur:
    
    for choix in features:
        
        for x in learning : 
            
            gbc=GradientBoostingClassifier(n_estimators=estim ,max_features=choix, learning_rate=x)
            gbc.fit(x_train, y_train)

            score=gbc.score(x_test,y_test)
    
            if score > meilleur_score :
                meilleur_score=score
                meilleur_param={'estimateur': estim, 'features': choix, 'learning': x }
            
            cm=ConfusionMatrix(gbc,classes=[0,1,2,3,4,5,6], percent=True)
            cm.fit(x_train,y_train)
            cm.score(x_test,y_test)
            cm.poof()
        
print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param))



''' ------------------ AdaBoostClassifier ---------------------'''

meilleur_score = 0

nb_estimateur=[40,50,60,70,80]

for estim in nb_estimateur:
    
    adab=AdaBoostClassifier(n_estimators=estim)
    adab.fit(x_train,y_train)
    
    score=adab.score(x_test,y_test)
    
    if score>meilleur_score:
        meilleur_score=score
        meilleur_param={'estimateur':estim}
        
    cm=ConfusionMatrix(adab,classes=[0,1,2,3,4,5,6], percent=True)
    cm.fit(x_train,y_train)
    cm.score(x_test,y_test)
    cm.poof()

print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param))



''' ---------- réseau de neurones ------------'''


meilleur_score=0

couche_cache=[60,80,100,120]

for k in couche_cache:
    
    neurone=MLPClassifier(hidden_layer_sizes=k)
    neurone.fit(x_train,y_train)
    
    score=neurone.score(x_test,y_test)
    
    if score>meilleur_score:
        meilleur_score=score
        meilleur_param={'couche_cache':k}

    cm=ConfusionMatrix(neurone,classes=[0,1,2,3,4,5,6], percent=True)
    cm.fit(x_train,y_train)
    cm.score(x_test,y_test)
    cm.poof()
print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param))   



''' ---------------------- Kppv -------------------- '''

meilleur_score=0

voisins=[3,4,5,6,7,8,9,10]

for k in voisins:
    
    kppv=KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    kppv.fit(x_train, y_train)

    score=kppv.score(x_test,y_test)

    if score>meilleur_score:
        meilleur_score=score
        meilleur_param={'voisins':k}
        
    cm=ConfusionMatrix(kppv,classes=[0,1,2,3,4,5,6], percent=True)
    cm.fit(x_train,y_train)
    cm.score(x_test,y_test)
    cm.poof()
print('meilleur score : {:.2f}'.format(meilleur_score))
print('meilleur paramètre : {}'.format(meilleur_param)) 
