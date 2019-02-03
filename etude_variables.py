# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:18:31 2018

@author: Jérôme
"""

import pandas as pd
import datetime as dt


"""----------------------------------------------------------------------------------------------"""
"""    Programme permettant de nettoyer la base de données et créer de nouvelles variables : 
    Les variables RFM, nommées ici nbr commande pour F, Dep_der_commande pour R
    et total_depense pour M. D'autres variables comme le minimum dépensé, le maxi, la moyenne,
    le temps écoulé entre la première et la dernière commande...                                 """
"""----------------------------------------------------------------------------------------------"""




df=pd.read_excel('vente.xlsx')


#print(df.apply(lambda x : sum(x.isnull()),axis=0))


''' suppression des lignes vides : on ne peut pas imputer le customerId autrement '''
df=df.dropna()


''' il reste 406829 lignes sur 541909 : nbr de vide dans customerID = 135080 soit environ 25% de la base '''
#print(len(df))


''' vérification de la quantité : certaines quantités sont négatives et représentent 8905 lignes, soit 2% de la base'''
df=df[(df['Quantity']>0)]
#print(len(df_quantite))


''' vérification des problèmes de quantité : difficile de comprendre certaines quantités négatives : elles
apparaissent illogiques souvent au regard du type d'article '''

#df_quant_neg=df[(df['Quantity']<0)]
#print(df_quant_neg.head(20))


''' vérification des pays d'origine des clients : plus de 354 000 viennent d'UK ( sur 397924 ) : filtrage possible.'''
#print(df.Country.nunique())

df=df.drop_duplicates()

#print(df.describe())


''' définition de la date actuel : la plus ancienne enregistrée '''
jour_actuel=dt.datetime(2011,12,9)



'''---------- ajout d'une colonne : prix total = quantité * prix unitaire---------- '''

df['prix_total']=df['Quantity']*df['UnitPrice']


''' -------------- Mise en place des variables RFM et d'autres variables ------------------------'''

# recherche de stats sur le panier d'un visiteur : le minimum dépensé, maxi, la moyenne et le nbr de transactions

stats_client=df.groupby(['CustomerID'])['prix_total'].agg(['count','min','max','mean','sum'])

stats_client.columns=['nbr_commande','min_depense','max_depense','moy_depense','total_depense']



# recherche du temps écoulé entre la première commande et la dernière, par rapport à la date jour_actuel

table_min=df.groupby(['CustomerID']).agg({'InvoiceDate': lambda date : (jour_actuel-date.min()).days})
table_max=df.groupby(['CustomerID']).agg({'InvoiceDate': lambda date : (jour_actuel-date.max()).days})

table_date=pd.concat([table_min,table_max], axis=1)

table_date.columns=['Dep_prem_commande','Dep_der_commande']
table_date['temps_commande']=table_date['Dep_prem_commande']-table_date['Dep_der_commande']


# fusion des bases 

base=pd.concat([stats_client, table_date], axis=1)

base['moy_tps_commande']=base['temps_commande']/base['nbr_commande']

#print(base.head(10))


# recherche du nombre de clients n'ayant fait qu'un seul achat ( résultat = 1.66 % )
nbr_client_unique=base[base['nbr_commande']==1].shape[0]
nbr_client=base.shape[0]
#print("nbr de clients avec achat unique : ", nbr_client_unique, "soit en pourcentage de la base : ", (nbr_client_unique/nbr_client)*100)


base.to_csv('base_donnees.csv',sep=',')
