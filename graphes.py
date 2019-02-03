# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:00:04 2018

@author: Jérôme
"""

import numpy as np
import matplotlib.pyplot as plt



"""------------------------------------------------------------------------------------"""
"""    Programme permettant le tracé de différents graphiques explicatifs :
        - représentation des variables RFM suivant les clusters
        - représentation d'autres variables ( min depense, max depense, tps commande
        - répartition du clustering                                                    """
"""------------------------------------------------------------------------------------"""



#nbr_clusters=6
nbr_clusters=7

index=np.arange(nbr_clusters)

moyenne_recent_base=(91.04,91.04,91.04,91.04,91.04,91.04,91.04)
moyenne_freq_base=(90.51,90.51,90.51,90.51,90.51,90.51,90.51)
moyenne_somme_base=(2048.22,2048.22,2048.22,2048.22,2048.22,2048.22,2048.22)

'''
moyenne_recent=(175.19,251.29,33.75,79.31,49.99,43.51)
moyenne_freq=(24.28,17.53,170.88,23.13,69.63,29.71)
moyenne_somme=(573.51,207.54,4020.42,1195.37,866.32,384.75)
'''
'''
moyenne_recent=(251.29,79.30,33.75,154.87,49.99,43.51,181.72)
moyenne_freq=(17.53,23.13,170.88,3.12,69.63,29.71,31.09)
moyenne_somme=(207.54,1195.37,4020.42,1073.58,866.32,384.75,412.82)
'''
moyenne_recent=(44.15,218.40,91.70,59.39,269.41,125.91,34.39)
moyenne_freq=(75.27,22.10,5.35,38.86,11.77,30.48,162.60)
moyenne_somme=(949.75,299.47,2828.66,387.96,138.62,480.98,3998.51)



fig, ax = plt.subplots()

bar_width=0.35
opacity=0.4

fig1=ax.bar(index, moyenne_recent,bar_width, alpha=opacity,color='b',label='moyenne_recency')

fig2=ax.bar(index+bar_width, moyenne_freq, bar_width, alpha=opacity, color='y', label='moyenne_frequency')

fig3=ax.bar(index+bar_width*2, moyenne_somme, bar_width, alpha=opacity, color='r', label='total_depense')




ax.set_xlabel('Clusters')
ax.set_ylabel('Moyennes')
ax.set_title('Représentation des clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
ax.legend()

fig.tight_layout()
plt.show()


plt.bar(index, moyenne_recent, color='b', label='moyenne_recency')

plt.xlabel('Clusters')
plt.ylabel('Moyennes')
plt.title('Représentation de la moyenne R des clusters')
plt.show()





fig, ax = plt.subplots()

bar_width=0.4

fig1=ax.bar(index, moyenne_recent_base, bar_width, color='b', label='moyenne_recency_base')

fig2=ax.bar(index+bar_width, moyenne_recent, bar_width, alpha=0.4, color='r', label='moyenne_recency_clusters')

ax.set_xlabel('Clusters')
ax.set_ylabel('Moyennes')
ax.set_title('Représentation de la moyenne Recency des clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()






fig, ax = plt.subplots()

bar_width=0.4

fig3=ax.bar(index, moyenne_freq_base, bar_width, color='b', label='moyenne_frequency_base')

fig4=ax.bar(index+bar_width, moyenne_freq, bar_width, alpha=0.4, color='r', label='moyenne_frequency_clusters')

ax.set_xlabel('Clusters')
ax.set_ylabel('Moyennes')
ax.set_title('Représentation de la moyenne Frequency des clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()







fig, ax = plt.subplots()

bar_width=0.4

fig5=ax.bar(index, moyenne_somme_base, bar_width, color='b', label='moyenne_somme_base')

fig6=ax.bar(index+bar_width, moyenne_somme, bar_width, alpha=0.4, color='r', label='moyenne_somme_clusters')


ax.set_xlabel('Clusters')
ax.set_ylabel('Moyennes')
ax.set_title('Représentation de la moyenne Somme dépensée des clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()




''' ----------------- graphe répartition des clusters suivant les algo Kmean et CAH  ------------ '''


labels = 'cluster 0','cluster 1', 'cluster 2', 'cluster 3', 'cluster 4' , 'cluster 5', 'cluster 6'

size_kmean=[744,460,137,282,392,512,1812]
size_cah=[683,549,1788,107,644,235,333]

fig1, ax1 =plt.subplots()

ax1.pie(size_kmean, labels=labels, shadow=True, autopct='%1.0f%%',startangle=-90)
ax1.axis('equal')
plt.title('Répartition des clusters avec K-mean')
plt.show()


fig2, ax2 = plt.subplots()

ax2.pie(size_cah, labels=labels, autopct='%1.0f%%',shadow=True)
ax2.axis('equal')
plt.title('Répartition des clusters avec CAH')
plt.show()




''' -------------------- graphe des minimums, maximums dépensés, ainsi que moyenne tps entre commande ------'''



minimum_depense=[5.28,8.20,757.63,7.03,15.08,7.50,8.13]
maximum_depense=[78.44,53.36,2566.04,52.15,38.89,68.93,173.26]
tps_commande=[233.16,14.73,24.88,243.00,3.41,20.37,167.40]

fig, ax = plt.subplots()

bar_width=0.4

fig_mini=ax.bar(index, minimum_depense, bar_width, color='b', label='minimum dépensé')

ax.set_xlabel('Clusters')
ax.set_ylabel('Minimum dépense')
ax.set_title('Représentation du minimum dépensé pour chaque clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()




fig_maxi=ax.bar(index, maximum_depense, bar_width, color='b', label='maximum dépensé')

ax.set_xlabel('Clusters')
ax.set_ylabel('Maximum dépense')
ax.set_title('Représentation du maximum dépensé pour chaque clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()


fig_tps=ax.bar(index, tps_commande, bar_width, color='b', label='tps commande')

ax.set_xlabel('Clusters')
ax.set_ylabel('Tps entre première et dernière commande')
ax.set_title('Représentation du temps entre la première et la dernière commande pour chaque clusters')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('0:744', '1:460', '2:137', '3:282', '4:392','5:512','6:1812'))
ax.legend()

fig.tight_layout()
plt.show()
