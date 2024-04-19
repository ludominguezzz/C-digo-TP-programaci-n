import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from funpymodeling.data_prep import todf

#Cargo las matrices int, t1 y t2
INT = np.genfromtxt('INT.csv', dtype=int, delimiter=' ')
T1 = np.genfromtxt('PT1.csv', dtype=int, delimiter=' ')
T2 = np.genfromtxt('PT2.csv', dtype=int, delimiter=' ')
A2 = np.genfromtxt('PA2.csv', dtype=int, delimiter=' ')
A1 = np.genfromtxt('PA1.csv', dtype=int, delimiter=' ')

#estadística descriptiva
#print(np.min(INT), np.max(INT), np.mean(INT), np.std(INT))
#print(np.min(T1), np.max(T1), np.mean(T1), np.std(T1))
#print(np.min(A1), np.max(A1), np.mean(A1), np.std(A1))
#print(np.min(T2), np.max(T2), np.mean(T2), np.std(T2))
#print(np.min(A2), np.max(A2), np.mean(A2), np.std(A2))


#Creo una mascara en la matriz intensidades que me indique a donde esta el núcleo. Lo reprento con ceros.
INT1 = np.where(INT < 200, 0, 1)


#multiplico mascara por matrices T1 y T2 para identificar el núcleo en estas.
INTSN = INT*INT1
T1SN = T1 * INT1
T2SN = T2 * INT1
A1SN = A1 * INT1
A2SN = A2 * INT1

### Creo un array multidimensional con las matrices correspondientes a una célula.
celula = np.array([[INTSN], [T1SN], [A1SN], [T2SN], [A2SN]])

#Aplano (flat) el array bidimensional:
T1SN= np.concatenate(T1SN)
T2SN= np.concatenate(T2SN)
A1SN= np.concatenate(A1SN)
A2SN= np.concatenate(A2SN)

#Convierto el array en lista con los datos diferentes de 0 para obtener el promedio de las matirces SIN NUCLEO.
#Repito este bucle con las 4 matrices.

matriz = ["T1", "T1SN", "T2", "T2SN", "A1", "A1SN", "A2", "A2SN"]
promedios = []

T1_MEAN = np.mean(T1)
promedios.append(T1_MEAN)

T1SN_LISTA = []
for i in T1SN:
    if i != 0:
        T1SN_LISTA.append(i)
T1SN_MEAN = np.mean(T1SN_LISTA)
promedios.append(T1SN_MEAN)

T2_MEAN = np.mean(T2)
promedios.append(T2_MEAN)

T2SN_LISTA = []
for i in T2SN:
    if i != 0:
        T2SN_LISTA.append(i)
T2SN_MEAN = np.mean(T2SN_LISTA)
promedios.append(T2SN_MEAN)

A1_MEAN = np.mean(A1)
promedios.append(A1_MEAN)

A1SN_LISTA = []
for i in A1SN:
    if i != 0:
        A1SN_LISTA.append(i)
A1SN_MEAN = np.mean(A1SN_LISTA)
promedios.append(A1SN_MEAN)

A2_MEAN = np.mean(A2)
promedios.append(A2_MEAN)

A2SN_LISTA = []
for i in A2SN:
    if i != 0:
       A2SN_LISTA.append(i)
A2SN_MEAN = np.mean(A2SN_LISTA)
promedios.append(A2SN_MEAN)

fig, ax = plt.subplots()
ax.bar(x = matriz, height = promedios)
plt.show()


"""
#Uno los valores formando un vector de dos elementos:
T1T2 = np.array(list(zip(T1SN, T2SN))).reshape(len(T2SN), 2)

#Convierto el vector de dos elementos en un data frame de 2 columnas: 0 = t1 y 1 = t2.
DF = todf(T1T2)

fl = []
for i in DF[1]:
    if i > 2200 and i < 2500:   #lipofuscina
        fl.append(1)
    elif i > 2500 and i < 3000: #fad
        fl.append(2)
    elif i == 0:   #nucleo 
        fl.append(0)
    else:   #otros
        fl.append(3)

#Agrego al df la columna correspondiente al tipo de fluoróforo.
DF[2] = fl


#grafico
plt.scatter(DF[0], DF[1], c=DF[2], marker='*', s=3)
plt.xlabel('T1')
plt.ylabel('T2')
plt.title("T1-T2 Ex:405 nm - Em:500-550 nm")
plt.show()


#Algoritmo K-means:
kmeans = KMeans(n_clusters=4).fit(DF)

#obtener los centroides
centroides = kmeans.cluster_centers_

plt.scatter(DF[0], DF[1], c=kmeans.labels_, s=3)
plt.scatter(centroides[:,0], centroides[:,1], c='red', marker='*', s=50)
plt.xlabel('T1')
plt.ylabel('T2')
plt.title("Clustering T1-T2 Ex:405 nm - Em:500-550 nm")
plt.show()

"""


