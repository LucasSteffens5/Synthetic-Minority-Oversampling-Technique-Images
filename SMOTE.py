# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 14:02:41 2021

@author: lucasoliveira
"""

# Sobreamostrando base de dados desbalanceada utilizando do SMOTE
from collections import Counter
from imblearn.over_sampling import ADASYN #SVMSMOTE, SMOTE , ADASYN # Altere a variação de SMOTE aqui
from matplotlib import pyplot
from numpy import where
from keras_preprocessing.image import ImageDataGenerator
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from PIL import Image

# Variaveis globais
dimensaoDeEntrada = (250,250)

diretorioBaseDeDados = 'C:\\Users\\lucasoliveira\\Desktop\\' # Altere para o diretorio em sua maquina
caminhoImagensSobreamostradas = 'C:\\Users\\lucasoliveira\\Desktop\\ImagensSobreamostradas\\maligna'  # Altere para o diretorio em sua maquina
nomePastaDasImagens = 'BaseBM'
# Carregando a base de dados para a memoria
diretorioDasImagens = os.path.join(diretorioBaseDeDados, nomePastaDasImagens)

geradorDeDados = ImageDataGenerator(rescale=1./255)
tamanhoDoLote = 32

# Região de funções auxiliares

def retornaNomeDasPastasPresentesNoDiretorio(path): # Percorre todo diretorio e pega o nome da pasta do arquivo
    for p, folder, files in os.walk(os.path.abspath(path)):
        return folder


def extraiVetoresDeImagens(diretorio):   
    gerador = geradorDeDados.flow_from_directory(
        diretorio,
        target_size=dimensaoDeEntrada,
        batch_size=tamanhoDoLote)
    
    i = 0
    contagem = gerador.samples
    rotulos = np.zeros(shape=(contagem,2)) # Define o vetor de rótulos
    vetoresDecaracteristica = np.zeros(shape=(contagem, dimensaoDeEntrada[0], dimensaoDeEntrada[1], 3))
    print(gerador.class_indices)
    
    for inputs_batch, pacoteDosRotulos in gerador:        
        vetoresDecaracteristica[i * tamanhoDoLote : (i + 1) * tamanhoDoLote] = inputs_batch     
        rotulos[i * tamanhoDoLote : (i + 1) * tamanhoDoLote] = pacoteDosRotulos
        i += 1
        
        if i * tamanhoDoLote >= contagem:
            break
        
    rotulos = rotulos[:,1:]     
    rotulos = rotulos.astype(np.int32)
    rotulos = rotulos.flatten()
    return vetoresDecaracteristica, rotulos, contagem

def plotaDistribuicao2D(contador):
    for rotulo, _ in contador.items():
    	linha_x = where(y == rotulo)[0]
    	pyplot.scatter(X[linha_x, 0], X[linha_x, 1], label=str(rotulo))
    pyplot.legend()
    pyplot.show()
    

def salvaImagensDaClasseSobreamostrada(vetorDeImagens,rotulosDasImagens,classeSobreamostrada, caminhoImagensSobreamostradas):
    contadorAuxiliar = 0
    for im in vetorDeImagens:
        if(rotulosDasImagens[contadorAuxiliar] == classeSobreamostrada):
            im=im*255.0 # Desnormaliza as imagens
            teste = Image.fromarray(np.uint8(im))
            teste.save(caminhoImagensSobreamostradas+str(contadorAuxiliar)+'.png')
        contadorAuxiliar+=1

# Final Região de funções auxiliares






#Iniciando o algoritmo


X, y, qtdImagens= extraiVetoresDeImagens(diretorioDasImagens)

X = X.reshape(X.shape[0], -1) # np.reshape(X, (qtdImagens, dimensaoDeEntrada[0]*dimensaoDeEntrada[1]*3))#

        

# resumir a distribuição das classes
contador = Counter(y)
print(contador)
plotaDistribuicao2D(contador)

# reamostra base de dados
oversample = ADASYN(sampling_strategy = 'minority',n_jobs = -1)
X, y = oversample.fit_resample(X, y)


X_res = X.reshape(X.shape[0], dimensaoDeEntrada[0], dimensaoDeEntrada[1], 3) # Volta vetores de imagens para dimensões origianis
# resume nova distribuição dos dados

contador = Counter(y)
plotaDistribuicao2D(contador)
print(contador)

salvaImagensDaClasseSobreamostrada(X_res,y,1,caminhoImagensSobreamostradas)