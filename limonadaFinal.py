#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:52:00 2019

@author: davidosollo
"""

#TAKAGI SUGENO

#LIBSS

import math
import matplotlib.pyplot as plt
import numpy as np
import random


from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt


# imports specific to the plots in this example
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data


####################
#F U N C I O N E S()
#################################################

def PrimeraGen(NumCromo_F):
    for i in range(0,NumCromo_F): 
        for j in range(0, NumCols): 
            
            if j < 28:
                MatPadres[i][j] = random.randint(1, 1000)
            elif j> 27 and j < 33:
                MatPadres[i][j] = random.randint(2, 1000) 
            else:
                MatPadres[i][j] = random.randint(1, 1000)
           

        CalculateError(MatPadres,i)

def Torneo():
    global numCompe
    for i in range(0,NumCromo):
        ErrorChamp = 9000000
        CompeChamp = 0
        for j in range(0,numCompe):
            Compe = random.randint(0, NumCromo-1)
            if ErrorChamp > MatPadres[Compe][NumCols]:
                CompeChamp = Compe
                ErrorChamp = MatPadres[Compe][NumCols]
                
                
        for k in range(0,NumCols+1):
            MatHijos[i][k] = MatPadres[CompeChamp][k]
            
# Reproduccion            
def Reproduccion():
    CicloRep = NumCromo//2
    MatBaby = [[0 for i in range(NumCols+1)] for i in range(2)]
        
    for i in range(0,CicloRep,2):
        SegPos = random.randint(0,(ByteSize*NumCols)-2)
        byteNum = int(SegPos//ByteSize);
        NumCorr = SegPos - (byteNum * ByteSize)
        
        NumPart = int(MatHijos[i][byteNum]);
        cromo_x = NumPart >> NumCorr;
        cromo_x = cromo_x << NumCorr;
        cromo_y = NumPart - cromo_x
        
        NumPart = int(MatHijos[i+1][byteNum]);
        cromo_xx = NumPart >> NumCorr;
        cromo_xx = cromo_xx << NumCorr;
        cromo_yy = NumPart - cromo_xx
        
        for j in range(0,NumCols+1):
            if j == byteNum:
                MatBaby[0][j]= cromo_x + cromo_yy
                MatBaby[1][j]= cromo_xx + cromo_y
            elif j < byteNum:
                MatBaby[0][j] =  (MatHijos[i][j])
                MatBaby[1][j] =  (MatHijos[i+1][j])
            else:
                MatBaby[1][j] =  (MatHijos[i][j])
                MatBaby[0][j] =  (MatHijos[i+1][j])
                
                
        for k in range(0,NumCols):
            if MatBaby[0][k] == 0:
                MatBaby[0][k] = 1
                
            if MatBaby[1][k] == 0:
                MatBaby[1][k] = 1        
                
                
        CalculateError(MatBaby,0)
        CalculateError(MatBaby,1)
        
        for k in range(0,NumCols+1):
            MatHijos[i][k] = MatBaby[0][k]
            MatHijos[i+1][k] = MatBaby[1][k]
        


def ChangeBit():
    for i in range(0,NumCromo): 
        BitSta = 0
        SegPos = random.randint(0,(ByteSize*NumCols))
        byteNum = int(SegPos//ByteSize);
        NumCorr = SegPos - (byteNum * ByteSize)
        NumCheck = int(math.pow(2,NumCorr))
        VarMuta=MatHijos[i][byteNum]
        
        if VarMuta & NumCheck != 0:
             BitSta = 1
             
        if BitSta == 1:
             VarMuta = VarMuta - NumCheck
        else:
             VarMuta = VarMuta + NumCheck
             
        if VarMuta == 0:
            VarMuta = 1
        MatHijos[i][byteNum] = VarMuta 
        CalculateError(MatHijos,i)
        

def CopyMat():
    for i in range(0,NumCromo): 
        for j in range(0, NumCols+1): 
            MatPadres[i][j] = MatHijos[i][j]
            
def GetChamp():
    ErrorChamp = 9000000
    CompeChamp = 0
    
    for k in range(0,NumCromo):
        if ErrorChamp > MatPadres[k][NumCols]:
            CompeChamp = k
            ErrorChamp = MatPadres[k][NumCols]

    CalculateError(MatPadres,CompeChamp)    
    
def Elitismo():
    
     for i in range(0,NumCromo):
         for j in range(0, NumCols+1):            
             MatElite[i][j] = MatPadres[i][j]
            
     for i in range(NumCromo,NumCromo*2): 
         for j in range(0, NumCols+1): 
             MatElite[i][j] = MatHijos[i-NumCromo][j]
             
     MatElite.sort(key=takeCol)
     
     for i in range(0,NumCromo):
         for j in range(0, NumCols+1):            
             MatPadres[i][j] = MatElite[i][j] 
             
# Teke value for a column
def takeCol(elem):
    return elem[NumCols]

def Graficar(matGra):
    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))

    #---- First subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(azucar, limon, sabor, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(azucar, limon, matGra, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    
    ax.set_xlabel('azucar')
    ax.set_ylabel('limon')
    ax.set_zlabel('SABOR');

    plt.show()



def GaussCalc(Mat,Ren):
    Media =0
    DesvStd=0
    
    # Reglas 
    pSumTotCal = 0
    for i in range(0,NumPuntos):
        
        Media =Mat[Ren][27]/divMedia
        DesvStd=Mat[Ren][33]/divDevStd
        pbR_x[i]  = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + pbR_x[i]
        
        Media =Mat[Ren][28]/divMedia
        DesvStd=Mat[Ren][34]/divDevStd
        pmR_x[i] = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + pmR_x[i]
        
        Media =Mat[Ren][29]/divMedia
        DesvStd=Mat[Ren][35]/divDevStd
        paR_x[i] = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + paR_x[i]
        
        Media =Mat[Ren][30]/divMedia
        DesvStd=Mat[Ren][36]/divDevStd
        pbR_y[i] = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + pbR_y[i]
        
        Media =Mat[Ren][31]/divMedia
        DesvStd=Mat[Ren][37]/divDevStd
        pmR_y[i] = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + pmR_y[i]
        
        Media =Mat[Ren][32]/divMedia
        DesvStd=Mat[Ren][38]/divDevStd
        paR_y[i] = math.exp(-.5*pow(((i+1)-(Media))/DesvStd,2))
        pSumTotCal = pSumTotCal + paR_y[i]
         
    return pSumTotCal 
        
def TT_Takagi_Calc(Mat,Ren,pSumTot):
    #NumPuntos = 0
    for i in range(0,NumPuntos ):
        for j in range(0,NumPuntos):
            
            pSumGauss  = pbR_x[i]
            pSumGauss = pSumGauss + pmR_x[i]
            pSumGauss = pSumGauss + paR_x[i]
            
            pSumGauss = pSumGauss + pbR_y[j]
            pSumGauss = pSumGauss + pmR_y[j]
            pSumGauss = pSumGauss + paR_y[j]
            
            P = Mat[Ren][0]/divFact
            Q = Mat[Ren][9]/divFact
            R = Mat[Ren][18]/divFact
            pSumTakagui =               ((P*(i+1))+(Q*(j+1))+R) * paR_x[i] * paR_y[j]
            
            P = Mat[Ren][1]/divFact
            Q = Mat[Ren][10]/divFact
            R = Mat[Ren][19]/divFact        
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1))+R) * paR_x[i] * pmR_y[j]
            
            P = Mat[Ren][2]/divFact
            Q = Mat[Ren][11]/divFact
            R = Mat[Ren][20]/divFact      
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1))+R) * paR_x[i] * pbR_y[j]
            
            P = Mat[Ren][3]/divFact
            Q = Mat[Ren][12]/divFact
            R = Mat[Ren][21]/divFact
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1))+R) * pmR_x[i] * paR_y[j]
            
            P = Mat[Ren][4]/divFact
            Q = Mat[Ren][13]/divFact
            R = Mat[Ren][22]/divFact      
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1)+R)) * pmR_x[i] * pmR_y[j]
            
            P = Mat[Ren][5]/divFact
            Q = Mat[Ren][14]/divFact
            R = Mat[Ren][23]/divFact      
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1)+R)) *  pmR_x[i] * pbR_y[j]
            
            P = Mat[Ren][6]/divFact
            Q = Mat[Ren][15]/divFact
            R = Mat[Ren][24]/divFact
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1)+R)) * pbR_x[i] * paR_y[j]
            
            P = Mat[Ren][7]/divFact
            Q = Mat[Ren][16]/divFact
            R = Mat[Ren][25]/divFact      
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1)+R)) * pbR_x[i] * pmR_y[j]
            
            P = Mat[Ren][8]/divFact
            Q = Mat[Ren][17]/divFact
            R = Mat[Ren][26]/divFact      
            pSumTakagui = pSumTakagui + ((P*(i+1))+(Q*(j+1)+R)) *  pbR_x[i] * pbR_y[j]
            
            
            #pSumTakagui =               ((Mat[Ren][0]/10*(i+1))+(Mat[Ren][9 ]/10*(j+1))+Mat[Ren][18]/20) * paR_x[i] * paR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][1]/10*(i+1))+(Mat[Ren][10]/10*(j+1))+Mat[Ren][19]/20) * paR_x[i] * pmR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][2]/10*(i+1))+(Mat[Ren][11]/10*(j+1))+Mat[Ren][20]/20) * paR_x[i] * pbR_y[j]
        
            #pSumTakagui = pSumTakagui + ((Mat[Ren][3]/10*(i+1))+(Mat[Ren][12]/10*(j+1))+Mat[Ren][21]/20) * pmR_x[i] * paR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][4]/10*(i+1))+(Mat[Ren][13]/10*(j+1))+Mat[Ren][22]/20) * pmR_x[i] * pmR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][5]/10*(i+1))+(Mat[Ren][14]/10*(j+1))+Mat[Ren][23]/20) * pmR_x[i] * pbR_y[j]
            
            #pSumTakagui = pSumTakagui + ((Mat[Ren][6]/10*(i+1))+(Mat[Ren][15]/10*(j+1))+Mat[Ren][24]/20) * pbR_x[i] * paR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][7]/10*(i+1))+(Mat[Ren][16]/10*(j+1))+Mat[Ren][25]/20) * pbR_x[i] * pmR_y[j]
            #pSumTakagui = pSumTakagui + ((Mat[Ren][8]/10*(i+1))+(Mat[Ren][17]/10*(j+1))+Mat[Ren][26]/20) * pbR_x[i] * pbR_y[j]
            
            Mat_Val_z[i][j] =  pSumTakagui / pSumTot 
            saborResu[i,j] =  pSumTakagui / pSumTot
            

        
def CalculateError(Mat,Cromo):
    global ErrorChampGen 
    global ErrorChampGlobal
    global saborChamp 
    pSumTot = GaussCalc(Mat,Cromo)
    TT_Takagi_Calc(Mat,Cromo,pSumTot)
    AcuError = 0;
    for i in range(0,NumPuntos):
        for j in range(0,NumPuntos):
            AcuError = AcuError + abs(sabor[i,j] - saborResu [i,j])
        
    Mat[Cromo][NumCols] = AcuError 
    ErrorChampGen = AcuError 
    if AcuError < ErrorChampGlobal:
        ErrorChampGlobal = AcuError
        saborChamp = saborResu.copy()
        
    

        
#################################################
#M A I N ()
#################################################
        
#Variables Globales
        
limon   = np.array([1, 2, 3, 4, 5])
azucar  = np.array([1, 2, 3, 4, 5])
sabor   = np.array([[1, 2, 2, 2, 1],
                    [3, 7, 9, 7, 2],
                    [2, 9, 10, 9, 3],
                    [2, 6, 8, 5, 2],
                    [1, 1, 1, 1, 1]])    

saborResu   = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])      
    

saborChamp   = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])      
        
        
SizeDatos = 60
DivRes = SizeDatos / 10    
    
NumCromo =500
NumGeneracion = 100
NumCols = 39
ByteSize = 10

numCompe = 2
ErrorChampGlobal = 999999
ErrorChampGen = 999999

ChampCromo = 0

NumPuntos = 5

MaxNum = 10

divMedia = 130
divDevStd = 130
divFact = 70

#ARRAY 

ReSol = []

x = [0 for i in range(NumPuntos)]
y = [0 for i in range(NumPuntos)]

pbR_x = [0 for i in range(NumPuntos)]
pmR_x = [0 for i in range(NumPuntos)]
paR_x = [0 for i in range(NumPuntos)]

pbR_y = [0 for i in range(NumPuntos)]
pmR_y = [0 for i in range(NumPuntos)]
paR_y = [0 for i in range(NumPuntos)]


X = []
Y = []
Z = []


MatPadres = [[0 for i in range(NumCols+1)] for i in range(NumCromo)]
Mat_Val_z = [[0 for i in range(NumPuntos)] for i in range(NumPuntos)]
MatHijos = [[0 for i in range(NumCols+1)] for i in range(NumCromo)]
MatElite = [[0 for i in range(NumCols+1)] for i in range(NumCromo*2)]

PrimeraGen(NumCromo)


for i in range(0,NumGeneracion): 
       Torneo()
       
       Reproduccion()
       ChangeBit()
       #Elitismo()
       CopyMat()
       GetChamp()
       print("Generacion=",i," Error=", ErrorChampGen, " Min Error=",ErrorChampGlobal )
       Graficar(saborResu)

Graficar(saborChamp)
print("Grafica Campeona Min Error = ",ErrorChampGlobal)




