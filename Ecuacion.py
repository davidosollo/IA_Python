# -*- coding: utf-8 -*-
"""
Spyder Editor

Programa: IA Ecuacion
Autor:    David Osollo

"""

#LIBSS
import random
import math
import matplotlib.pyplot as plt

def PrimeraGen(NumCromo_F):
    for i in range(0,NumCromo_F): 
        for j in range(0, NumCols): 
            MatPadres[i][j] = valInit #15#random.randint(1, MaxNum)
        CalculateError(MatPadres,i)
            
def CalculateError(Mat,Cromo):
    AcuError = 0;
    for i in range(1,NumPuntos):
        xE = i/10;
        yE = Mat[Cromo][0]*(Mat[Cromo][1]*math.sin(xE/Mat[Cromo][2]) + Mat[Cromo][3]*math.cos(xE/Mat[Cromo][4])) + Mat[Cromo][5]*xE - Mat[Cromo][3]
        AcuError = abs(yE-y[i]) + AcuError
        
    Mat[Cromo][6] = int(AcuError)
    
def Torneo():
    for i in range(0,NumCromo):
        ErrorChamp = 9000000
        CompeChamp = 0
        for j in range(0,NumCompe):
            Compe = random.randint(0, NumCromo-1)
            if ErrorChamp > MatPadres[Compe][NumCols]:
                CompeChamp = Compe
                ErrorChamp = MatPadres[Compe][NumCols]
                
                
        for k in range(0,NumCols+1):
            MatHijos[i][k] = MatPadres[CompeChamp][k]
            
def GetChamp():
    ErrorChamp = 9000000
    CompeChamp = 0
    
    for k in range(0,NumCromo):
        if ErrorChamp > MatPadres[k][NumCols]:
            CompeChamp = k
            ErrorChamp = MatPadres[k][NumCols]
            
    for i in range(0,NumPuntos):
        ChampX[i] = i/10
        ChampY[i] = MatPadres[CompeChamp][0]*(MatPadres[CompeChamp][1]*math.sin(ChampX[i]/MatPadres[CompeChamp][2]) + MatPadres[CompeChamp][3]*math.cos(ChampX[i]/MatPadres[CompeChamp][4])) + MatPadres[CompeChamp][5]*ChampX[i] - MatPadres[CompeChamp][3]
    
    if BestFit[NumCols] > ErrorChamp :
         for k in range(0,NumCols+1):
            BestFit[k] = MatPadres[CompeChamp][k]
        
    #print("CompeChamp=",CompeChamp)
    #print("ErrorChamp=",ErrorChamp)
    print("Best=", BestFit[NumCols])    
    

def CopyMat():
    for i in range(0,NumCromo): 
        for j in range(0, NumCols+1): 
            MatPadres[i][j] = MatHijos[i][j]
            

        
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
    

# Reproduccion            
def Reproduccion():
    CicloRep = NumCromo//2
    MatBaby = [[0 for i in range(NumCols+1)] for i in range(2)]
        
    for i in range(0,CicloRep,2):
        SegPos = random.randint(0,(10*NumCols)-2)
        byteNum = int(SegPos//10);
        NumCorr = SegPos - (byteNum * 10)
        
        NumPart = int(MatHijos[i][byteNum]);
        cromo_x = NumPart >> NumCorr;
        cromo_x = cromo_x << NumCorr;
        cromo_y = NumPart - cromo_x
        
        NumPart = int(MatHijos[i+1][byteNum]);
        cromo_xx = NumPart >> NumCorr;
        cromo_xx = cromo_xx << NumCorr;
        cromo_yy = NumPart - cromo_xx
        
#        Resu1 = 0
#        Resu2 = 0
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
        
        #Resu1 = Resu1 + MatBaby[0][j] * math.pow(x,(GradoEcu-j));
        #Resu2 = Resu2 + MatBaby[1][j] * math.pow(x,(GradoEcu-j));
            
    
        #MatBaby[0][NumCols]=abs(Resu1-ValorEcu)
        #MatBaby[1][NumCols]=abs(Resu2-ValorEcu)
        CalculateError(MatHijos,i)
        CalculateError(MatHijos,i+1)
        
def ChangeBit():
    for i in range(0,NumCromo): 
        BitSta = 0
        SegPos = random.randint(0,(10*NumCols))
        byteNum = int(SegPos//10);
        NumCorr = SegPos - (byteNum * 10)
        NumCheck = int(math.pow(2,NumCorr))
        VarMuta=MatHijos[i][byteNum]
        
        #if NumCheck > VarMuta :
        #    print("Error")
        
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


def Graficar():
    #plt.figure(1, figsize = [7, 7],clear=True)
    #plt.subplot(2, 1, 1)
    #plt.title("Ecuacion")
    #plt.xlabel("eje-x")
    #plt.ylabel("eje-y")
    #plt.plot(ejex,ejey, marker='o', linestyle='-', color='r', label="ruta")
    plt.subplot(2, 1, 2)
    plt.title("Ecuacion")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.plot(x,y, marker='o', linestyle='-', color='g', label="ruta")
    plt.plot(x,y)
    plt.plot(ChampX,ChampY)
    plt.show()
    plt.close()

     
    
def GenEcuacion():
    
    for i in range(0,NumPuntos):
        x[i] = i/10
        y[i] = A*(B*math.sin(x[i]/C) + D*math.cos(x[i]/E)) + F*x[i] -D
        
        

#Variables Globales
NumCromo = 100
NumGen = 100
NumCols = 6
NumPuntos = 1000
MaxNum = 90
NumCompe = 8
ChampCromo = 0
turnOff = 0
valInit = 15

A = 10
B = 25
C = 3
E = 10
F = 6
D = 70

#Main 

MatPadres = [[0 for i in range(NumCols+1)] for i in range(NumCromo)]
#MatElite = np.matrix =[[0 for i in range(NumCols+1)] for i in range(NumCromo*2)]
MatElite = [[0 for i in range(NumCols+1)] for i in range(NumCromo*2)]
x = [0 for i in range(NumPuntos)]
y = [0 for i in range(NumPuntos)]

ChampX = [0 for i in range(NumPuntos)]
ChampY = [0 for i in range(NumPuntos)]

MatHijos = [[0 for i in range(NumCols+1)] for i in range(NumCromo)]
BestFit = [0 for i in range(NumCols+1)]
BestFit[NumCols] = 99999999
GenEcuacion()
PrimeraGen(NumCromo)


for i in range(0,NumGen): 
   Torneo()
   Reproduccion()
   ChangeBit()


   #MatPadres = MatHijos.copy()
   #Elitismo()
   CopyMat()
   GetChamp()
   print("Generacion=",i)
   Graficar()

   #if  (i>(NumGen//2)) and ((i%10)==0)and (BestFit[NumCols] > 100000):
   #if  ((i>(NumGen//2)) and (i%25)==0)and (BestFit[NumCols] > 30000):
   #     PrimeraGen(95)

       #NumCompe = 



print("Mejor Combinacion:")
print(BestFit)


