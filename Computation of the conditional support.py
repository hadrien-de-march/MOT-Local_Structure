# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

p=2.1
d=8
steps = 10000.
RatioSearchRootsInf = 5.
RatioSearchRootsSup = 1.2
Max = 5.
Min = -10.

#print("log(exp(1))",np.log(np.exp(1.)))

withY = 1
PrintDots = 1
Givey = 0

def mod(x):
    return np.sqrt(np.sum(x*x))
    
def modSqu(x):
    return np.sum(x*x)

def Cx(x):
    return p*mod(x)**(p-1)*x/mod(x)
    
ybrut=np.random.normal(0,1,(d+1,d))

lbdabrut =np.random.rand(d+1)


lbda = lbdabrut/np.sum(lbdabrut)



ymoy = np.dot(lbda,ybrut)

y = np.copy(ybrut)

for i in range(d+1):
    y[i] = y[i]-ymoy

print(y)

print(lbda)
    
print(np.dot(lbda,y))

#print(mod(Cx(np.array([5,-3,-1])))-mod(p)*mod(np.array([5,-3,-1]))**(p-1))



Cxy = list(map(Cx,y))

DeltaCxy = np.eye(d)

#print(range(d))

for i in range(d):
    DeltaCxy[i] = Cxy[i]-Cxy[d]
    
Deltay = np.eye(d)

for i in range(d):
    Deltay[i] = y[i]-y[d]

H = np.dot(np.transpose(DeltaCxy),np.transpose(np.linalg.inv(Deltay)))

v = Cxy[0]-np.dot(H,y[0])


print(Cxy[d]-np.dot(H,y[d])-v)




def Chi(x):
    if p>=2:
        return (((mod(x/p))**(2/(p-2))
                -mod(np.dot(np.linalg.inv(x*np.eye(d)-H),v))**2)
                *np.linalg.det(x*np.eye(d)-H)**2
                )
    else:
        return ((1-(mod(x/p))**(2/(2-p))
                 *mod(np.dot(np.linalg.inv(x*np.eye(d)-H),v))**2)
                 *np.linalg.det(x*np.eye(d)-H)**2
                 )

def F(x):
    sizes1 = np.array(list(map(mod,y)))
    sizes = list(map(lambda y:mod(p)*y**(p-2),sizes1))
    Denominator = lbda/(sizes-np.double(x))
    Numerator = Denominator*sizes
    return np.sum(Numerator)/np.sum(Denominator)
    
def Norm(x):
    sizes1 = np.array(list(map(mod,y)))
    sizes = list(map(lambda y:mod(p)*y**(p-2),sizes1))
    i = np.argmin(sizes)
    vect = lbda*(sizes-F(x))/(sizes-np.double(x))
    yRed = y-y[i]
    return(modSqu(np.dot(vect,yRed)))
        
    
def NewChi(x):
    return np.log(Norm(x))-np.log(x/mod(p))*2/(p-2)
    
def ChiMin(x):
    return np.sign(NewChi(x))*min(np.abs(NewChi(x)),np.abs(Chi(x)))

def Inv(x):
    sizes1 = np.array(list(map(mod,y)))
    sizes = list(map(lambda y:mod(p)*y**(p-2),sizes1))
    vect = lbda*(sizes-F(x))/(sizes-np.double(x))
    return(vect)
    
def Pow(x):
    return (x/mod(p))**(2/(p-2))

print("Coeff dominant",Chi(1000)/1000**(2.*d))
print("1-norme de v^2", (1-mod(v)**2))


lambdas = np.sort(list(map(lambda x:mod(p)*mod(x)**(p-2),y)))
print("F(lambdas)-lambdas",list(map(F,0.00001+lambdas))-lambdas)

lambdasRed = np.sort(list(map(lambda x: mod(x)**2,y)))
print("Norm(lambdas)-lambdasRed",np.sort(list(map(Norm,0.00001+lambdas)))-lambdasRed)
print("lambdas nuls = ",list(map(Chi,lambdas)))

EigenH = np.sort(np.linalg.eigvals(H))

print("eigen values of H = ", EigenH)

Ratio = EigenH[d-1]/EigenH[0]
xmin = EigenH[0]/(1.+(Ratio-1.)*RatioSearchRootsInf)
xmax = EigenH[d-1]*(1.+(Ratio-1.)*RatioSearchRootsSup)
size = np.log(xmax/xmin)/2
prec = size/steps

print("lambdas = ",lambdas)

print("Norm(xmax) = ",Norm(xmax))

print("Pow(xmax) = ",Pow(xmax))

Abs = list(map(np.exp,np.log(xmin)+prec*(1+np.array(range(int(2*size/prec))))))

def Born(x):
    return min(max(x,Min),Max)


#fig = plt.figure(0)
#OrdF = map(F,Abs)
#OrdFBorn = map(Born,OrdF)
#plt.plot(Abs,OrdFBorn)
#plt.plot(Abs,Abs)



#OrdNorm = map(Norm,Abs)
#OrdNormBorn = map(Born,OrdNorm)
#fig = plt.figure(1)
#plt.plot(Abs,OrdNormBorn)
#plt.plot(Abs,map(Born, (map(Pow, Abs))))


OrdLogNorm = list(map(np.log,list(map(Norm,Abs))))
LogAbs = list(map(np.log,Abs))
logP = np.log(p)
OrdLogPow = list(map(lambda x:2/(p-2)*(x-logP),LogAbs))
fig = plt.figure(5)
plt.plot(LogAbs,list(map(Born,OrdLogNorm)))
plt.plot(LogAbs,list(map(Born, OrdLogPow)))

#fig = plt.figure(6)
#plt.plot(Abs,map(Born,OrdLogNorm))
#plt.plot(Abs,map(Born, OrdLogPow))

print("first done")


#OrdChi = map(Chi,Abs)
#OrdChi = map(NewChi,Abs)
OrdChi = np.array(OrdLogNorm)-np.array(OrdLogPow)

BornOrd = list(map(Born,OrdChi))

#plt.plot(Abs,BornOrd)

Rac = np.zeros(int(2*size/prec))
nRac = 0
SgChi = np.sign(OrdChi[0])

for i in range(int(2*size/prec)):
    if np.sign(OrdChi[i])!=SgChi:
        nRac=nRac+1
        Rac[nRac-1]=Abs[i]
        SgChi = np.sign(OrdChi[i])
        
print("eigen values of H = ",np.linalg.eigvals(H))
        
Roots = np.zeros(nRac)

for i in range(nRac):
    Roots[i] = Rac[i]


print("Roots1 = ", Roots)
print("nb Roots1 = ",np.size(Roots))
Roots = list(map(lambda x: float(scipy.optimize.newton_krylov(ChiMin, x, f_tol=1e-14, maxiter=20)),Roots))
print("lambdas = ",lambdas)
print("Roots2 = ",Roots)
print("nb Roots2 = ",np.size(Roots))


def yFromLambda(l,withY = 1):
    if withY:
        return np.dot(np.linalg.inv(l*np.eye(d)-H),v)
    else :
        return np.dot(np.transpose(np.linalg.inv(Deltay)),np.dot(np.linalg.inv(l*np.eye(d)-H),v))
    
T = list(map(lambda x: yFromLambda(x,withY),Roots))
if Givey:
    print("values of the points = ",T)

if withY:
    y = y
else:
    y = np.dot(y,np.linalg.inv(Deltay))

if PrintDots:
    if d==2:
        figdots = plt.figure(2)
        AbsRoots = list(map(lambda x:x[0],T))
        OrdRoots = list(map(lambda x:x[1],T))
        plt.plot(AbsRoots,OrdRoots, marker='o', linestyle='None', color='r')
        Absy = list(map(lambda x:x[0],y))
        Ordy = list(map(lambda x:x[1],y))
        plt.plot(Absy,Ordy, marker='o', linestyle='None', color='b')
        plt.plot([T[0][0]],[T[0][1]], marker='o', linestyle='None', color='y')
    
    if d==3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        
        AbsRoots = list(map(lambda x:x[0],T))
        OrdRoots = list(map(lambda x:x[1],T))
    
        Absy = list(map(lambda x:x[0],y))
        Ordy = list(map(lambda x:x[1],y))
        
        AltRoots = list(map(lambda x:x[2],T))
        ax.scatter(AbsRoots, OrdRoots, AltRoots, color='r')
        Alty = list(map(lambda x:x[2],y))
        ax.scatter(Absy, Ordy, Alty, color='b')
        ax.scatter([T[0][0]],[T[0][1]], [T[0][2]], color='k')