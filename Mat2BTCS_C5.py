#   The following is a program that simulates breakthrough curves for
#   adsorbent beds based on physicochemical properties of an adsorbent
#   powder. The program was adopted based on the methodology from the
#   following reference:
#
#   R. Krishna, J.R. Long. "Screening Metal-Organic Frameworks by Analysis
#   of Transient Breakthrough of Gas Mixtures in a Fixed Bed Adsorber." J.
#   Phys. Chem. C, 2011, 115, 12941-12950.
#
#   D.M. Ruthven. "Principles of Adsorption and Adsorption Processes."
#   Chapter 8.
#
#=========================================================================#
#   Author: Joshua A. Thompson
#   Date Created: Jan. 28, 2013
#   Last Date Modified: Jan. 28, 2013
#   e-mail: thompson.josh.a@gmail.com
#=========================================================================#

#This model is based on the following assumptions: (1) the adsorption   
#process is isothermal; (2) no chemical reaction occurs in the column; 
#(3) the packing material is made of porous particles that are spherical
#and uniform in size; (4) the bed is homogenous and the concentration  
#gradient in the radial direction of the bed is negligible; (5) the flow 
#rate is constant and invariant with the column position ( Warchoł
#and Petrus, 2006); and, (6) the activity coefficient of each species is
# unity. 
#
#The original code of the model was written in Matlab, and can run only
#for two-component system. This code in python was extended from the Matlab
#code,and can run only for any multi-component system.
#The author of the python code: Xuanjun Wu at WHUT  
#Last Date Modified: Oct. 15, 2021
#e-mail: wuxuanjun@whut.edu.cn
#=========================================================================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyiast
from tqdm import trange, tqdm
import sys


'''
#=========================================================================#
#   Isotherm Parameters
qmax = np.array([13.1690,7.2494])  #mol/kg
#Saturation loading of each component
components = len(qmax) 
#Number of components
bi = np.array([0.005416,0.000826])  #kPa^-1
'''

outfile = "MOF4641-2.csv"


#Select the Isotherm Files
isotherm1 = 'mof-4641_CH4.csv' #CH4 on ZIF-8 @ 303K http://dx.doi.org/10.1002/chem.200902144
isotherm2 = 'mof-4641_CO2.csv' #CO2 on ZIF-8 @ 303K http://dx.doi.org/10.1002/chem.200902144
isotherm3 = 'mof-4641_N2.csv' #N2  on ZIF-8 @ 303K http://dx.doi.org/10.1002/chem.200902144
isotherm4 = 'mof-4641_CO.csv' 
isotherm5 = 'mof-4641_H2.csv' 

#Assemble the isotherm filenames into a list
isotherm_files = [ isotherm1, isotherm2, isotherm3, isotherm4 , isotherm5  ]
df_data = []
adsorbates = ['CH4','CO2','N2','CO','H2']
adsorbent = 'mof-4641'
ncomp = len(adsorbates)

#Specify the Isotherm Fitting Models (Langmuir,)
model1 = "Langmuir"
model2 = "Langmuir"
model3 = "Langmuir"
model4 = "Langmuir"
model5 = "Langmuir"
models = [model1, model2, model3, model4, model5]

#Instantiate the pure-component isotherms
for isotherm in isotherm_files:
    df_data.append(pd.read_csv(isotherm))
    
isotherms = [ pyiast.ModelIsotherm(df, loading_key="Loading (mmol/g)", pressure_key="Pressure (bar)", model=mdl) for df,mdl in zip(df_data,models) ]
for k in range(len(models)):
    pyiast.plot_isotherm(isotherms[k],withfit=True,xlogscale=False)
num_species = len(isotherms)

qmax = np.array([])
bi = np.array([])
for i in range(num_species):
    lang_para = isotherms[i].params  # dictionary of identified parameters
    qmax =np.append(qmax,lang_para['M'])
    bi = np.append(bi,lang_para['K']/100)

#qmax = np.array([13.1690,7.2494])
#bi = np.array([0.005416,0.000826])
print(bi,qmax)

#adsorbates = ['CH4','N2']
rho = 0.546743 #kg/L, Framework density 0.450
#=========================================================================#
#Thermodynamic data
Temp = 298 # Input temperature in degrees Celsius, K
R = 8.314  # Ideal gas constant, kPa*L/mol/K
p0 = 100 # Starting total pressure, kPa  
pt = p0 # Starting total pressure, kPa
                                                                                                      
y0 = np.array([0.06,0.15,0.01,0.03,0.75])  #Component 1 and Component 2
#y0 = np.array([0.5,0.5])  #Component 1 and Component 2
eps = 0.4 #Bed voidage 
u0 = 0.40 #Velocity, m/sec 

#========================================================================# 

#Discretization and initialization 

delz = 0.005 #Discretized bed length, bed is normalized to 0-1 
zmax = 0.5 #Dimensionless bed length 
m = int(zmax/delz) #Number of points 
z = np.linspace(0,zmax,num=m+1) #Zeroed z matrix 
delt = 0.025 #Discretized time scale, time is normalized to velocity and bed 
tmax = 500 #Max breakthrough time, dimensionless time 
n = int(tmax/delt) #Number of points 
t = np.linspace(0,tmax,num=n+1) #Zeroed t matrix
 
pres = np.zeros((ncomp,n+1,m+1)) # Pressure Matrix for pressure of each component
py = np.zeros((ncomp,n+1,m+1))  #Matrix for partial pressure of each component
pstore = np.zeros((ncomp,m+1)) #Storage matrix for pressure of each component
ustore = np.zeros((1,m+1)) #Storage matrix for velocity 
y = np.zeros((ncomp,n+1,m+1))  #Matrix for mole fraction of each component
u = np.zeros((n+1,m+1))  #Matrix for superficial velocity 
q = np.zeros((ncomp,n+1,m+1))  #Matrix for adsorbed gas of each component
dqdy = np.zeros((ncomp,ncomp))
ynew = np.zeros(ncomp) 
tbreak = np.zeros(ncomp)
tstoich = np.zeros(ncomp)
LUB = np.zeros(ncomp)

alpha1 = (1-eps)/eps*rho*R*Temp/p0

#Initial conditions
for i in range(m+1): 
    for k in range(ncomp):
        pres[k,0,i] = 0 
        py[k,0,i] = 0 
        y[k,0,i] = 0 
    u[0,i] = 0 

for j in range(1,n+1): #Boundary conditions 
    for k in range(ncomp):
        pres[k,j,0] = y0[k]*p0 
        py[k,j,0] = y0[k]*p0 
        y[k,j,0] = y0[k] 
    u[j,0] = u0 



#=========================================================================# 

#PDE Solver 
for i in trange(n): 
    for j  in range(1,m+1): 
        p0 = pt 
        sumbp = 1
        for k in range(ncomp):
            sumbp = sumbp + bi[k]*p0*y[k,i,j]
        deno_sq =  sumbp**2   
        for k in range(ncomp):
            for h in range(ncomp):
                if (h==k):
                    dqdy[k,h] = qmax[k]*bi[k]*p0*(sumbp-bi[k]*p0*y[k,i,j])/deno_sq
                else:
                    dqdy[k,h] = -qmax[k]*bi[k]*p0*bi[h]*p0*y[k,i,j]/deno_sq

        sumdqdy = np.sum(dqdy, axis=1)
        
        #dq1dy = -1.1892*y1[i,j]^5+4.373*y1[i,j]^4+7.2572*y1[i,j]^3+7.6065*y1[i,j]^2-6.04*y1[i,j]+3.6737 #IAST 
        #dq2dy = 0.5556*y1[i,j]^5-2.1645*y1[i,j]^4+3.7656*y1[i,j]^34.05*y1[i,j]^2+3.2202*y1[i,j]-1.9423 #IAST
        for k in range(ncomp):       
            func1 = 1+alpha1*(dqdy[k,k]-y[k,i,j]*sumdqdy[k])
            func2 = alpha1*sumdqdy[k]*(-1)
            if (k==0):
                u11 = u[i,j-1] - delz*func2*u[i,j-1]/func1*(y[k,i,j]-y[k,i,j-1])/delz
            else:
                u11 = u11            
            ynew[k] = y[k,i,j]-delt*u11/func1*(y[k,i,j]-y[k,i,j-1])/delz 
        
        yt = np.sum(ynew)
        #print(yt)
            
        if u11 == 0:
            p0 = 0 
        else: 
            p0 = pt 
        #update variables
        for k in range(ncomp): 
            y[k,i+1,j] = ynew[k]
            py[k,i+1,j] = ynew[k]*p0
        u[i,j] = u11 
        #dy2dy = (y2[i+1,j]-y2[i,j])/(y1[i+1,j]-y1[i,j])
        #dy3dy = -1-dy2dy
        if j == m:
           for k in range(ncomp):        
               if ynew[k] <= 0.02*y0[k]:
                   tbreak[k] = t[i+1]
                   pstore[k] = py[k,i+1,:] 
                   ustore = u[i,:] 
               if ynew[k]/y0[k] <= 0.5: 
                   tstoich[k] = t[i+1]
    #End of X-Loop
    #break
#End of Time-Loop
print(' *** DONE')
LUB = (1-tbreak/tstoich) 
print(tbreak,tstoich)
print(LUB)

txt = 'Outlet dimensionless concentration'
dict_df = {'Dimensionless time':t}
for k in range(ncomp): 
    labeltxt = txt+ ' '+str(k+1)
    dict_df[labeltxt]=y[k,:,-1]
#dict_df = {'Dimensionless time':t,labeltxt+ ' 1':y1[:,-1],labeltxt+ ' 2':y2[:,-1]}
df_results = pd.DataFrame(dict_df)
df_results.to_csv(outfile,index=False)

#Plot the outlet concentrations as a function of time
fig = plt.figure()
for k in range(ncomp): 
    plt.plot(t[:],y[k,:,-1] ,label=adsorbates[k], markevery=1)  #/y0[k]

#x1,x2,y1,y2 = plt.axis()
#plt.axis((0,10,0,1.5))
plt.xlabel('Dimensionless time, $\\tau = tu/\epsilon$L')
plt.ylabel('Outlet mole fraction, ${y_{i,out}}$')
plt.legend()
plt.show()
