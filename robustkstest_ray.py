#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from scipy import interpolate
import time
import ray
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


from utils import batch_clip_samples,create_ecdf,create_interpecdf


# In[3]:

#function to compute the trimmed KS distance
ray.shutdown()
ray.init()
@ray.remote
def calc_trimdk(s):
    s2=s[0]
    k=s[1]
    alfa=alpha[k]
    print("bootstrap iter, model iter, alpha=",s1,s2,k)
    
    #sorted samples1
    Rndavg=np.sort(Rnd2)
    
    #sorted samples2
    Rndindv=np.sort(Rnd1[s2])
    
    #pooled samples
    Rnd=np.concatenate((Rndavg,Rndindv))
    
    #ecdf of samples2
    ecdf=ECDF(Rndindv)
    Rnd=np.sort(Rnd)
    Fn=np.array([ecdf(j) for j in Rnd])
    
    #ecdf of samples1
    ecdf=ECDF(Rndavg)
    ecdf_F0=np.array([ecdf(j) for j in Rndavg])
    interp_F0=np.zeros_like(Rnd)
    
    #computing linearly interpolated ecdf of samples1
    f2= interpolate.interp1d(Rndavg,ecdf_F0)
    for i in range(len(Rnd)):
        if Rnd[i] < Rndavg[0]:
            interp_F0[i]=0
        elif Rnd[i] > Rndavg[-1]:
            interp_F0[i]=1
        else:
            interp_F0[i]=f2(Rnd[i])
            
    #linearly interpolated ecdf of samples1
    F0=interp_F0
    
    #computing the robust KS distance
    
    Fn_temp=np.insert(Fn,0,0)
    sample1=F0
    n=len(F0)
    x_u= np.array([(sample1[i] - (Fn_temp[i]/(1 - alfa))) for i in range(0,n)])
    x_l = np.array([(sample1[i] - (Fn[i]/(1 - alfa))) for i in range(0,n)])
    h_up = [max(x_u[i:n]) for i in range(0,n)]
    h_lo = [min(x_l[0:i+1]) for i in range(0,n)]
    h_up.append(-alfa/(1-alfa))
    h_lo.insert(0,0)
    h_upnew= [ h_up[i] + Fn_temp[i]/(1-alfa) for i in range(0,n+1)]
    h_lonew= [ h_lo[i] + Fn_temp[i]/(1-alfa) for i in range(0,n+1)]
    h_upnew=np.array(h_upnew)
    h_lonew=np.array(h_lonew)
    h_up=np.array(h_up)
    h_lo=np.array(h_lo)
    h_opt1 = (h_up + h_lo)/2
    h_opt= [min(h_opt1[i],0) for i in range(0,n+1)]
    h_opt= [max(h_opt[i],-alfa/(1-alfa)) for i in range(0,n+1)]
    h_a= [ h_opt[i] + Fn_temp[i]/(1-alfa)  for i in range(0,n+1)]
    h_opt=np.array(h_opt)
    h_a=np.array(h_a)
    #robust KS distance
    trim_dK=max(max((x_u-h_opt[0:n])),
                max((h_opt[1:n+1]-x_l)))
    return [trim_dK,s2,k]


# In[5]:

#load logits from 2*M models
with open('CIFAR10_2_logitstest_batchshuffleinit1.pkl', 'rb') as f:
    mod_stats1= pickle.load(f)
with open('CIFAR10_2_logitstest_batchshuffleinit2.pkl', 'rb') as f:
    mod_stats2= pickle.load(f)


# In[6]:


mod_stats=np.concatenate((mod_stats1,mod_stats2),axis=0)
print(mod_stats.shape)


# In[7]:


ep=49
logits=mod_stats[:,:,:,ep]
logitgap=logits[:,:,1]-logits[:,:,0]
#clip logit gaps to fix support
clipped_logitgap,Rnd_min,Rnd_max=batch_clip_samples(logitgap)


# In[ ]:


alpha=np.array([0.00,0.010,0.025,0.050,
                0.075,0.10,0.150,
                0.20,0.25,0.30,0.35,0.40,0.45])

bootstrap_size=100

#Samples from 2*M models
samples=clipped_logitgap

#Divide models for creating G' and Gc
samples1=samples[:int(len(clipped_logitgap)/2)]
samples2=samples[int(len(clipped_logitgap)/2):]
trim_dk=np.ones((bootstrap_size,samples1.shape[0],len(alpha)))
samp_size=samples1.shape[1]
Rnd1=np.zeros((samples2.shape[0],int(samp_size/2)))
Rnd2=np.zeros((int(samp_size/2))) 


#Iterate over bootstrap iters
for s1 in range(bootstrap_size):
    trimmed_dist=[]
    x=np.random.choice(np.arange(0,samp_size),samp_size)

    #Create iid samples for G' and Gc
    x1=x[:int(len(x)/2)]
    x2=x[int(len(x)/2):]
    t=time.time()
    Rnd1=samples1[:,x1]
    Rnd_indv=samples2[:,x2]
    #val_range=np.average(Rnd_indv,axis=0)
    val_range=Rnd_indv.flatten()
    F_indv=np.zeros_like(Rnd_indv)
    #iterate over no. of models
    for s2 in range(samples1.shape[0]):
        samp=np.sort(Rnd_indv[s2,:])
        ecdf=ECDF(samp)
        F_indv[s2,:]=np.array([ecdf(j) for j in samp])
    F_average=np.average(F_indv,axis=0)
    #Inverse transform sampling to generate samples for reference function $\hat{G}$
    counts=collections.defaultdict(int)
    for i in range(len(x)):
        counts[choice(F_average,val_range)]+=1
    val=[]
    count=[]
    for key in counts.keys():
        val.append(key)
        count.append(counts[key])
    val.pop(0)
    count.pop(0)
    rand_samp=[]
    for i in range(len(val)):
        rand_samp.append(val[i]*np.ones(count[i]))
    samp_rnd=[]
    for lists in rand_samp:
        for l in lists:
            samp_rnd.append(l)
    Rnd2=np.random.choice(samp_rnd,int(samp_size/2))
    #iterate over different trimming levels
    for s2 in range(samples1.shape[0]):
      for k in range(len(alpha)):
        s=[s2,k]
        trimmed_dist.append(calc_trimdk.remote(s))
    all_trim=np.array(ray.get(trimmed_dist))
    i=0
    for s2 in range(samples1.shape[0]):
        for k in range(len(alpha)):
            trim_dk[s1,s2,k]=all_trim[i][0]
            i=i+1
    ray.shutdown()
    elapsed=t-time.time()
    print("Time elapsed for alpha",elapsed)

# In[23]:


with open("trimdk_batchshuffleinit.pkl","wb") as f:
    pickle.dump(trim_dk,f)


# In[24]:

#setting the threshold using two sample DKW inequality
n=len(Rnd2)
epsilon=0.01
delta_a=np.sqrt((1/n)*(np.log(2/epsilon)))
delta=delta_a+1/n
print(delta)


# In[25]:

#computing alpha_hat (trimming level) needed to not reject the null
rej=np.zeros((trim_dk.shape[1],trim_dk.shape[2]))
alpha_bootstrap=np.zeros((trim_dk.shape[0],trim_dk.shape[1]))
for q in range(trim_dk.shape[0]):
    for j in range(trim_dk.shape[1]):
        t=trim_dk[q,j,:]
        for s in range(trim_dk.shape[2]):
            if t[s]>delta:
                rej[j,s]=0
            else:
                rej[j,s]=1
        #alpha_hat=0.50 is to be interpreted as a trimming level >=0.50
        if (len(np.where(rej[j,:]==1)[0])==0):
            alpha_bootstrap[q,j]=0.50
        else:
            alpha_bootstrap[q,j]=alpha[np.where(rej[j,:]==1)[0][0]]
alpha_hat=np.average(alpha_bootstrap,axis=0)
print(alpha_hat)


# In[26]:


with open("alphahat_batchshuffleinit","wb") as f:
    pickle.dump(alpha_hat,f)



