import os, random, shutil
import numpy as np
import pandas as pd


pwd = os.getcwd()
data_path = pwd + '\\20_newsgroups'
clas = os.listdir(data_path)

os.mkdir(pwd+'\\partition')
for i in range(0,5):
    os.mkdir(pwd+'\\partition'+'\\'+str(i))
    for cla in clas:
        os.mkdir(pwd+'\\partition'+'\\'+str(i)+'\\'+cla)

for cla in clas:
    files = os.listdir(data_path+'\\'+cla)
    random.shuffle(files)
    p_list = [files[i::5] for i in range(5)]
    for j in range(0,5):
        for name in p_list[j]:
            shutil.move(data_path+'\\'+cla+'\\'+name, pwd+'\\partition\\'+str(j)+'\\'+cla+'\\')


def cal(validation,k,up,down):
    N = 0
    N_c =[]
    for cla in clas:
        filenumber = 0
        for i in range(0,5):
            if i == validation:
                continue
            files = os.listdir(pwd+'\\'+'partition'+'\\'+str(i)+'\\'+cla)
            filenumber=filenumber+len(files)
        N_c.append(filenumber)
        N = N + filenumber
    N_p = np.array(N_c)/N
    N_p = np.log(N_p)
    
    total_df=[]
    total=[]
    for cla in clas:
        l_1=[]
        for i in range(0,5):
            if i==validation:
                continue
            files = os.listdir(pwd+'\\'+'partition'+'\\'+str(i)+'\\'+cla) # 文件名
            for file in files:
                try:
                    f = open(pwd+'\\'+'partition'+'\\'+str(i)+'\\'+cla+'\\'+file,"r",encoding='utf8') 
                    line = f.read()
                    l = line.split()
                    total_df.append(l)
                    l_1.extend(l)
                except:
                    continue
        # print(cla)
        total.append(l_1)
    df_dic =  {}
    
    for file in total_df:
        for word in set(file):
            if word in df_dic:
                df_dic[word]+=1
            else:
                df_dic[word]=1
    
    dele=[]
    for key in list(df_dic.keys()):
        if df_dic[key]>up or df_dic[key]<down: # 选取阈值
            dele.append(key)
            df_dic.pop(key)
    b = len(df_dic)
    
    pdic=[]
    for cla in total:
        dic_1={}
        a=0
        for word in cla:
            if word in df_dic:
                a+=1
                if word in dic_1:
                    dic_1[word] += 1
                else:
                    dic_1[word] =1
        for key in dic_1:
            dic_1[key] = np.log((dic_1[key]+k)/(a+b*k))
        pdic.append(dic_1)
        
        
    for cla in clas:
        coun = np.zeros(20)
        files=os.listdir(pwd+'\\'+'partition'+'\\'+str(validation)+'\\'+cla)
        for file in files:
            di = {}
            try:
                f = open(pwd+'\\'+'partition'+'\\'+str(validation)+'\\'+cla +'\\'+file,"r",encoding='utf8')
                line = f.read()
            except:
                continue
            l = line.split()
            for word in l:
                if word in di:
                    di[word]+=1
                else:
                    di[word]=1
            reli = []
            for i in range(0,20):
                noex = np.log(k/(len(total[i])+b*k))
                re = N_p[i]
                for key in di:
                    if key in pdic[i]:
                        tem = di[key]*pdic[i][key]
                    else:
                        tem = di[key]*noex
                    re = re+tem
                reli.append(re)
            ind = reli.index(max(reli))
            coun[ind] = coun[ind]+1
        if cla == 'alt.atheism':
            m = coun
        else:
            m = np.vstack((m,coun))
            
    P_list = []
    R_list = []
    F_list = []
    acc = 0
    for i in range(0,20):
        p = m[i][i]/np.sum(m.T[i])
    # print(p)
        r = m[i][i]/np.sum(m[i])
        acc = acc+m[i][i]
        f = 2*p*r/(p+r)
        P_list.append(p)
        R_list.append(r)
        F_list.append(f)
    p = np.mean(P_list)
    r = np.mean(R_list)
    f = np.mean(F_list)
    a = acc/m.sum()
#     print(p)
#     print(r)
#     print(f)
#     print(a)
    return p,r,f,a


def val(k,up,down):
    pl = []
    rl = []
    fl = []
    al = []
    for v in range(0,5):
        p,r,f,a = cal(v,k,up,down)
        pl.append(p)
        rl.append(r)
        fl.append(f)
        al.append(a)
    return [np.mean(pl),np.mean(rl),np.mean(fl),np.mean(al)]


rep = val(1,1000,0)
print(rep)