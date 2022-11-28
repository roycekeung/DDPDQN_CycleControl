

# from xml.dom import minidom

# # parse an xml file by name
# file = minidom.parse('MKtdsig2560.xml')

# #use getElementsByTagName() to get tag
# models = file.getElementsByTagName('PlanData')

# # one specific item attribute
# print('model #2 attribute:')
# print(models)
# print(models[0].attributes['CycleTime'].value)
# print(models[1].attributes['CycleTime'].value)
# print(models[2].attributes['CycleTime'].value)
# print(models[3].attributes['CycleTime'].value)
# print(1-True)

# import DISCO2_PyModule_MongKok
# print("succeed to import DISCO2_PyModule_MongKok")

# from dataclasses import dataclass, field
# import itertools

# from typing import List
# ### timing plan data structure
# @dataclass(frozen=True)
# class TimingplanStruct():
#     agentid:int = 0
#     sigid:int=-1
#     offset:int=0
#     cycletime:int=100

# action0 = TimingplanStruct(agentid=0)
# print("action0: ", action0)
# print("action0.cycletime: ", action0.cycletime, type(action0.cycletime))



# @dataclass(frozen=False)
# class ActionsContainer:
#     Container:List[TimingplanStruct]= field(init = True, default_factory=List, repr=False)
#     actions_num:int=0

# Actions = ActionsContainer(Container=list())
# Actions.Container.append(action0)
# print("Actions.Container[0].cycletime: ", Actions.Container[0].cycletime, type(Actions.Container[0].cycletime))

# import pandas as pd
# import numpy as np

# df = pd.read_csv(r'tmpSimRecAll.xml', skiprows=6, sep=",", header=None, encoding='utf-8')
# df.columns = ["Time","Cell","Occ","y_new","y_in","y_out","delay"]

# i=df[df["Cell"]==73]['Occ']
# print(i)


# featcellids= [73,9,10]
# df = pd.read_csv(r'tmpSimRecAll.xml', skiprows=6, sep=",", header=None, encoding='utf-8')
# df.columns = ["Time","Cell","Occ","y_new","y_in","y_out","delay"]
# df1 = pd.DataFrame()
# for cellid in featcellids:
#     df2= pd.DataFrame()
#     for timestep in range(0,2,1):

#         df3= df.loc[(df['Time']==timestep) & (df['Cell']==cellid),['Occ','y_out']]
#         df2 = pd.concat([df2, df3], ignore_index=True)
#         print(df2)
#     df1 = pd.concat([df1, df2], axis=1)
#     print('state:', df1)
# state=df1.T 
# state = np.array(state)


# print('final state:',state, type(state), state.shape)

# # print([timestep for timestep in range(0,120,5)])

# # ### merge horizontally
# # df_concat = pd.concat([df1, df2], axis=1)

# memory=[]
# ## [state, action, reward, next_state, is_done]
# memory.append([state, 0, 1])


# featcellids= [11,12,13]
# df1 = pd.DataFrame()
# for cellid in featcellids:
#     df2= pd.DataFrame()
#     for timestep in range(0,2,1):

#         df3= df.loc[(df['Time']==timestep) & (df['Cell']==cellid),['Occ','y_out']]
#         df2 = pd.concat([df2, df3], ignore_index=True)
#         print(df2)
#     df1 = pd.concat([df1, df2], axis=1)
#     print('state:', df1)
# state=df1.T 
# state = np.array(state)
# print('final state:',state, type(state), state.shape)

# memory.append([state, 2, 3])

# def sampling( batch_size= 32):
#     ## could select the same index multiple times
#     sample_index = np.random.choice(len(memory), size=batch_size , replace=True)
#     batch_memory = []
#     for x in sample_index:
#         batch_memory.append(memory[x])
        
#     states, actions, rewards= map(np.asarray, zip(*batch_memory))
#     # states = np.array(states).reshape(batch_size, -1)  ## row of each state_t
    
#     return states, actions, rewards


# states, actions, rewards = sampling(batch_size= 2)
    
# print('sampling state:',states, type(states), states.shape)
# print('sampling actions:',actions, type(actions), actions.shape)
# print('sampling rewards:',rewards, type(rewards), rewards.shape)

# actions_value = np.array([[1,2,3,3,4,5]]).ravel()
# print(actions_value)
# action = np.argmax(actions_value)
# print(action)



# a = np.array([[0.],
#  [0.],
#  [0.],
#  [0.],
#  [0.],
#  [0.],
#  [0.],
#  [0.]])
# print('a:', a, type(a), a.shape)
# a = np.expand_dims(a, axis=0)
# # a.reshape(1,8,1)
# print('a:', a, type(a), a.shape)




import DISCO2_PyModule_MongKok
import pandas as pd

### --- --- --- setupHolder --- --- --- 
### before Sim of RL
### init C++ holder
__holder = DISCO2_PyModule_MongKok.Holder()
__holder.loadTdSigData(filePath="MKtdsig2560.xml")  ### <-- change here

### set cellids that input as feature map
targetcellids = [73, 9, 10, 11, 12, 13]   ### <-- change here

### thru switch sigsetid to make change in tdsigdata plan
### output sigsetid

sigsetid=0 ; dmdSetId=0 ;

__holder.loadScenario(path="MKNathanNet.xml")  ### <-- change here
dmdSetId = __holder.loadDemandSet(path="MKdemand.xml")  ### <-- change here


## init ActionsContainer and TimingplanStruct

# parse an xml file by name
from xml.dom import minidom
file = minidom.parse('MKtdsig2560.xml')

#use getElementsByTagName() to get tag
models = file.getElementsByTagName('PlanData')

Base_sigsetids = []
Actionids= []
# for nth_model , _ in enumerate(models):
    ### to holder DISCO
    ### PlanId="0" in mk; depends on how many timing plan u have
sigsetid = __holder.exportTdSigDataToDISCO(planId=0, sigSetId=-1)  ### <-- change here
Actionids.append(sigsetid)
print("sigsetid: ", sigsetid)
__holder.saveSignalSet(sigsetid, "D:\HKUST\\research_project\\me\\RL\\code\\DDPDQN_CycleControl\\tSig.xml")



# cycletime = int(models[nth_model].attributes['CycleTime'].value)
### tmp ignore green and red for now

# if (nth_model==0):
#     Base_sigsetids.append(sigsetid)    ## tmp default one agent for now








### run the base plan fisrt in order to init the initial state
__prev_DmdId = 0
__OngoingSimTime = 0

__prev_DmdId = __holder.getFinalCellRecDmdSetId()    ### getFinalCellRecDmdSetId return -1 ????
print("__prev_DmdId: ", __prev_DmdId)
__holder.setUseCellPreLoad(useCellPreLoad=True)
__holder.setStoreFinalCellRec(True)

__holder.setStartEndTime(1, 0, 120)  ## recType, startTime, endTime
avgdelay = __holder.runSim(sigSet=sigsetid, dmdSet=0)
print("avgdelay: ", avgdelay)
__OngoingSimTime+=120

### gen state
__holder.printSimRecAll('tmpSimRecAll.xml')    ### printSimRecAll only have 0 timestep ????
df = pd.read_csv(r'tmpSimRecAll.xml', skiprows=6, sep=",", header=None, encoding='utf-8')
df.columns = ["Time","Cell","Occ","y_new","y_in","y_out","delay"]
df1 = pd.DataFrame()
# for cellid in action.featcellids:
#     df2= pd.DataFrame()
#     for timestep in range(0,120,5):
#         df3= df.loc[(df['Time']==timestep) & (df['Cell']==cellid),['Occ','y_out']]
#         df2 = pd.concat([df2, df3], ignore_index=True)
        
#     df1 = pd.concat([df1, df2], axis=1)
    
# state=df1.T 
# state = np.array(state)

__prev_DmdId = __holder.getFinalCellRecDmdSetId()    ### getFinalCellRecDmdSetId return -1 ????
print("__prev_DmdId: ", __prev_DmdId)


__holder.importPreLoadFromDmdSet(occFromDmdSet=0, toDmdSet=0)



