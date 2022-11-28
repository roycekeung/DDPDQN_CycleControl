'''
based on
<<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
but it only handle single intersection => one agent one controller
have phase sequence constraint
''' 

# import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import argparse
import numpy as np
from collections import deque
import random
from xml.dom import minidom

import os
import time
import datetime
import math

# from modelAgent import *
from CustomizedModelAgent import *
from env import Environment
from DQNtype import AlgoType, ArchType
from config import *
from Utils import Genid, TimingplanStruct, ActionsContainer, DmdLoader

### DISCO lib
# import PyModule_AdaptiveLogic as ALogic
import DISCO2_PyModule_MongKok

## add later when upto the use of linux vm as server training
parser = argparse.ArgumentParser()
parser.add_argument('--output_graph', type=bool, default=False)
parser.add_argument('--max_episode', type=int, default=1000)
parser.add_argument('--display_reward_threshold', type=int, default=200)
parser.add_argument('--max_ep_steps', type=int, default=3600)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)  ## some 64
parser.add_argument('--replay_mem_size ', type=int, default=2000)
parser.add_argument('--epoch ', type=int, default=1000)
parser.add_argument('--randomseed  ', type=bool, default=True)
parser.add_argument('--save_model  ', type=bool, default=True)

parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)



GPU = "Nvidia"
CPU = "Intel(R) Core(TM)"
MACHINE = "Colab"
### set up following config when u have better GPU with CUDA driver to do training
# os.environ["CUDA_VISIBLE_DEVICES"] = 1
config_dict = {
    "GPU_USED" : 0,
    "GPU" : GPU,
    "CPU" : CPU,
    "MACHINE" : MACHINE,
}




tf.keras.backend.set_floatx('float64')
projectname = 'DDPDQN_Discrete_TrafficControl'
trialname = 'DDPDQN_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# ### upload training record to wandb server
# wandb.init(
#     name=trialname, 
#     project=projectname,
#     config = config_dict,
#     entity="Royce")

### for local tensorboard backup
log_dir = "./logs"
tensorboard = TensorBoard(log_dir=log_dir)



if __name__ == "__main__":
    ### --- --- --- setupHolder --- --- --- 
    ### before Sim of RL
    ### init C++ holder
    holder = DISCO2_PyModule_MongKok.Holder()
    holder.loadTdSigData(filePath="MKtdsig2560.xml")  ### <-- change here

    ### set cellids that input as feature map
    DetectorCellIds = [10, 30, 50, 70, 90, 110, 130, 150]   ### <-- change here
    
    ### thru switch sigsetid to make change in tdsigdata plan
    ### output sigsetid
    
    sigsetid=0 ; dmdSetId=0 ;
    
    holder.loadScenario(path="MKNathanNet.xml")  ### <-- change here
    dmdSetId = holder.loadDemandSet(path="MKdemand.xml")  ### <-- change here
    

    ## init ActionsContainer and TimingplanStruct
    Actions = ActionsContainer(Container=list())
    # parse an xml file by name
    file = minidom.parse('MKtdsig2560.xml')

    #use getElementsByTagName() to get tag
    models = file.getElementsByTagName('PlanData')
    GenSigidObj = Genid()
    Base_sigsetids = []
    Actionids= []
    cycletime=0
    for nth_model , _ in enumerate(models):
        ### to holder DISCO
        ### PlanId="0" in mk; depends on how many timing plan u have
        sigsetid = holder.exportTdSigDataToDISCO(planId=nth_model, sigSetId=-1)  ### <-- change here
        Actionids.append(sigsetid)
        
        cycletime = int(models[nth_model].attributes['CycleTime'].value)
        ### tmp ignore green and red for now
        action = TimingplanStruct(GenSigidObj= GenSigidObj, 
                                  agentid=0, 
                                  sigid=sigsetid ,
                                  cycletime= cycletime, 
                                  sigtiming=[0,50,50,100], 
                                  StageName=['gstart','gend','rstart','rend'],
                                  featcellids= DetectorCellIds)
        Actions.Container.append(action)
        
        if (nth_model==0):
            Base_sigsetids.append(sigsetid)    ## tmp default one agent for now
    


    
    cost_his = []
    if OUTPUT_GRAPH:
        # create the file writer object
        writer = tf.summary.create_file_writer(logdir= log_dir)
        
    ### Preload DmdCell rate
    dmdCellRates = [85, 86, 90]
    preLoadNumCount = int(30*60/cycletime)   ## 30min of preload and 120 cycletime
    DmdLoaders=[]
    for n in range(0,preLoadNumCount):
        tmpDmdLoaders = []
        if (n==0):
            tmpDmdLoaders.append(DmdLoader(startTime=0, endTime=cycletime, CellId=85, rate= 0.41666666666666669))
            tmpDmdLoaders.append(DmdLoader(startTime=0, endTime=cycletime, CellId=86, rate= 0.16666666666666666))
            tmpDmdLoaders.append(DmdLoader(startTime=0, endTime=cycletime, CellId=90, rate= 0.16666666666666666))

        else:
            for dmdCellID in dmdCellRates:
                newRate = np.random.uniform(0.008,0.9)
                tmpDmdLoaders.append(DmdLoader(startTime=0, endTime=cycletime, CellId=dmdCellID, rate= newRate))
        DmdLoaders.append(tmpDmdLoaders)

    ## init env
    state_interval = 5
    env = Environment(holder=holder, Actions= Actions, basesigsetids= Base_sigsetids, actionids=Actionids ,initDmdId=dmdSetId, DmdLoaders= DmdLoaders, state_interval = 5, maxSimTime= MAX_EP_STEPS*cycletime)
    

    feature_shape = (math.floor(cycletime/state_interval), len(DetectorCellIds)*2)  ### <-- change here
    agent = AgentController(agentid=0, env=env, feature_shape=feature_shape, n_actions=len(Actionids) , e_greedy=0.9, e_decay=0.995, e_min=0.01, replace_target_iter = 40, tensorboard= tensorboard, algo_type=AlgoType.Dueling, architecture = ArchType.Conv1d_dense , isbuild_net=True )
    agent.learn(max_ep= MAX_EPISODE, max_timestep=MAX_EP_STEPS, min_Lr_step=50, Lr_iter=5, algo_type=AlgoType.Double )
    eval_net = agent.get_eval_net_model()

        
    if SAVE_MODEL:
        ### get current file directory path
        folderPath = os.path.join(os.getcwd(), trialname+'.h5')
        ### save trained model
        eval_net.save_model(folderPath)

    
    # ### plot cost but if training in linux vm then ignore this
    if OUTPUT_GRAPH:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(agent.get_cost_his())), agent.get_cost_his())
        plt.ylabel('Cost')
        plt.xlabel('accumulative steps')
        plt.show()        
        plt.savefig('Cost vs accumulative steps.png')
