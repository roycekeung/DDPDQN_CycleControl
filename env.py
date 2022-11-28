    
from config import *
from Utils import Genid, TimingplanStruct, ActionsContainer
from overloading import overload

from xml.dom import minidom
import pandas as pd
import numpy as np
import math

### DISCO lib
# import PyModule_AdaptiveLogic as ALogic
import DISCO2_PyModule_MongKok

__all__ = ["Environment"]




class Environment(object):
    __instance = None
    __Actionobjs = None
    __basesigsetids=[]
    __actionids= []
    __holder = None
    initDmdId = 0
    __prev_DmdId = 0
    __OngoingSimTime = 0
    
    DmdLoaders = []
    
    def __init__(self, holder, Actions, basesigsetids, actionids, initDmdId, DmdLoaders, state_interval=5, maxSimTime=3600):
        '''
        either vissim, SUMO or CTM
        '''
        if self.__instance is None:
            Environment.__instance = self
            if Actions is None:
                self.__Actionobjs = ActionsContainer(Container=list())
            else:
                self.__Actionobjs = Actions
            #if isinstance(DISCO2_PyModule_MongKok.Holder(),holder):
            self.__holder = holder
            self.__basesigsetids = basesigsetids
            self.__actionids = actionids
            self.initDmdId = initDmdId
            self.DmdLoaders =DmdLoaders
            self.state_interval = state_interval
            self.maxSimTime = maxSimTime

        else:
            raise Exception("cannot instantiate a Environment again")
    
    ### --- --- --- --- getter --- --- --- --- 
    @staticmethod
    def get_instance():
        if Environment.__instance is None:
            Env = Environment()

        return Env.__instance
        
    def get_basesigsetids(self):
        return self.__basesigsetids
    
    def get_actionids(self):
        return self.__actionids
        
    ### --- --- --- --- setter --- --- --- --- 
    def reset(self, agentid, action_sigid):
        '''
        feature_shape = (whatever x size , whatever y size, 1 or 2(pos and speed ) or x 4  coz frames of history)
        state value is a two-value vector < position, speed >
        of the inside vehicle. The position dimension is a binary value,
        which denotes whether there is a vehicle in the grid. If there is
        a vehicle in a grid, the value in the grid is 1; otherwise, it is
        0. The value in the speed dimension is an integer, denoting the
        vehicle’s current speed in m/s
        '''
        ### run the base plan fisrt in order to init the initial state
        self.__prev_DmdId = 0
        self.__OngoingSimTime = 0
        
        # self.__prev_DmdId = self.__holder.getFinalCellRecDmdSetId()    ### getFinalCellRecDmdSetId return -1 ????
        
        action = self.__Actionobjs.get_Action(agentid= agentid, sigid=action_sigid)
    
        self.__holder.setUseCellPreLoad(useCellPreLoad=True)
        self.__holder.setStoreFinalCellRec(True)
        endTime = action.cycletime
        self.__holder.setStartEndTime(1, 0, endTime)  ## recType, startTime, endTime
        
        ### need to change the demand input rate after simulation preloading 
        ### eg. within whole 3hr sim require 30min preloading and every sim preload should consistant with the same preloading rate afterwards could be vary
        ## preload
        for DmdLoader in self.DmdLoaders[0]:
            self.__holder.clearAllDemandIntervals(0, dmdCellId=DmdLoader.CellId)
            self.__holder.addDemandInterval(dmdSet=0, dmdCellId=DmdLoader.CellId, startTime= DmdLoader.startTime, endTime= DmdLoader.endTime, newRate= DmdLoader.rate)

        self.__holder.clearPreLoad(dmdSetId=0)
        avgdelay = self.__holder.runSim(sigSet=action_sigid, dmdSet=self.initDmdId)
        self.__OngoingSimTime+=action.cycletime
        
        ### gen state
        self.__holder.printSimRecAll('tmpSimRecAll.xml')    ### printSimRecAll only have 0 timestep ????
        df = pd.read_csv(r'tmpSimRecAll.xml', skiprows=6, sep=",", header=None, encoding='utf-8')
        df.columns = ["Time","Cell","Occ","y_new","y_in","y_out","delay"]
        df1 = pd.DataFrame()
        for cellid in action.featcellids:
            df2= pd.DataFrame()
            for timestep in range(0,action.cycletime,self.state_interval):
                ### sec change to miliseconds *1000
                df3= df.loc[(df['Time']==timestep*1000) & (df['Cell']==cellid),['Occ','y_out']]
                df2 = pd.concat([df2, df3], ignore_index=True)
                
            df1 = pd.concat([df1, df2], axis=1)
            
        # df1=df1.T 
        state = np.array(df1)
        
        # self.__holder.importPreLoadFromDmdSet(occFromDmdSet=self.initDmdId, toDmdSet=0)

        
        return state
        
    
    def step(self, agentid, action_sigid):
        is_done = False

        

        ## action
        '''
        possible def
        1. possible actions, action space is defined by how to update the duration of every phase in the next cycle  
           could be minus or plus certain sec on a phase (adaptive); phse sequence kept the same 
           <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. number of phase, account for yellow, red and min green; but sequence of actions vary (phase)
        selected action means that of phase will be extend by certaian time step
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Control>>
        3. num actions = 2; 1= change phase ; 0= no change
            <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>
        '''

        
        ## state
        '''
        possible def of state
        1. matrix vehicles position ( + speed)  <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. num of waiting veh or waiting que length  <<Design of reinforcement learning parameters for seamless application of adaptive traffic signal contro>>
        3. Arrival of vehicles to the current green direction and Queue Length at red directions 
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        4. the maximum queue length associated with each phase
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        5. cumulative delay for phase i is the summation of the cumulative delay of all the vehicles that are travelling on the L(i).
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        6. distance between curent position of vehicle to intersection
        <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>
        7. last traffic signal in a lane (like 1 green, 0 red(no amber red coz action space cant represent it as well)
        <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>
        8. matrix for traffic states in a single intersection, dont care about actual geometry
          lanes r placed horizontally and stack up together into a state input matrix
          <<Cooperative Control for Multi-Intersection Traffic Signal Based on Deep Reinforcement Learning and Imitation Learning>>
        '''
        

        
        ## reward 
        ## in <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        ## reward = sum of waiting time of i th veh at t th cycle - sum of waiting time of i th veh at t+1 th cycle
        
        '''
        possible def of reward
        1. cumulative waiting time differece between two cycle (or t+1 , t)    
           <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. difference between the total cumulative delays of two consecutive actions (current state vs previous state) (or t+1 , t)   
           <<Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning>>
        3. queue length differece between two cycle (or t+1 , t)
        4. sum num of passing veh in outgoing lanes / sum num of stopped veh in incoming lanes   
           <<Traffic Signal Optimization for Multiple Intersections Based on Reinforcement Learning>
        '''
        
        ## is_done True = 1 ; False =0
        '''
        possible def of is_done
        1. sim ends
        '''
        

        action = self.__Actionobjs.get_Action(agentid= agentid, sigid=action_sigid)
    
        self.__holder.setUseCellPreLoad(useCellPreLoad=True)
        self.__holder.setStoreFinalCellRec(True)
        endTime = action.cycletime
        self.__holder.setStartEndTime(1, 0, endTime)    ## recType, startTime, endTime
        
        ### need to change the demand input rate after simulation preloading 
        ### eg. within whole 3hr sim require 30min preloading and every sim preload should consistant with the same preloading rate afterwards could be vary
        count = math.floor(self.__OngoingSimTime/ action.cycletime)

        if count < len(self.DmdLoaders):
            ## preload
            for DmdLoader in self.DmdLoaders[count]:
                self.__holder.clearAllDemandIntervals(dmdSet=0, dmdCellId=DmdLoader.CellId)
                self.__holder.addDemandInterval(dmdSet=0, dmdCellId=DmdLoader.CellId, startTime= DmdLoader.startTime, endTime= DmdLoader.endTime, newRate= DmdLoader.rate)
        else:
            ## random DmdRate
            newRates = np.random.uniform(0.008,0.9,len(self.DmdLoaders[0]))
            for index, DmdLoader in enumerate(self.DmdLoaders[0]):
                self.__holder.clearAllDemandIntervals(dmdSet=0, dmdCellId=DmdLoader.CellId)
                self.__holder.addDemandInterval(dmdSet=0, dmdCellId=DmdLoader.CellId, startTime= DmdLoader.startTime, endTime= DmdLoader.endTime, newRate= newRates[index])
        ## for checking DemandSet rate inside
        ## self.__holder.saveDemandSet(0, 'saveDemandSet.xml')
        avgdelay = self.__holder.runSim(sigSet=action_sigid, dmdSet=0)
        
        ### gen state
        self.__holder.printSimRecAll('tmpSimRecAll.xml')
        df = pd.read_csv(r'tmpSimRecAll.xml', skiprows=6, sep=",", header=None, encoding='utf-8')
        df.columns = ["Time","Cell","Occ","y_new","y_in","y_out","delay"]
        df1 = pd.DataFrame()
        for cellid in action.featcellids:
            df2= pd.DataFrame()
            for timestep in range(0,action.cycletime,self.state_interval):
                ### sec change to miliseconds *1000
                df3= df.loc[(df['Time']==timestep*1000) & (df['Cell']==cellid),['Occ','y_out']]
                df2 = pd.concat([df2, df3], ignore_index=True)
                
            df1 = pd.concat([df1, df2], axis=1)
            
        # df1=df1.T 
        next_state = np.array(df1)
        
        self.__prev_DmdId = self.__holder.getFinalCellRecDmdSetId()
        ### occupancy
        self.__holder.importPreLoadFromDmdSet(occFromDmdSet=self.__prev_DmdId, toDmdSet=0)
        
        #################################### find reward  ##################################
        reward = 1/avgdelay
        
        
        
        self.__OngoingSimTime+=action.cycletime
        
        if (self.__OngoingSimTime == self.maxSimTime):
            is_done = True
        
        return next_state, reward, is_done
        
    
    # @overload
    # def addin_action(self, obj)->None:
    #     if isinstance(TimingplanStruct, obj):
    #         self.__Actionobjs.Container.append(obj)
        
    # @overload
    # def addin_action(self, GenSigidObj, agentid:int, sigtiming:list, StageName:list)->None:
    #     self.__Actionobjs.Container.append(TimingplanStruct(GenSigidObj=GenSigidObj, agentid=agentid, sigtiming=sigtiming, StageName=StageName))
        
    # def addin__Actionobjs(self, *args)->None:
    #     for arg in args:       
    #         if isinstance(TimingplanStruct, arg):
    #             self.__Actionobjs.Container.append(arg)

    
    
    
if __name__ == '__main__':
    ## test
    Env= Environment()
    print("Env: ", Env)
    
    