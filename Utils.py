from dataclasses import dataclass, field
import itertools

from typing import List

__all__ = ["Genid", "TimingplanStruct", "ActionsContainer"]

### id Counter
class Genid:
    id_iter = itertools.count()
    
    def __init__(self):
        self.id = next(Genid.id_iter)

    def getid(self):
        return self.id
    
    def nextid(self):
        self.id = next(Genid.id_iter)
        return self.id


### timing plan data structure
@dataclass(frozen=True)
class TimingplanStruct():
    GenSigidObj:Genid = None
    agentid:int = 0
    sigid:int=-1
    offset:int=0
    cycletime:int=100
    sigtiming: List[int] = field(init = True, default_factory=List, repr=False)
    StageName: List[str] = field(init = True, default_factory=List, repr=False)
    featcellids: List[int] = field(init = True, default_factory=List, repr=False)
    
    def __post_init__(self)->None:
        if self.GenSigidObj != None and self.sigid==-1:
            object.__setattr__(self, 'sigid', self.GenSigidObj.nextid())
        
    def __str__(self)->str:
        return f'\nagentid: {self.agentid} sigid: {self.sigid} \n{self.StageName} \n{self.sigtiming}'
    

@dataclass(frozen=False)
class ActionsContainer:
    Container:List[TimingplanStruct]= field(init = True, default_factory=List, repr=False)
    actions_num:int=0
     
    def __post_init__(self)->None: 
        self.actions_num = len(self.Container)

    ### --- --- --- --- getter --- --- --- --- 
    def get_Action(self, agentid, sigid)->TimingplanStruct:
        action = None
        for n in self.Container:
            if n.agentid == agentid and n.sigid == sigid:
                action = n
                break

        return action
    
    ### --- --- --- --- setter --- --- --- --- 
    def set_(self):
        pass
    
    def __str__(self)->str:
        _text = f"".join( [f'\n{action.__str__()}' for action in self.Container] )
        return f"{_text:-^10}"
    
@dataclass(frozen=False)
class DmdLoader:
    startTime:int=0
    endTime:int=120
    CellId:int=0
    rate:float=0
     
    ### --- --- --- --- setter --- --- --- --- 
    def set_(self):
        pass
    
    def __str__(self)->str:
        return f'\startTime: {self.startTime} endTime: {self.endTime} \nCellId: {self.CellId} rate: {self.rate}'
        
    
if __name__ == '__main__':
    ## test
    GenSigidObj = Genid()
    action0 = TimingplanStruct(GenSigidObj= GenSigidObj, agentid=0, sigtiming=[0,50,50,100], StageName=['gstart','gend','rstart','rend'])
    action1 = TimingplanStruct(GenSigidObj= GenSigidObj, agentid=0, sigtiming=[0,40,40,100], StageName=['gstart','gend','rstart','rend'])
    
    Actions = ActionsContainer(Container=list())
    Actions.Container.append(action0)
    Actions.Container.append(action1)
    
    print("action0: ", action0)
    
    print("Actions: ", Actions)

