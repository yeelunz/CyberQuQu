
from dataclasses import dataclass



class BE:
    def __init__(self):
        self.battle_log = ""
        self.efm = EFM(self)
    
    
    def add_battle_log(self,log:str):
        self.battle_log += log
        return self.battle_log
    
@dataclass
class EFM:
    env:BE


test = BE()

# 在外部呼叫
test.add_battle_log("123")
# pritn BE.battle_log
print(test.battle_log)

# 在EMF內部呼叫
test.efm.env.add_battle_log("456")
print(test.battle_log)





    