import json
import time

class Gdata:
    """
     這個是用來儲存資料的類別
     分為data和version
     
     其中
     - data 是 DICT格式，主要儲存的資料，按照type不同而有不同的資料結構
     - version 是 INT格式 用來確保資料是否過期
     - type 是 STR格式 用來確保資料的類型
        有以下幾種類型
        - cross_validation_pc : 用來記錄交叉驗證的資料
        - cross_validation_ai : 用來記錄交叉驗證的資料
        - version_change : 用來記錄版本變更的資料
        - elo : 用來記錄elo的資料
        - version_legacy : 用來記錄完整版本的職業技能資訊的資料
        
     - name並不作為唯一識別，只是用來標記資料的名稱
     - hash 是用來確保資料的唯一性 
    """
    def __init__(self, data,version,type ,name = str(time.time())):
        self.data = data
        self.version = version
        self.type = type
        self.name = name
        self.hash = hash(str(data)+str(version)+type+name)
    def save(self,file_name = None):
        # 用json格式儲存
        if file_name == None:
            file_name = self.name+".json"
        path = f"data/{self.type}/{file_name}"
            
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
        
        