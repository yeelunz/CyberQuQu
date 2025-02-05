# status_effects.py

import random

class StatusEffect:
    """
    異常狀態基底類，可繼承：
    - name: 狀態名稱
    - duration: 持續回合數
    - max_duration: 最大持續回合數(不設定此項則為當前duration)
        (建議)直接在子類中設定max_duration，避免效果衝突
    - stackable: 是否可疊加
    - max_stack: 最大堆疊數(若是stackable=True，則必須設定此項，否則預設不能堆疊)
        (建議)直接在子類中設定max_duration，避免效果衝突
    - stacks: 當前堆疊數
    - type: 效果類型（'dot', 'buff', 'track', 'control', 'special'）
    - source: 效果來源
        (重要) 同一技能如果同時對敵我雙方都造成需要source區分的效果，需要設定source為source_e或是source_p
        否則雙方是同一角色時會無法區分效果的施用對象
        
    - type詳述:
        dot:
        同一個名字的狀態，其功能必定相同，可以延長時間
            - stackable 同一時間只得存在一個可疊加同名狀態。已存在時會更新duration
            - non-stackable 同一時間只得存在一個不可疊加同名狀態。已存在時會更新duration
        buff:
        同一個名字的狀態，其功能不會相同，且無法延長時間
            - stackable 同一時間可以存在多個可堆疊同名狀態。已存在時不會更新duration，但會更新stack
            - non-stackable 同一時間可以存在多個不可堆疊同名狀態。已存在時不會更新duration
        track:
        同一個名字的狀態，其功能不會相同，可以延長時間
            - stackable 同一時間可以存在多個同名狀態。已存在時會更新duration，且會更新stack
            - non-stackable 同一時間可以存在多個同名狀態。已存在時不會更新duration
        control, special:
        同一個名字的狀態，其功能必定相同，不可以延長時間
            - stackable 同一時間只得存在一個可堆疊同名狀態。已存在時不會更新duration，不會更新stack
            - non-stackable 同一時間只得存在一個不可堆疊同名狀態。已存在時不會更新duration
    """
    def __init__(self, name: str, duration: int, max_duration: int = None, stackable: bool = True, max_stack: int = 1, stacks: int = 1, type: str = 'dot', source=None, id: int = 0):
        self.name = name
        self.duration = duration  # 當前持續時間
        self.max_duration = max_duration if max_duration else duration
        self.stackable = stackable  # 是否可疊加
        self.stacks = stacks  # 當前堆疊數
        self.max_stack = max_stack  # 最大堆疊數
        self.source = source  # 檢查效果來源
        self.type = type
        self.id = id
        # 如果id 被設定為0，則報錯
        if self.id == 0:
            raise ValueError("id 不能為0")
        if not stackable:
            max_stack = 1

    def on_apply(self, target):
        """當效果被施加時執行的邏輯"""
        # 確認duration是否超過max_duration
        self.duration = min(self.duration, self.max_duration)
        # 確認stack是否超過max_stack
        self.stacks = min(self.stacks, self.max_stack)
        pass

    def on_tick(self, target):
        """每回合執行的邏輯"""
        # if buff / dot / track
        if self.type == 'dot':
            pass
        elif self.type == 'buff':
            pass
        elif self.type == 'track':
            pass

    def on_remove(self, target):
        """當效果被移除時執行的邏輯"""
        pass
    
    def set_stack(self,stacks,target):
        if self.stackable:
            # not exceed max_stack
            # check if max stacks and stcaks is int

            fin_stack = min(stacks,self.max_stack)
            self.stacks = fin_stack
        else:
            raise ValueError("不可疊加的效果無法設定層數")
        

# 能力值增減相關效果

class DamageMultiplier(StatusEffect):
    """
        設定一個不可疊加狀態(即同技能效果不會重複施加)，但來自於不同技能則可以疊加
        #TODO 但是obs一樣看不到第二個以後的效果，也看不到multiplier的變化
        則需要設定：
            -duration: 效果持續時間
            -multiplier: 傷害倍率
            -請手動設置 stackable: False
            -source: 效果來源，建議設定為 skill_id
                若是同一技能分別給敵方及我方使用時
                需要額外設定p 或是 e 來區分
        # 如果要使用可以疊加的狀態時：
            - 請設定stackable: True
            - 請設定max_stack: 可以疊加的最大層數
            - 請設定stack: 當前疊加的層數
        否則無法進行疊層
    """
    def __init__(self, multiplier: float, duration: int, stacks = 1,stackable: bool = False, max_stack: int = 1,source=None):
        super().__init__(
            name='攻擊力',
            duration=duration,
            stackable=stackable,
            max_stack=max_stack,
            type='buff',
            id=1,
            source=source,
            stacks=stacks
        )
        self.multiplier = multiplier
        
        # check source
        if not source:
            # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")

    def on_apply(self, target):
        super().on_apply(target)
        
        target['damage_multiplier'] *= self.multiplier  # 直接設置倍率

    def set_stack(self, stacks, target):
        # 先儲存原本的stack
        old_stack = self.stacks
        # 設定新的stack
        super().set_stack(stacks,target)
        new_stack = self.stacks
        # 判斷現在的stack是增加還是減少
        if new_stack > old_stack:
            # 如果是增加，則要進行乘法
            for i in range(new_stack - old_stack):
                target['damage_multiplier'] *= self.multiplier
        else:
            # 如果是減少，則要進行除法
            for i in range(old_stack - new_stack):
                target['damage_multiplier'] /= self.multiplier

    def on_remove(self, target):
        # 這邊要考慮如果是stackable的話，要根據stack來還原
        # 例如：stack = 3, multiplier = 1.5
        # 1.5^3 = 3.375
        # 3.375 / 1.5 = 2.25
        # 2.25 / 1.5 = 1.5
        # 因此要進行3次除法，才能得到原本的multiplier
        
        now_stack = self.stacks
        for i in range(now_stack):
            target['damage_multiplier'] /= self.multiplier  # 恢復為初始值

        # 

        super().on_remove(target)
        
    def update(self, target,now_stack,add_stack):
        # 只能在buff類的stackable效果中使用，用來更新效果
        # 根據stack來更新新的層數，addstack 幾次，就要乘幾次
        self.stacks = now_stack+add_stack
        for i in range(add_stack):
            target['damage_multiplier'] *= self.multiplier



class DefenseMultiplier(StatusEffect):
    """
        設定一個不可疊加狀態(即同技能效果不會重複施加)，但來自於不同技能則可以疊加
        #TODO 但是obs一樣看不到第二個以後的效果，也看不到multiplier的變化
        則需要設定：
            -duration: 效果持續時間
            -multiplier: 傷害倍率
            -請手動設置 stackable: False
            -source: 效果來源，建議設定為 skill_id
                若是同一技能分別給敵方及我方使用時
                需要額外設定p 或是 e 來區分
        # 如果要使用可以疊加的狀態時：
            - 請設定stackable: True
            - 請設定max_stack: 可以疊加的最大層數
            - 請設定stack: 當前疊加的層數
        否則無法進行疊層
    """
    def __init__(self, multiplier: float, duration: int, stacks = 1,stackable: bool = False, max_stack: int = 1,source=None):
        super().__init__(
            name='防禦力',
            duration=duration,
            stackable=stackable,
            max_stack=max_stack,
            type='buff',
            id=2,
            source=source,
            stacks=stacks
        )
        self.multiplier = multiplier
        if not source:
            # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")

    def on_apply(self, target):
        super().on_apply(target)
        target['defend_multiplier'] *= self.multiplier  # 直接設置倍率


    def on_remove(self, target):
        # 這邊要考慮如果是stackable的話，要根據stack來還原
        # 例如：stack = 3, multiplier = 1.5
        now_stack = self.stacks
        for i in range(now_stack):
            target['defend_multiplier'] /= self.multiplier

        super().on_remove(target)
    def update(self, target,now_stack,add_stack):
        # 只能在buff類的stackable效果中使用，用來更新效果
        # 根據 addstack
        self.stacks = now_stack+add_stack
        for i in range(add_stack):
            target['defend_multiplier'] *= self.multiplier
        

        # 
    def set_stack(self, stacks,target): 
        # 先儲存原本的stack
        old_stack = self.stacks
        # 設定新的stack
        super().set_stack(stacks,target)
        new_stack = self.stacks
        # 判斷現在的stack是增加還是減少
        if new_stack > old_stack:
            # 如果是增加，則要進行乘法
            for i in range(new_stack - old_stack):
                target['defend_multiplier'] *= self.multiplier
        else:
            # 如果是減少，則要進行除法
            for i in range(old_stack - new_stack):
                target['defend_multiplier'] /= self.multiplier
        # battle log



class HealMultiplier(StatusEffect):
    """
        設定一個不可疊加狀態(即同技能效果不會重複施加)，但來自於不同技能則可以疊加
        #TODO 但是obs一樣看不到第二個以後的效果，也看不到multiplier的變化
        則需要設定：
            -duration: 效果持續時間
            -multiplier: 傷害倍率
            -請手動設置 stackable: False
            -source: 效果來源，建議設定為 skill_id
                若是同一技能分別給敵方及我方使用時
                需要額外設定p 或是 e 來區分
        # 如果要使用可以疊加的狀態時：
            - 請設定stackable: True
            - 請設定max_stack: 可以疊加的最大層數
            - 請設定stack: 當前疊加的層數
        否則無法進行疊層
    """
    def __init__(self, multiplier: float, duration: int, stacks = 1,stackable: bool = False, max_stack: int = 1,source=None):
        super().__init__(
            name='治癒力',
            duration=duration,
            stackable=stackable,
            max_stack=max_stack,
            type='buff',
            id=3,
            source=source,
            stacks=stacks
        )
        self.multiplier = multiplier
        self.multiplier = multiplier
        if not source:
                # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")

    def on_apply(self, target):
        super().on_apply(target)
        target['heal_multiplier'] *= self.multiplier  # 直接設置倍率


    def on_remove(self, target):
        # 這邊要考慮如果是stackable的話，要根據stack來還原
        now_stack = self.stacks
        for i in range(now_stack):
            target['heal_multiplier'] /= self.multiplier

        super().on_remove(target)
    def update(self, target,now_stack,add_stack):
        # 只能在buff類的stackable效果中使用，用來更新效果
        # 根據 addstack
        self.stacks = now_stack+add_stack
        for i in range(add_stack):
            target['heal_multiplier'] *= self.multiplier

    def set_stack(self, stacks,target):
        # 先儲存原本的stack
        old_stack = self.stacks
        # 設定新的stack
        super().set_stack(stacks,target)
        new_stack = self.stacks
        # 判斷現在的stack是增加還是減少
        if new_stack > old_stack:
            # 如果是增加，則要進行乘法
            for i in range(new_stack - old_stack):
                target['heal_multiplier'] *= self.multiplier
        else:
            # 如果是減少，則要進行除法
            for i in range(old_stack - new_stack):
                target['heal_multiplier'] /= self.multiplier
        # battle log
        # XX效果的層數變為XX



class HealthPointRecover(StatusEffect):
    """
        設定一個不可疊加狀態(即同技能效果不會重複施加)，但來自於不同技能則可以疊加
        #TODO 但是obs一樣看不到第二個以後的效果，也看不到multiplier的變化
        #TODO 但目前還不 support 多stack而增加的回復量
        則需要設定：
            -duration: 效果持續時間
            -multiplier: 此處為固定值，即每回合回復的HP
            -source: 效果來源，建議設定為 skill_id
                若是同一技能分別給敵方及我方使用時
                需要額外設定p 或是 e 來區分
            - env: 環境變數，必須設定
            - roundCalculate: 回復方式，預設為直接回復，若有其他方式，請自行設定為函數
            - roundCalculateArg: 回復方式的參數
            最後會以 roundCalculate(roundCalculateArg) 來計算該回合回復的數值
            - self_mutilation: 是否為自傷效果，預設為False
    """
    def __init__(self, duration: int = 1, hp_recover: int =  10,stackable: bool = False, source = None,env = None,roundCalculate=None,roundCalculateArg=None,self_mutilation = False):
        super().__init__(
            name='生命值持續變更',
            duration=duration,
            stackable=False,
            max_stack=1,
            type='special',
            id=11
        )
        self.hp_recover = hp_recover
        self.env = env
        self.roundCalculate = roundCalculate
        self.roundCalculateArg = roundCalculateArg
        self.self_mutilation = self_mutilation
        if not source:
            # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")

    def on_apply(self, target):
        super().on_apply(target)

    def on_tick(self, target):
        if self.roundCalculate:
            self.hp_recover = self.roundCalculate(self.roundCalculateArg) 
            self.env.deal_healing(target,self.hp_recover,self_mutilation = self.self_mutilation)
        else:
            self.env.deal_healing(target,self.hp_recover,self_mutilation = self.self_mutilation)
# 傷害類/控制類負面效果

class MaxHPmultiplier(StatusEffect):
    """
        你只被允許使用以下參數(且必須使用):
            -duration: 效果持續時間
            -multiplier: 最大HP倍率
            -請手動設置 stackable: False
            -source: 效果來源，建議設定為 skill_id
                若是同一技能分別給敵方及我方使用時
                需要額外設定p 或是 e 來區分
           
    """
    def __init__(self, multiplier: float, duration: int, stackable: bool = False,stacks: int = 1, max_stack: int = 1,source=None):
        super().__init__(
            name='生命值',
            duration=duration,
            stackable=stackable,
            max_stack=max_stack,
            type='buff',
            id=12,
            source=source,
            stacks=stacks
        )
        self.multiplier = multiplier
        if not source:
            # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")

    def on_apply(self, target):
        super().on_apply(target)
        target['max_hp'] *= self.multiplier  # 直接設置倍率

        # 同時回復增加的生命值
        target['hp'] += target['max_hp'] - target['hp']
        
    def on_remove(self, target):
        # 這邊要考慮如果是stackable的話，要根據stack來還原
        now_stack = self.stacks
        for i in range(now_stack):
            target['max_hp'] /= self.multiplier

        # 如果超出生命值最大值，則變為最大值
        target['hp'] = min(target['hp'], target['max_hp'])
        
        super().on_remove(target)
        
    def update(self, target,now_stack,add_stack):
        # 只能在buff類的stackable效果中使用，用來更新效果
        # 根據 addstack
        for i in range(add_stack):
            target['max_hp'] *= self.multiplier
            target['hp'] += target['max_hp'] - target['hp']

        # 同時回復增加的生命值
    def set_stack(self, stacks,target):
        # 先儲存原本的stack
        old_stack = self.stacks
        # 設定新的stack
        super().set_stack(stacks,target)
        new_stack = self.stacks
        
        # 判斷現在的stack是增加還是減少
        if new_stack > old_stack:
            # 如果是增加，則要進行乘法
            for i in range(new_stack - old_stack):
                target['max_hp'] *= self.multiplier
                target['hp'] += target['max_hp'] - target['hp']
        else:
            # 如果是減少，則要進行除法
            for i in range(old_stack - new_stack):
                target['max_hp'] /= self.multiplier
                # 如果超出生命值最大值，則變為最大值
                target['hp'] = min(target['hp'], target['max_hp'])

 
class Burn(StatusEffect):
    """
        你只被允許使用(且必須)使用
            -duration: 燃燒效果的持續時間
            -stacks   :單次施放的效果堆疊數
            
    """            
    def __init__(self, duration: int = 3,stacks:int = 1):
        super().__init__(
            name='燃燒',
            duration=duration,
            stackable=True,
            stacks=stacks,
            max_stack=3,
            type='dot',
            max_duration=3,
            id=4
        )
        self.dmg = 5

    def on_apply(self, target):
        super().on_apply(target)


    def on_tick(self, target):
        dmg = self.dmg * self.stacks
        dmg = min(dmg, self.dmg * self.max_stack)
        target['hp'] = max(0, target['hp'] - dmg)



class Poison(StatusEffect):
    """
        你只被允許(且必須)使用
            -duration: 毒效果的持續時間
            -stacks   :單次施放的效果堆疊數
    """
    def __init__(self, duration: int = 5,stacks:int = 1):
        super().__init__(
            name='中毒',
            duration=duration,
            stackable=True,
            stacks=stacks,
            max_stack=5,
            max_duration=5,
            type='dot',
            id=5
        )
        self.dmg = 3

    def on_apply(self, target):
        super().on_apply(target)


    def on_tick(self, target):
        dmg = self.dmg * self.stacks
        dmg = min(dmg, self.dmg * self.max_stack)
        target['hp'] = max(0, target['hp'] - dmg)

     
class Freeze(StatusEffect):
    """
        你只被允許使用兩個參數
            -duration: 凍結效果的持續時間
            -stacks   :單次施放的效果堆疊數
        
        冰凍的控制邏輯與麻痺跟眩暈不同，冰凍的效果是在行動階段進行處理
        冰凍：被施加冰凍的敵人，受到傷害時15%的機率會被凍結，無法行動1回合
        每個層數增加15%的機率，控制結束後，移除所有冰凍效果
        
        而暈眩與麻痺是在行動階段進行處理，就是duration幾回合，就會被控制幾回合
    """
    def __init__(self, duration: int = 1,stacks:int = 1):
        super().__init__(
            name='凍結',
            duration=duration,
            stacks=stacks,
            stackable=True,
            max_stack=5,
            type='dot',
            max_duration=5,
            id=6
        )
    
    def on_tick(self, target):
        # 凍結不會有持續傷害，已在行動階段處理
        pass


class ImmuneDamage(StatusEffect):
    def __init__(self, duration: int = 1, stackable: bool = False):
        super().__init__(
            name='免疫傷害',
            duration=duration,
            stackable=stackable,
            stacks=1,
            max_stack=1,
            type='special',
            id=7
        )

    def on_apply(self, target):
        super().on_apply(target)


    def on_remove(self, target):

        super().on_remove(target)


class ImmuneControl(StatusEffect):
    def __init__(self, duration: int = 2, stackable: bool = False):
        super().__init__(
            name='免疫控制',
            duration=duration,
            stackable=stackable,
            max_stack=1,
            type='special',
            id=8
        )

    def on_apply(self, target):
        super().on_apply(target)


    def on_remove(self, target):

        super().on_remove(target)


class BleedEffect(StatusEffect):
    """
    用於流血效果
        你只被允許使用2個參數，
        - duration它是流血效果的持續時間。
        - stacks: 單次施放的效果堆疊數
         
    """
    def __init__(self, duration: int = 5,stacks:int = 1):
        super().__init__(
            name='流血',
            duration=duration,
            stackable=True,
            stacks=stacks,
            max_stack=10,
            type='dot',
            max_duration=10,
            id=9
        )
        self.dmg = 1

    def on_apply(self, target):
        super().on_apply(target)


    def on_tick(self, target):
        dmg = self.dmg * self.stacks
        dmg = min(dmg, self.dmg * self.max_stack)
        target['hp'] = max(0, target['hp'] - dmg)


class Paralysis(StatusEffect):
    def __init__(self, duration: int = 1, stackable: bool = False):
        super().__init__(
            name='麻痺',
            duration=duration,
            stackable=False,
            max_stack=1,
            type='control',
            id=10
        )

    def on_apply(self, target):
        super().on_apply(target)
        target["skip_turn"] = True


    def on_remove(self, target):
        target["skip_turn"] = False

        super().on_remove(target)


class Track(StatusEffect):
    """
        必須要使用以下參數(且必須使用):
        name: 效果名稱(用來標記效果)
        duration: 持續回合數
        stackable: 是否可疊加
        max_stack: 最大堆疊數
        source: 效果來源
    """
    def __init__(self, name ,duration: int = 5,stacks = 1 ,stackable: bool = True, max_stack: int = 99,source=None):
        super().__init__(
            name=name,
            duration=duration,
            stackable=stackable,
            max_stack=max_stack,
            stacks=stacks,
            type='track',
            id = 13,
            source=source
        )
        # check source
        if not source:
            # 如果source為空，會無法追蹤效果的來源
            raise ValueError("source 不能為空")


    def on_apply(self, target):
        super().on_apply(target)


    def on_tick(self, target):
        # just to check
        pass
    def set_stack(self, stacks, target):
        # 保存原有堆疊數
        # raise ValueError(")
        # TODO 有空的話修一下
        raise ValueError("追蹤效果不支持set_stack，請直接用值設定stacks")
        

# 這個是用來將效果ID轉換成效果名稱的字典
# 用於顯示的功能
EFFECT_NAME_MAPPING = {
    1: "攻擊力變更",
    2: "防禦力變更",
    3: "治癒力變更",
    4: "燃燒",
    5: "中毒",
    6: "凍結",
    7: "免疫傷害",
    8: "免疫控制",
    9: "流血",
    10: "麻痺",
    11: "回血",
    12: "最大生命值變更",
    13: "追蹤",
}
    
    
    