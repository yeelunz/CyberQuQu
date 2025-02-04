# effect_manager.py
import numpy as np

from .status_effects import (
    Burn, Poison, Freeze, DamageMultiplier, DefenseMultiplier, HealMultiplier,
    ImmuneDamage, ImmuneControl, BleedEffect,  StatusEffect
)
from .effect_mapping import EFFECT_MAPPING, EFFECT_VECTOR_LENGTH
from .battle_event import BattleEvent

class EffectManager:
    def __init__(self, target,env):
        """
        初始化 EffectManager。
        :param target: 被管理的角色（玩家或敵人）
        """
        self.target = target
        self.env = env  # 保存 env
        self.active_effects = {}  # key: effect id, value: list of StatusEffect instances

    def add_effect(self, effect: StatusEffect):
        """
        添加一個狀態效果到目標。
        :param effect: 要添加的狀態效果實例
        """
        effect_id = effect.id
        if effect_id not in EFFECT_MAPPING:
            raise ValueError(f"Effect ID {effect_id} 未在 EFFECT_MAPPING 中定義。請更新 effect_mapping.py。")

        if effect_id not in self.active_effects:
            self.active_effects[effect_id] = []

        existing_effects = self.active_effects[effect_id]

        if effect.type in ['dot']:
            # 此處差別是不可疊加是會不會更新duration
            # 並且無論可不可以疊加，的效果不會有多個同名效果
            
            # TODO 缺少自定義效果功能(尚不支援同名效果並從source區分)
            # TODO 如果是自定義效果的話，僅能從source區分，不可從效果名稱區分
            # TODO refactor
            if effect.stackable:
                if existing_effects:
                    existing = existing_effects[0]  # 假設所有同名效果堆疊在第一個
                    # 檢查是否超出最大堆疊數
                    fin_stacks = min(existing.stacks+effect.stacks, existing.max_stack)
                    # 更新stack
                    existing.stacks = fin_stacks
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_stack_update",appendix={"effect_name":effect.name,"stacks":fin_stacks}))
                    # 更新duration
                    existing.duration = min(max(existing.duration, effect.duration), existing.max_duration)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_duration_update",appendix={"effect_name":effect.name,"duration":existing.duration}))
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
            else:
                # 不可堆疊，檢查是否已存在
                if existing_effects:
                    existing = existing_effects[0]
                    existing.duration = min(max(existing.duration, effect.duration), existing.max_duration)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_duration_update",appendix={"effect_name":effect.name,"duration":existing.duration}))
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))


        elif effect.type in['special','control']:
            # non-stackable 效果存在後即使疊加也不更新duration
            if effect.stackable:
                if existing_effects:
                    existing = existing_effects[0]  # 假設所有同名效果堆疊在第一個
                    # 檢查是否超出最大堆疊數
                    fin_stacks = min(existing.stacks+effect.stacks, existing.max_stack)
                    # 更新stack
                    existing.stacks = fin_stacks
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_stack_update",appendix={"effect_name":effect.name,"stacks":fin_stacks}))
                    
            else:
                # 不可堆疊，檢查是否已存在
                if existing_effects:
                    existing = existing_effects[0]
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply_fail",appendix={"effect_name":effect.name}))
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))


        elif effect.type == 'track':
            # track效果不需要去判斷是否可以堆疊，直接添加
            if effect.stackable:
                if existing_effects:
                    same_source = any(e.source == effect.source for e in existing_effects)
                    if same_source:
                        for e in existing_effects:
                            # refresh duration
                            e.duration = min(max(e.duration, effect.duration), e.max_duration)
                            # add stacks and stack check
                            added_stacks = min(effect.stacks, e.max_stack - e.stacks)
                            e.stacks += added_stacks
                            
                            self.env.add_event(user = self.target, event = BattleEvent(type="status_stack_update",appendix={"effect_name":effect.name,"stacks":e.stacks}))
                    else:
                        
                        self.active_effects[effect_id].append(effect)
                        self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
                        effect.on_apply(self.target)
                else:
                    self.active_effects[effect_id].append(effect)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
                    effect.on_apply(self.target)
            else:
                same_source = any(e.source == effect.source for e in existing_effects)
                if same_source:
                    for e in existing_effects:
                        # refresh duration
                        e.duration = max(e.duration, effect.duration)

                        self.env.add_event(user = self.target, event = BattleEvent(type="status_duration_update",appendix={"effect_name":effect.name,"duration":e.duration}))
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)

   
        elif effect.type == 'buff':
            # track效果不需要去判斷是否可以堆疊，直接添加
            if effect.stackable:
                if existing_effects:
                    same_source = any(e.source == effect.source for e in existing_effects)
                    if same_source:
                        for e in existing_effects:
                            # add stacks and stack check
                            added_stacks = min(effect.stacks, e.max_stack - e.stacks)
                            
                            e.update(self.target,e.stacks,added_stacks)
                            self.env.add_event(user = self.target, event = BattleEvent(type="status_stack_update",appendix={"effect_name":effect.name,"stacks":e.stacks}))
                    else:
                        self.active_effects[effect_id].append(effect)
                        self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
                        effect.on_apply(self.target)
                else:
                    # 不存在
                    self.active_effects[effect_id].append(effect)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
                    effect.on_apply(self.target)
            else:
                same_source = any(e.source == effect.source for e in existing_effects)
                if same_source:
                    for e in existing_effects:
                        self.env.add_event(event = BattleEvent(type="text",text=f"{self.target['profession'].name} 的 {effect.name} 效果已存在。"))

                else:
                    self.active_effects[effect_id].append(effect)
                    self.env.add_event(user = self.target, event = BattleEvent(type="status_apply",appendix={"effect_name":effect.name}))
                    effect.on_apply(self.target)
            
        else:
            self.env.add_event(event = BattleEvent(type="text",text=f"{self.target['profession'].name} 嘗試添加未知類型的效果: {effect.name}。"))


    def tick_effects(self):
        """
        每回合調用，處理所有活躍效果的效果和持續時間減少。
        """
        effects_to_remove = []
        for effect_id, effects in list(self.active_effects.items()):
            for effect in effects:
                effect.on_tick(self.target)
                effect.duration -= 1
                if effect.duration <= 0:
                    effect.on_remove(self.target)
                    effects_to_remove.append((effect_id, effect))

        # 移除已結束的效果
        for effect_id, effect in effects_to_remove:
            if effect in self.active_effects.get(effect_id, []):
                self.active_effects[effect_id].remove(effect)
                self.env.add_event(user = self.target, event = BattleEvent(type="status_remove",appendix={"effect_name":effect.name}))
                if not self.active_effects[effect_id]:
                    del self.active_effects[effect_id]

    def has_effect(self, effect_name: str , source = None) -> bool:
        """
        檢查目標是否擁有特定的效果。 如果指定source，則僅參考來源相同的效果。
        """
        for effect_id, effects in self.active_effects.items():
            for effect in effects:
                if effect.name == effect_name:
                    if source:
                        if effect.source == source:
                            return True
                    else:
                        return True
        return False

    def get_effects(self, effect_name: str, source = None):
        """
        獲取特定效果的所有實例。 如果指定source，則僅返回來源相同的效果。
        """
        result = []
        for effect_id, effects in self.active_effects.items():
            for effect in effects:
                if effect.name == effect_name:
                    if source:
                        if effect.source == source:
                            result.append(effect)
                    else:
                        result.append(effect)
        return result
    
        
    def remove_all_effects(self, effect_name: str,source = None):
        """
        移除所有指定名稱的效果。如果指定source，則僅移除來源相同的效果。
        """
        for effect_id, effects in list(self.active_effects.items()):
            for effect in effects:
                if effect.name == effect_name:
                    if source:
                        if effect.source == source:
                            effect.on_remove(self.target)
                            self.active_effects[effect_id].remove(effect)
                            self.env.add_event(user = self.target, event = BattleEvent(type="status_remove",appendix={"effect_name":effect.name}))
                    else:
                        effect.on_remove(self.target)
                        self.active_effects[effect_id].remove(effect)
                        self.env.add_event(user = self.target, event = BattleEvent(type="status_remove",appendix={"effect_name":effect.name}))
            if not self.active_effects[effect_id]:
                del self.active_effects[effect_id]

    def set_effect_stack(self, effect_name: int, tar, stacks: int, sources: str = None):
        """
        設置指定效果的堆疊數
        
        :param effect_name: status_effect類中的name屬性
        :param tar: 目標對象，通常是角色本身。
        :param stacks: 要設置的堆疊數。
        :param sources: 效果的來源，用於區分不同來源的效果。
        """
        # 對於特定的效果ID，需要指定來源
        if effect_name in ["攻擊力","防禦力","治癒力"]:  # 到需要source的效果ID列表
            # 只有：同名字，但效果不同的效果才需要sourc來區分
            if not sources:
                raise ValueError(f"Effect ID {effect_name} 必須指定 source。")
        
        # 檢查效果是否已存在
        # find effect by name and source
        for effect_id, effects in list(self.active_effects.items()):
            for effect in effects:
                # 如果有source才要檢查source
                # 如果沒有source來源的話，則設定所有的同名效果
                if effect.name == effect_name and (sources) and effect.source == sources:
                    effect.set_stack(stacks, tar)
                # 沒有source來源 全部同名效果都設定
                elif not sources:
                    if effect.name == effect_name:
                        effect.set_stack(stacks, tar)
    
    def get_effect_vector(self):
        """
        將當前所有的效果導出為向量，包括同名效果。
        格式為：[effect id, effect_stack, effect_max_stack, effect_duration, effect_multiplier] * 當前effect數量
        當effect是 'buff' 類型時，包含 multiplier；否則，effect_multiplier 填 0。
        """
        effect_vector = []
        for effect_id, effects in self.active_effects.items():
            for effect in effects:
                eff_id = effect.id
                stacks = effect.stacks
                max_stack = effect.max_stack
                duration = effect.duration
                # 檢查是否為 'buff' 類型且具有 multiplier 屬性
                if effect.type == 'buff' and hasattr(effect, 'multiplier'):
                    multiplier = effect.multiplier
                else:
                    multiplier = 0
                # 將效果資訊添加到向量中
                effect_vector.extend([eff_id, stacks, max_stack, duration, multiplier])
        return effect_vector
        
    def export_obs(self):
        """
        將當前所有的效果導出為固定長度的觀察向量。
        格式為：[存在, 堆疊數, 剩餘持續回合] * effect數量
        未存在的效果填充為 [0, 0, 0]
        """
        obs = [0.0] * EFFECT_VECTOR_LENGTH
        for effect_id, effects in self.active_effects.items():
            if effect_id in EFFECT_MAPPING and effects:  # 確保 effects 非空
                idx = EFFECT_MAPPING[effect_id]
                # 假設每個effect_id只會有一個效果實例（除非堆疊）
                effect = effects[0]
                obs[idx] = 1.0  # 存在標誌
                obs[idx + 1] = np.log(effect.stacks+1)
                if effect.duration > 30:
                    obs[idx + 2] = 1.0
                else:
                    obs[idx + 2] = effect.duration / 30.0 
        return obs
