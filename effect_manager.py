# effect_manager.py

from status_effects import (
    Burn, Poison, Freeze, DamageMultiplier, DefenseMultiplier, HealMultiplier,
    ImmuneDamage, ImmuneControl, BleedEffect, Stun, StatusEffect
)
from effect_mapping import EFFECT_MAPPING, EFFECT_VECTOR_LENGTH

class EffectManager:
    def __init__(self, target):
        """
        初始化 EffectManager。
        :param target: 被管理的角色（玩家或敵人）
        """
        self.target = target
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
                    # 更新duration
                    existing.duration = min(max(existing.duration, effect.duration), existing.max_duration)
                    self.target['battle_log'].append(
                        f"{self.target['profession'].name} 的 {effect.name} 效果已存在，持續回合更新為 {existing.duration}。"
                    )
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
            else:
                # 不可堆疊，檢查是否已存在
                if existing_effects:
                    existing = existing_effects[0]
                    existing.duration = min(max(existing.duration, effect.duration), existing.max_duration)
                    self.target['battle_log'].append(
                        f"{self.target['profession'].name} 的 {effect.name} 效果已存在，持續回合更新為 {existing.duration}。"
                    )
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)

        elif effect.type in['special','control']:
            # non-stackable 效果存在後即使疊加也不更新duration
            if effect.stackable:
                if existing_effects:
                    existing = existing_effects[0]  # 假設所有同名效果堆疊在第一個
                    # 檢查是否超出最大堆疊數
                    fin_stacks = min(existing.stacks+effect.stacks, existing.max_stack)
                    # 更新stack
                    existing.stacks = fin_stacks
                    
            else:
                # 不可堆疊，檢查是否已存在
                if existing_effects:
                    existing = existing_effects[0]
                    self.target['battle_log'].append(
                        f"{self.target['profession'].name} 的 {effect.name} 效果已存在，本次效果無效。"
                    )
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)

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
                         
                            self.target['battle_log'].append(
                                f"{self.target['profession'].name} 的 {effect.name} track 效果已存在。"
                            )
                    else:
                        self.active_effects[effect_id].append(effect)
                        effect.on_apply(self.target)
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
            else:
                same_source = any(e.source == effect.source for e in existing_effects)
                if same_source:
                    for e in existing_effects:
                        # refresh duration
                        e.duration = max(e.duration, effect.duration)
                        self.target['battle_log'].append(
                            f"{self.target['profession'].name} 的 {effect.name} track 效果已存在。"
                        )
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

                            self.target['battle_log'].append(
                                f"{self.target['profession'].name} 的 {effect.name} 效果已存在，堆疊數增加 {added_stacks}。"
                            )
                    else:
                        self.active_effects[effect_id].append(effect)
                        effect.on_apply(self.target)
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
            else:
                same_source = any(e.source == effect.source for e in existing_effects)
                if same_source:
                    for e in existing_effects:
                        self.target['battle_log'].append(
                            f"{self.target['profession'].name} 的 {effect.name}效果已存在。"
                        )
                else:
                    self.active_effects[effect_id].append(effect)
                    effect.on_apply(self.target)
            
        else:
            # 處理未知類型的效果，避免錯誤
            self.target['battle_log'].append(
                f"{self.target['profession'].name} 嘗試添加未知類型的效果: {effect.name}。"
            )

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
                if not self.active_effects[effect_id]:
                    del self.active_effects[effect_id]

    def has_effect(self, effect_name: str) -> bool:
        """
        檢查目標是否擁有特定的效果。
        """
        for effect_id, effects in self.active_effects.items():
            for effect in effects:
                if effect.name == effect_name:
                    return True
        return False

    def get_effects(self, effect_name: str):
        """
        獲取特定效果的所有實例。
        """
        result = []
        for effect_id, effects in self.active_effects.items():
            for effect in effects:
                if effect.name == effect_name:
                    result.append(effect)
        return result

    def remove_all_effects(self, effect_name: str):
        """
        移除所有指定名稱的效果。
        """
        for effect_id, effects in list(self.active_effects.items()):
            for effect in effects:
                if effect.name == effect_name:
                    effect.on_remove(self.target)
                    self.active_effects[effect_id].remove(effect)
                    self.target['battle_log'].append(
                        f"{self.target['profession'].name} 的 {effect.name} 效果已被移除。"
                    )
            if not self.active_effects[effect_id]:
                del self.active_effects[effect_id]

    def set_effect_stack(self, effect_id: int, tar, stacks: int, sources: str = None):
        """
        設置指定效果的堆疊數。正數增加堆疊，負數減少堆疊。
        
        :param effect_id: 效果的唯一識別碼（整數）。
        :param tar: 目標對象，通常是角色本身。
        :param stacks: 要設置的堆疊數（正數增加，負數減少）。
        :param sources: 效果的來源，用於區分不同來源的效果。
        """
        # 對於特定的效果ID，需要指定來源
        if effect_id in [1, 2, 3, 12, 999]:  # 到需要source的效果ID列表
            # 只有：同名字，但效果不同的效果才需要sourc來區分
            if not sources:
                raise ValueError(f"Effect ID {effect_id} 必須指定 source。")
        
        # 初始化列表以存儲需要移除的效果
        effects_to_remove = []
        
        # 檢查效果是否已存在
        if effect_id in self.active_effects:
            for effect in self.active_effects[effect_id]:
                if sources:
                    # 只更新匹配來源的效果
                    if effect.source == sources:
                        effect.set_stack(stacks, tar)
                else:
                    # 如果不需要來源，則更新所有匹配的效果
                    effect.set_stack(stacks, tar)
                
                # 如果堆疊數量小於或等於0，標記為需要移除
                if effect.stacks <= 0:
                    effect.on_remove(tar)
                    effects_to_remove.append(effect)
        
        # 移除標記的效果（避免在迭代時直接修改列表）
        for effect in effects_to_remove:
            self.active_effects[effect_id].remove(effect)
        
        # 如果該效果ID的所有效果都被移除，則從active_effects中刪除該鍵
        if effect_id in self.active_effects and not self.active_effects[effect_id]:
            del self.active_effects[effect_id]


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
                obs[idx + 1] = effect.stacks
                obs[idx + 2] = effect.duration
        return obs
