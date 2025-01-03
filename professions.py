# professions.py

from status_effects import (
    Burn, Poison, Freeze, 
    DamageMultiplier, DefenseMultiplier, HealMultiplier,HealthPointRecover,Paralysis,
     ImmuneDamage, ImmuneControl, BleedEffect
)
import random
from skills import SkillManager, sm

import math

class BattleProfession:
    def __init__(self, profession_id, name, base_hp, passive_desc="", baseAtk=1.0, baseDef=1.0):
        self.profession_id = profession_id
        self.name = name
        self.base_hp = base_hp
        self.max_hp = base_hp
        self.passive_desc = passive_desc
        self.baseAtk = baseAtk
        self.baseDef = baseDef
        self.default_passive_id = (profession_id *-1 )-1

    def get_available_skill_ids(self):
        return []
    def passive(self, user, targets, env):
        pass

    def apply_skill(self, skill_id, user, targets, env):
        env.battle_log.append(
            f"{self.name} 使用了技能 {sm.get_skill_name(skill_id)}。"
        )
        # check cooldown
        cooldowns_skill_id = skill_id - self.profession_id * 3
        if user["cooldowns"].get(cooldowns_skill_id, 0) > 0:
            env.battle_log.append(
                f"{self.name} 的技能 {sm.get_skill_name(skill_id)} 正在冷卻中。"
            )
            return


class Paladin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=0,
            name="聖騎士",
            base_hp=270,
            passive_desc="聖光：攻擊時，15%機率回復最大血量的15%，回復超出最大生命時，對敵方造成50%的回復傷害",
            baseAtk=1.0,
            baseDef=1.2
        )
        self.heal_counts = {}

    def get_available_skill_ids(self):
        return [0, 1, 2]
    
    def passive(self, user, targets, env):
        # 被動技能：15%機率回復最大血量的15%，超出部分造成50%回復傷害
        if random.random() < 0.15:
            heal_amount = int(self.max_hp * 0.15)
            env.battle_log.append(
                f"{self.name} 聖光觸發，恢復了血量。"
            )
            env.deal_healing(user, heal_amount,rate = 0.5,heal_damage = True,target = targets[0]) 

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 0:
            # 技能 0 => 對單體造成40點傷害
            dmg = 40 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == 1:  
            # 技能 1 => 本回合迴避攻擊，回復10點血量。冷卻3回合。
            user["is_defending"] = True
            heal_amount = 10 
            env.deal_healing(user, heal_amount,rate= 0.5,heal_damage = True,target = targets[0])
            # 設置技能冷卻
            user["cooldowns"][1] = 3
            env.battle_log.append(
                f"{self.name} 的技能「堅守防禦」進入冷卻 3 回合。"
            )
        elif skill_id == 2:
            # 技能 2 => 恢復血量，第一次40, 第二次20, 第三次及以後5
            times_healed = user.get("times_healed", 0)
            if times_healed == 0:
                heal_amount = 40
            elif times_healed == 1:
                heal_amount = 20
            else:
                heal_amount = 5
            env.deal_healing(user, heal_amount,rate=0.5,heal_damage = True,target = targets[0])
            user["times_healed"] = times_healed + 1

class Mage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=1,
            name="法師",
            base_hp=180,
            passive_desc="魔力充盈：攻擊造成異常狀態時，15%機率額外疊加一層異常狀態(燃燒或凍結)。",
            baseAtk=1.2,
            baseDef=0.9
        )

    def get_available_skill_ids(self):
        return [3, 4, 5]
    
    def passive(self, user, targets, env):
        # 被動技能：攻擊造成異常狀態時，15%機率額外疊加一層異常狀態(燃燒或凍結)
        for target in targets:
            for effects in target["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, (Burn, Freeze)):
                        if random.random() < 0.15:
                            extra_status = random.choice([
                                Burn(duration=3, stacks=1),
                                Freeze(duration=3, stacks=1)
                            ])
                            env.apply_status(target, extra_status)
                            env.battle_log.append(
                                f"{self.name} 的被動技能「魔力充盈」觸發，對 {target['profession'].name} 施加了額外的 {extra_status.name}。"
                            )
    
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 3:
            # 技能 3 => 對單體造成35點傷害並疊加1層燃燒（最多3層），每層燃燒造成5點傷害
            dmg = 35 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            burn_effect = Burn(duration=3, stacks=1)
            env.apply_status(targets[0], burn_effect)
            # 被動技能：15%機率額外疊加燃燒或凍結
            self.passive(user, targets, env)
                
        elif skill_id == 4:
            # 技能 4 => 對單體造成35點傷害並疊加1層凍結（最多3層）
            dmg = 35 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            freeze_effect = Freeze(duration=3, stacks=1)
            env.apply_status(targets[0], freeze_effect)
            self.passive(user, targets, env)
            
        elif skill_id == 5:
            # 技能 5 => 對敵方全體造成15點傷害，每層燃燒或是凍結增加25點傷害
            base_dmg = 15 * self.baseAtk 
            for target in env.enemy_team:
                if target["hp"] > 0:
                    # 計算燃燒或是凍結
                    total_layers = 0
                    for effects in target["effect_manager"].active_effects.values():
                        for eff in effects:
                            if isinstance(eff, (Burn, Freeze)):
                                total_layers += eff.stacks
                    dmg = base_dmg + 25 * total_layers
                    env.deal_damage(user, target, dmg, can_be_blocked=True)
                    # set burn and freeze stacks to 0
                    env.set_status(target, 4 , 0)
                    env.set_status(target, 6 , 0)
            # 被動技能：15%機率額外疊加燃燒或凍結
            self.passive(user, targets, env)

class Assassin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=2,
            name="刺客",
            base_hp=195,
            passive_desc="飲血：攻擊時額外造成敵方當前3%生命值的傷害",
            baseAtk=1.15,
            baseDef=0.85
        )

    def get_available_skill_ids(self):
        return [6,7,8]

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 6:
            # 技能 6 => 對單體造成30點傷害30%機率傷害翻倍
            dmg = 30 * self.baseAtk
            dmg += int(targets[0]["hp"] * 0.03)
            if random.random() < 0.30:
                env.battle_log.append(
                    f"致命暗殺擊中要害！"
                )
                dmg *= 2
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
           
        elif skill_id == 7:
            # 毒爆=>  引爆中毒的對象，每層造成10點傷害，並回復5點血量
            for target in env.enemy_team:
                if target["hp"] > 0:
                    # 計算燃燒或是凍結
                    total_layers = 0
                    for effects in target["effect_manager"].active_effects.values():
                        for eff in effects:
                            if isinstance(eff, (Poison)):
                                total_layers += eff.stacks
                    dmg =  10 * total_layers
                    env.battle_log.append(
                        f"毒爆引爆對敵人造成傷害！"
                        )
                    env.deal_damage(user, target, dmg, can_be_blocked=True)
                    heal_amount = 5 * total_layers
                    env.battle_log.append(
                        f"毒爆引爆回復自身血量！"
                        )
                    env.deal_healing(user, heal_amount)
                    env.set_status(target, 5 , 0)
        elif skill_id == 8:
            # 對單體造成10點傷害並疊加中毒1~3層（最多5層），每層中毒造成3點傷害
            # 70% 1層 25% 2層 5% 3層
            add_stacks = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05], k=1)[0]
            effect = Poison(duration=3, stacks=add_stacks)
            env.apply_status(targets[0], effect)
        
class Archer(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=3,
            name="弓箭手",
            base_hp=175,
            passive_desc="鷹眼：攻擊時10%機率造成2倍傷害",
            baseAtk=1.1,
            baseDef=0.95
        )

    def get_available_skill_ids(self):
        return [9, 10, 11]
    
    def passive(self, env , dmg):
        # 被動技能：攻擊時10%機率造成2倍傷害
        if random.random() < 0.10:
            env.battle_log.append(
                f"{self.name} 的被動技能「鷹眼」觸發，攻擊造成兩倍傷害！"
            )
            return dmg * 2
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 9:
            # 技能 9 => 對單體造成50點傷害，使對方防禦力下降15%。
            dmg = 50 * self.baseAtk 
            dmg = self.passive(env, dmg)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 降低對方防禦力25%，持續2回合
            def_buff = DefenseMultiplier(multiplier=0.85, duration=2, stackable=False,source=skill_id)
            env.apply_status(targets[0], def_buff)

        elif skill_id == 10:
            # 技能 10 => 2回合間增加150%傷害，或是自身防禦力降低50%
            choice = random.choice(['buff', 'damage'])
            if choice == 'buff':
                dmg_multiplier = 2.5
                dmg_buff = DamageMultiplier(multiplier=dmg_multiplier, duration=2, stackable=False,source=skill_id)
                env.battle_log.append(
                    f"{self.name} 補充成功。"
                )
                env.apply_status(user, dmg_buff)
            else:
                def_multiplier = 0.5
                def_debuff = DefenseMultiplier(multiplier=def_multiplier, duration=1, stackable=False,source=skill_id)
                env.battle_log.append(
                    f"{self.name} 補充失敗。"
                )
                env.apply_status(user, def_debuff)
                
        elif skill_id == 11:
            # 技能 11 => 對單體造成30點傷害，並回復15點血量
            dmg = 30 * self.baseAtk
            dmg = self.passive(env, dmg)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            heal_amount = 15
            env.deal_healing(user, heal_amount)

class Berserker(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=4,
            name="狂戰士",
            base_hp=360,
            passive_desc="狂暴：若自身血量<50%時，攻擊增加失去生命值的20%的傷害。",
            baseAtk=1.0,
            baseDef=0.7
        )

    def get_available_skill_ids(self):
        return [12, 13, 14]
    
    def passive(self, user, dmg, env):
        if user["hp"] < (user["max_hp"] * 0.5):
            loss_hp = user["max_hp"] * 0.5 - user["hp"]
            dmg += loss_hp * 0.2
            env.battle_log.append(
                f"{self.name} 的被動技能「狂暴」觸發，攻擊時增加了 {int(loss_hp * 0.2)} 點傷害。")
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 12:
            # 技能 12 => 對單體造成30點傷害，並自身反嗜傷害基值的5%。
            dmg = 30 * self.baseAtk 
            dmg = self.passive(user, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 自身反噬
            heal = dmg * 0.05
            
            env.battle_log.append(
                f"{self.name} 受到反噬。")
            env.deal_healing(user, heal, self_mutilation = True)
            
        elif skill_id == 13:
            # 技能 13 => 消耗150點血量，接下來5回合每回合回復40點生命值。冷卻5回合。
            if user["hp"] > 150:
                env.deal_healing(user, 150,self_mutilation = True)
                heal_effect = HealthPointRecover(hp_recover=40, duration=5, stackable=False,source=skill_id,env=env)
                env.apply_status(user, heal_effect)
                # 設置技能冷卻
                user["cooldowns"][skill_id] = 5
                # 怒吼進入五回合冷卻
                env.battle_log.append(
                    f"「熱血」進入 5 回合的冷卻。")    
            else:
                env.battle_log.append(
                    f"{self.name} 嘗試使用「熱血」，但血量不足。"
                )
                
        elif skill_id == 14:
            # 技能 14 => 犧牲30點血量，接下來2回合免控，並提升35%防禦力。
            if user["hp"] > 30:
                env.deal_healing(user, 30,self_mutilation = True)
                immune_control = ImmuneControl(duration=2, stackable=False)
                env.apply_status(user, immune_control)
                def_buff = DefenseMultiplier(multiplier=1.35, duration=2, stackable=False,source=skill_id)
                env.apply_status(user, def_buff)
            else:
                env.battle_log.append(
                    f"{self.name} 嘗試使用「血怒」，但血量不足。"
                )

class DragonGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=5,
            name="龍神",
            base_hp=200,
            passive_desc="龍血: 每回合疊加一個龍神狀態，龍神狀態每層增加3%最大生命值、3%攻擊力、3%防禦力",
            baseAtk=1.0,
            baseDef=1.0
        )
        self.dragon_soul_stacks = 0

    def get_available_skill_ids(self):
        return [15, 16, 17]
    def passive(self, user, targets, env):
        # 在env中的passive 實作
        pass

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 15:
            # 技能 15 => 對單體造成25點傷害，每層龍血狀態增加3點傷害。
            base_dmg = 25 * self.baseAtk 
            # get name="龍血buff"的效果
            dragon_soul_effect = user["effect_manager"].get_effects("龍血buff")
            # get stacks
            if dragon_soul_effect:
                stacks = dragon_soul_effect.stacks
            else:
                stacks = 0
            dmg = base_dmg + stacks * 3
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            
        elif skill_id == 16:
            # 技能 16 => 回復120點血量，接下來3回合每回合扣除30點血量，冷卻4回合。
            heal_amount = 120
            env.deal_healing(user, heal_amount)
            bleed_effect = HealthPointRecover(hp_recover=30, duration=3, stackable=False,source=skill_id,env=env,self_mutilation=True)
            env.apply_status(user, bleed_effect)
            # 設置技能冷卻
            user["cooldowns"][skill_id] = 4
        elif skill_id == 17:
            # 技能 17 => 消除一半的龍神狀態的層數，造成層數*20的傷害。
            # get name="龍血buff"的效果
            dragon_soul_effect = user["effect_manager"].get_effects("龍神buff")[0]
            # get stacks
            if dragon_soul_effect:
                stacks = dragon_soul_effect.stacks
            else:
                stacks = 0
            if stacks > 0:
                damage = stacks * 20
                # 消耗了X層龍神狀態
                env.battle_log.append(
                    f"「神龍燎原」消耗了 {stacks} 層龍神狀態。"
                )
                env.deal_damage(user, targets[0], damage, can_be_blocked=True)
                # remove half stacks
                half_stacks = stacks // 2
                # 1 2 and 12 and 999
                # 1: DamageMultiplier
                # 2: DefenseMultiplier
                # 12: Max HP Increase
                # 999: DragonSoul Tracker
                source = self.default_passive_id
                env.set_status(user, 1 , half_stacks,source = source)
                env.set_status(user, 2 , half_stacks,source = source)
                env.set_status(user, 12 , half_stacks,source = source)
                env.set_status(user, 999 , half_stacks,source = source)
            else:
                env.battle_log.append(
                    f"{self.name} 嘗試使用「荒龍燎原」，但沒有龍神狀態。"
                )

class BloodGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=6,
            name="血神",
            base_hp=210,
            passive_desc="攻擊時50%對敵方附加流血狀態，每層流血狀態造成1點傷害，最多可以疊加10層(流血傷害持續5回合)。",
            baseAtk=1.05,
            baseDef=1.0

        )
        self.bleed_stacks = 0

    def get_available_skill_ids(self):
        return [18, 19, 20]
    def passive(self, user, targets, env):
        if random.random() < 0.5:
            env.apply_status(targets[0], BleedEffect(duration=5, stacks=1))
            env.battle_log.append(
                f"{self.name} 的被動技能「血神」觸發！"
            )
        
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
    
        if skill_id == 18:
            # 技能 18 => 血斬：造成25傷害，疊加一層流血狀態。
            dmg = 25 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            bleed_effect = BleedEffect(duration=5, stacks=1)  # 
            env.apply_status(targets[0], bleed_effect)
            # 被動技能：血神流血附加
            self.passive(user, targets, env)
            
        elif skill_id == 19:
            # 技能 19 => 飲血：消耗敵方現在一半的流血狀態，每層消耗的流血狀態對敵方造成5點傷害，並回復3點血量。
            target = targets[0]
            bleed_effects = target["effect_manager"].get_effects("流血")
            total_bleed = sum(eff.stacks for eff in bleed_effects)
            if total_bleed > 0:
                consumed_bleed = total_bleed // 2
                damage = consumed_bleed * 5
                heal_amount = consumed_bleed * 3
                env.deal_damage(user, target, damage, can_be_blocked=True)
                env.deal_healing(user, heal_amount)
                # set bleed stacks to half
                env.set_status(target, 9 , consumed_bleed)
                self.passive(user, targets, env)
                
            else:
                env.battle_log.append(
                    f"{self.name} 嘗試使用「飲血」，但目標沒有流血狀態。"
                )
        elif skill_id == 20:
            # 技能 20 => 血神：隨機對敵方疊加1~5層流血狀態
            target =targets[0]
            add_stacks = random.randint(1, 5)
            bleed_effect = BleedEffect(duration=5, stacks=add_stacks)
            env.apply_status(target, bleed_effect)
            
class SteadfastWarrior(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=7,
            name="剛毅武士",
            base_hp=240,
            passive_desc="堅韌壁壘：每回合開始時恢復已損生命值的10%。",
            baseAtk=0.9,
            baseDef=1.3
        )
    # 
    def passive(self, user, targets, env):
        # 已在env 中實作
        pass
    
    def get_available_skill_ids(self):
        return [21, 22, 23]

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 21:
            # 技能 21 => 對單體造成35點傷害，並降低其20%防禦力，持續2回合
            dmg = 35 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            def_buff = DefenseMultiplier(multiplier=0.8, duration=2, stackable=False,source=skill_id)
            env.apply_status(targets[0], def_buff)

        elif skill_id == 22:
            # 技能 22 => 本回合增加30%防禦力，並回復25點生命值
            def_buff = DefenseMultiplier(multiplier=1.3, duration=1, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)
            heal_amount = 25
            actual_heal = env.deal_healing(user, heal_amount)

        elif skill_id == 23:
            # 技能 23 => 對攻擊者立即造成其本次攻擊傷害的200%，此技能需冷卻3回合
            # 已經在env _process_passives_end_of_turn中實作 
            pass

class SunWarrior(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=8,
            name="烈陽勇士",
            base_hp=205,
            passive_desc="太陽庇佑：自身為燃燒狀態時，攻擊力時，傷害增加35%。",
            baseAtk=1.25,
            baseDef=0.75
        )

    def get_available_skill_ids(self):
        return [24, 25, 26]
    
    def passive(self, user,dmg, env):
        # 被動技能：自身為燃燒狀態時，攻擊力增加35%
        if user["effect_manager"].has_effect('燃燒'):
            dmg = dmg * 1.35
            env.battle_log.append(
                f"{self.name} 「太陽庇佑」觸發。"
            )
            return dmg * 1.35
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 24:
            # 技能 24 => 對單體造成30點傷害，並附加「燃燒」狀態
            dmg = 30 * self.baseAtk 
            dmg = self.passive(user, dmg, env)
            
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            burn_effect = Burn(duration=3, stacks=1)
            env.apply_status(targets[0], burn_effect)

        elif skill_id == 25:
            # 技能 25 => 本回合防禦力增加30%，並對攻擊者附加1層燃燒
            # 剩下在env中實作
            def_buff = DefenseMultiplier(multiplier=1.3, duration=1, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)

        elif skill_id == 26:
            # 技能 26 => 對敵方全體造成20點傷害，並使目標燃燒效果加倍，根據燃燒層數造成額外傷害
            base_dmg = 20 * self.baseAtk
            base_dmg = self.passive(user, base_dmg, env)
            
            for target in env.enemy_team:
                if target["hp"] > 0:
                    # 造成基礎傷害
                    env.deal_damage(user, target, base_dmg, can_be_blocked=True)
                    burn_effects = target["effect_manager"].get_effects("燃燒")
                    
                    for burn in burn_effects:
                        original_stacks = burn.stacks
                        extra_dmg = (burn.stacks - original_stacks) * 5  # 每層燃燒造成5點傷害
                        if extra_dmg > 0:
                            env.deal_damage(user, target, extra_dmg, can_be_blocked=True)
                            env.set_status(target, 4 , original_stacks * 2)


class Ranger(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=9,
            name="荒原遊俠",
            base_hp=210,
            passive_desc="冷箭：受到攻擊時，15%機率反擊25點傷害。",
            baseAtk=1.3,
            baseDef=0.9
        )

    def get_available_skill_ids(self):
        return [27, 28, 29]
    def passive(self, user, targets, env):
        # 已在env 中 deal damage 實作
        pass
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 27:
            # 技能 27 => 續戰：造成35傷害，每次連續使用攻擊技能時多增加10點傷害。
            times_used = user.get("skills_used", {}).get(skill_id, 0)
            dmg = (35 + (10 * times_used)) * self.baseAtk 
            
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            user["skills_used"][skill_id] = times_used + 1
        elif skill_id == 28:
            # 技能 28 => 埋伏：一回合內，提升50%防禦。
            def_buff = DefenseMultiplier(multiplier=1.5, duration=1, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)

        elif skill_id == 29:
            # 技能 29 => 荒原：消耗15點生命力，免疫一回合的傷害。
            if user["hp"] > 15:
                user["hp"] -= 15
                immune_damage = ImmuneDamage(duration=1, stackable=False)
                env.apply_status(user, immune_damage)
            else:
                env.battle_log.append(
                    f"{self.name} 嘗試使用「荒原」，但血量不足。"
                )


class ElementalMage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=10,
            name="元素法師",
            base_hp=170,
            passive_desc="元素之力：攻擊時25%造成(麻痺、冰凍、燃燒)其中之一",
            baseAtk=1.15,
            baseDef=1.0
        )

    def get_available_skill_ids(self):
        return [30, 31, 32]
    
    def passive(self, user, targets, env):
        # 元素之力：攻擊時25%造成(麻痺、冰凍、燃燒)其中之一
        if random.random() < 0.25:
            effect = random.choice([Burn(duration=3, stacks=1), Freeze(duration=3, stacks=1), Paralysis(duration=2)])
            env.battle_log.append(
                f"{self.name} 攻擊時爆發了「元素之力」。"
            )
            env.apply_status(targets[0], effect)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 30:
            # 技能 30 => 雷霆護甲：2回合內，受到傷害15%機率直接麻痺敵人。
            def_buff = DefenseMultiplier(multiplier=1.2, duration=2, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)
            # 護甲麻痺部分在env中process_passives_end_of_turn實作

        elif skill_id == 31:
            # 技能 31 => 凍燒雷：造成40點傷害，每層麻痺、冰凍、燃燒，額外造成10點傷害
            dmg = 40 * self.baseAtk * env.damage_coefficient
            for target in env.enemy_team:
                if target["hp"] > 0:
                    env.deal_damage(user, target, dmg, can_be_blocked=True)
                    # 計算所有異常狀態的堆疊數總和
                    total_layers = 0
                    for effects in target["effect_manager"].active_effects.values():
                        for eff in effects:
                            if eff.name in ["麻痺", "凍結", "燃燒"]:
                                total_layers += eff.stacks
                    extra_dmg = 10 * total_layers * env.damage_coefficient
                    if extra_dmg > 0:
                        env.deal_damage(user, target, extra_dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == 32:
            # 技能 32 => 雷擊術：50%機率使敵方暈眩1~3回合
            target = targets[0]
            # 70% 2 25% 3 10% 4
            para_duration = random.choices([2, 3, 4], weights=[0.7, 0.25, 0.05], k=1)[0]
            if random.random() < 0.5:
                stun_effect = Paralysis(duration=para_duration)
                env.apply_status(target, stun_effect)
 
            
class HuangShen(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=11,  # 確保職業ID唯一
            name="荒神",
            base_hp=205,
            baseAtk=0.95,
            baseDef=1.2,
            passive_desc="荒蕪：當回合共計受到血量最大值的35%以上傷害時：在回合結束時回復已損生命的5%。",
        )
    def get_available_skill_ids(self):
        return [33, 34, 35]
    def passive(self, user, targets, env):
        # 已在env 中實作
        pass
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 33:
            # 技能 33 => 對單體造成65點傷害，自身受到15點傷害。
            dmg = 20 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            dmg_heal = 15
            env.deal_healing(user, dmg_heal, self_mutilation = True)
        elif skill_id == 34:
        #  技能 34 =>  荒神戰意：技能使用完，下次會切換效果，周而復始
        #   第一次：增加自身25%攻擊力，持續3回合
        # 	第二次：增加自身25%治癒力，持續3回合
        # 	第三次：提升自身25%防禦力，持續3回合。
            # 技能 34 => 對敵方全體造成15點傷害
            time_used = user.get("skills_used", {}).get(skill_id, 0)
            if time_used % 3 == 0:
                atk_buff = DamageMultiplier(multiplier=1.25, duration=3, stackable=False,source=skill_id)
                env.apply_status(user, atk_buff)
            elif time_used % 3 == 1:
                heal_buff = HealMultiplier(multiplier=1.25, duration=3, stackable=False,source=skill_id)
                env.apply_status(user, heal_buff)
            elif time_used % 3 == 2:
                def_buff = DefenseMultiplier(multiplier=1.25, duration=3, stackable=False,source=skill_id)
                env.apply_status(user, def_buff)
            user["skills_used"][skill_id] = time_used + 1
        # ：造成90點傷害，自身生命值越低。會降低傷害並回復自身生命值
        elif skill_id == 35:
            #  基礎傷害90，會隨著生命值越低而降低傷害，傷害最低 15
            #  基礎回血5，會隨著生命值越低，增加回血，最高 50
            hp_ratio = user["hp"] / user["max_hp"]
            damage = 25 + (90 - 15) * math.exp(-8.53 * (1 - hp_ratio))
            # 計算回血
            heal = 5 + (50 - 5) * (1 - math.exp(-0.84 * (1 - hp_ratio)))
            env.deal_damage(user, targets[0], damage, can_be_blocked=True)
            env.deal_healing(user, heal)
            
class GodOfStar(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=12,
            name="星神",
            base_hp=210,
            passive_desc="天啟星盤：星神在戰鬥中精通增益與減益效果的能量運用。每當場上有一層「能力值增益」或「減益」效果時，攻擊時會為自身額外增加 5點傷害 並回復 5點生命值。。",
            baseAtk=1.12,
            baseDef=1.12
        )
    def get_available_skill_ids(self):
        return [36, 37, 38]
    def passive(self, user, targets, env):
        # get all status types is 'buff'
        buff_effects = []
        for eff in user["effect_manager"].active_effects.values():
            for effect in eff:
                if effect.type == "buff":
                    buff_effects.append(effect)
        for  eff in targets[0]["effect_manager"].active_effects.values():
            for effect in eff:
                if effect.type == "buff":
                    buff_effects.append(effect)
        # get len
        buff_effect_count = len(buff_effects)
        bounous_damage = 5 * buff_effect_count
        bounous_heal = 5 * buff_effect_count
        # battle log "能量聚於天啟星盤，攻擊時額外造成了 {bounous_damage} 點傷害，並回復了 {bounous_heal} 點生命值。"
        env.battle_log.append(
            f"能量聚於天啟星盤，攻擊時會額外造成 {bounous_damage} 點傷害，並會回復 {bounous_heal} 點生命值。"
        )
        return bounous_damage, bounous_heal
#     sm.add_skill(Skill(36, "光輝流星", "對敵方單體造成 15 點傷害，並隨機為自身附加以下一種增益效果，持續 3 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。", 'damage'))
# sm.add_skill(Skill(37, "災厄隕星", "為自身恢復 15 點生命值，並隨機為敵方附加以下一種減益效果，持續 3 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。", 'damage'))
# sm.add_skill(Skill(38, "虛擬創星圖", "對敵方單體造成 45 點傷害。", 'damage'))
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        dmg, heal = self.passive(user, targets, env)
        if skill_id == 36:
            # 技能 36 => 對單體造成30點傷害
            dmg += 15 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 隨機為自身附加以下一種增益效果，持續 3 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。
            buff = random.choice([DamageMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id), DefenseMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id), HealMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id)])
            env.apply_status(user, buff)
        elif skill_id == 37:
            # 技能 37 => 為自身恢復 15 點生命值
            heal += 15
            env.deal_healing(user, heal)
            # 隨機為敵方附加以下一種減益效果，持續 3 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。
            debuff = random.choice([DamageMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id), DefenseMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id), HealMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id)])
            env.apply_status(targets[0], debuff)
        elif skill_id == 38:
            # 技能 38 => 對敵方單體造成 45 點傷害
            dmg += 45 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)