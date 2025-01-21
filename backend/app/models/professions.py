# professions.py

from .status_effects import (
    Burn, Poison, Freeze, 
    DamageMultiplier, DefenseMultiplier, HealMultiplier,HealthPointRecover,Paralysis,
     ImmuneDamage, ImmuneControl, BleedEffect,Track
)
import random
from .skills import SkillManager, sm

from .profession_var import *

from .battle_event import BattleEvent
import math

class BattleProfession:
    def __init__(self, profession_id, name, base_hp, passive_name,passive_desc="", baseAtk=1.0, baseDef=1.0):
        self.profession_id = profession_id
        self.name = name
        self.base_hp = base_hp
        self.max_hp = base_hp
        self.passive_name = passive_name
        self.passive_desc = passive_desc
        self.baseAtk = baseAtk
        self.baseDef = baseDef
        self.default_passive_id = (profession_id *-1 )-1

    def get_available_skill_ids(self,cooldowns:dict):
        # get cooldowns
        ok_skills = []
        if cooldowns[0] == 0:
            ok_skills.append(self.profession_id* 3)
        if cooldowns[1] == 0:
            ok_skills.append(self.profession_id*3 +1)
        if cooldowns[2] == 0:
            ok_skills.append(self.profession_id*3 +2)
        # return is real skill id
        return ok_skills
    def damage_taken(self, user, targets, env,dmg):
        """
            自身受到傷害時的特效
        """
        pass
    def passive(self, user, targets, env):
        """
         這邊只處理文字部分，如要動畫則在該profession中實作
        """
        pass

    def on_turn_start(self, user, targets, env, id):
        """
        當回合開始時的效果
            self: 使用者
            targets: 目標
            env: 環境
            id : 技能id(如果是被動技能則為-1，否則為該技能的id)
        """
        pass
    def on_turn_end(self,  user, targets, env, id):
        """
        當回合開始時的效果
            self: 使用者
            targets: 目標
            env: 環境
            id : 技能id(如果是被動技能則為-1，否則為該技能的id)
        """
        pass
    def on_attack_end(self, user, targets, env):
        pass
    def apply_skill(self, skill_id, user, targets, env):
        cooldowns_skill_id = skill_id - self.profession_id * 3
        env.add_event(user = user, event = BattleEvent(type="skill",appendix={"skill_id":skill_id,"relatively_skill_id":cooldowns_skill_id}))
        # check cooldown
        # get skill id's cooldown
        
        if skill_id in self.get_available_skill_ids(user["cooldowns"]):
            # set cooldown
            local_skill_id = skill_id - self.profession_id * 3
            user["cooldowns"][local_skill_id] = sm.get_skill_cooldown(skill_id)
            if sm.get_skill_cooldown(skill_id) > 0:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」進入冷卻 {sm.get_skill_cooldown(skill_id)} 回合。"))
            
            return -1
        else:
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」還在冷卻中。"))
            return -1

class Paladin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=0,
            name="聖騎士",
            base_hp=PALADIN_VAR['PALADIN_BASE_HP'],
            passive_name = "聖光",
            passive_desc="攻擊時，30% 機率恢復最大血量的15%的生命值，回復超出最大生命時，對敵方造成50%的回復傷害",
            baseAtk=PALADIN_VAR['PALADIN_BASE_ATK'],
            baseDef=PALADIN_VAR['PALADIN_BASE_DEF']
        )
        self.heal_counts = {}

    def passive(self, user, targets, env):
        # 被動技能：15%機率回復最大血量的15%，超出部分造成100%回復傷害
        super().passive(user, targets, env)
        
        if random.random() < PALADIN_VAR['PALADIN_PASSIVE_TRIGGER_RATE']:
            heal_amount = int(self.max_hp * 0.15)
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 聖光觸發，恢復了血量。"))
            env.deal_healing(user, heal_amount,rate = 1,heal_damage = True,target = targets[0]) 

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
            heal_amount = 15
            env.deal_healing(user, heal_amount,rate= 1,heal_damage = True,target = targets[0])


        elif skill_id == 2:
            # 技能 2 => 恢復血量，第一次50, 第二次35, 第三次及以後15
            times_healed = user.get("times_healed", 0)
            if times_healed == 0:
                heal_amount = 50
            elif times_healed == 1:
                heal_amount = 35
            else:
                heal_amount = 15
            env.deal_healing(user, heal_amount,rate=1,heal_damage = True,target = targets[0])
            user["times_healed"] = times_healed + 1

class Mage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=1,
            name="法師",
            base_hp=249,
            passive_name = "魔力充盈",
            passive_desc="攻擊造成異常狀態時，50% 機率額外疊加一層異常狀態(燃燒或冰凍)。",
            baseAtk=1.65,
            baseDef=1.0
        )

    def passive(self, user, targets, env):
        # 被動技能：攻擊造成異常狀態時，15%機率額外疊加一層異常狀態(燃燒或凍結)
        target = targets[0]
        if random.random() < 0.5:
            extra_status = random.choice([
                Burn(duration=3, stacks=1),
                Freeze(duration=3, stacks=1)
            ])
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「魔力充盈」觸發，對 {target['profession'].name} 施加了額外的 {extra_status.name}。"))
            env.apply_status(target, extra_status)
 
            
    
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
            base_dmg = 25 * self.baseAtk
            target = targets[0]
            # 計算燃燒或是凍結
            total_layers = 0
            for effects in target["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, (Burn, Freeze)):
                        total_layers += eff.stacks
            dmg = base_dmg + 40 * total_layers
            env.deal_damage(user, target, dmg, can_be_blocked=True)
            # set burn and freeze stacks to 0
            # remove burn and freeze
            target["effect_manager"].remove_all_effects("燃燒")
            target["effect_manager"].remove_all_effects("凍結")

class Assassin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=2,
            name="刺客",
            base_hp=258,
            passive_name = "刺殺",
            passive_desc="攻擊時額外造成敵方當前5%生命值的傷害",
            baseAtk=1.15,
            baseDef=0.95
        )
    def passive(self,targets,damage):
        return  damage + int(targets[0]["hp"] * 0.05)
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 6:
            # 技能 6 => 對單體造成45點傷害35%機率傷害翻倍
            dmg = 40 * self.baseAtk
            dmg = self.passive(targets,dmg)
            if random.random() < 0.35:
                env.add_event(event = BattleEvent(type="text",text=f"擊中要害！"))
                dmg *= 2
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
           
        elif skill_id == 7:
            # 毒爆=>  引爆中毒的對象，每層造成10點傷害，並回復8點血量
            for target in env.enemy_team:
                if target["hp"] > 0:
                    # 計算燃燒或是凍結
                    total_layers = 0
                    for effects in target["effect_manager"].active_effects.values():
                        for eff in effects:
                            if isinstance(eff, (Poison)):
                                total_layers += eff.stacks
                    dmg =  15 * total_layers
                    dmg = self.passive(targets,dmg)

                    env.deal_damage(user, target, dmg, can_be_blocked=True)
                    heal_amount = 15 * total_layers
                    env.deal_healing(user, heal_amount)
                    target["effect_manager"].remove_all_effects("中毒")
        elif skill_id == 8:
            # 對單體造成10點傷害並疊加中毒1~3層（最多5層），每層中毒造成3點傷害
            # 45% 1層 35% 2層 20% 3層
            add_stacks = random.choices([1, 2, 3], weights=[0.35, 0.45, 0.2], k=1)[0]
            effect = Poison(duration=3, stacks=add_stacks)
            env.apply_status(targets[0], effect)
        
class Archer(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=3,
            name="弓箭手",
            base_hp=234,
            passive_name = "鷹眼",
            passive_desc="攻擊時15% 機率造成2倍傷害；敵方防禦基值越高時，額外增加鷹眼的觸發機率。",
            baseAtk=1.15,
            baseDef=1.03
        )
    
    def passive(self, env , dmg,tar):
        # 被動技能：攻擊時10%機率造成2倍傷害
        t  = tar[0]
        prob = 0.15
        if t["profession"].baseDef > 1:
            prob = 0.15 + (t["profession"].baseDef - 1) *2
            # but not exceed 0.5
            prob = min(prob, 0.50)
        if random.random() < prob:
            env.add_event(event = BattleEvent(type="text",text=f"被動技能「鷹眼」觸發，攻擊造成兩倍傷害！"))
            return dmg * 2
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 9:
            # 技能 9 => 對單體造成50點傷害，使對方防禦力下降10%。
            dmg = 50 * self.baseAtk 
            dmg = self.passive(env, dmg,targets)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 降低對方防禦力25%，持續2回合
            def_buff = DefenseMultiplier(multiplier=0.9, duration=2, stackable=False,source=skill_id)
            env.apply_status(targets[0], def_buff)

        elif skill_id == 10:
            # 技能 10 => 2回合間增加150%傷害，或是自身防禦力降低50%
            choice = random.choice(['buff', 'damage'])
            if choice == 'buff':
                dmg_multiplier = 3.5
                dmg_buff = DamageMultiplier(multiplier=dmg_multiplier, duration=2, stackable=False,source=skill_id)

                env.add_event(event = BattleEvent(type="text",text=f"箭矢補充成功。"))
                env.apply_status(user, dmg_buff)
            else:
                def_multiplier = 0.5
                def_debuff = DefenseMultiplier(multiplier=def_multiplier, duration=1, stackable=False,source=skill_id)

                env.add_event(event = BattleEvent(type="text",text=f"箭矢補充失敗。"))
                env.apply_status(user, def_debuff)
                
        elif skill_id == 11:
            # 技能 11 => 對單體造成30點傷害，並回復15點血量
            dmg = 30 * self.baseAtk
            dmg = self.passive(env, dmg,targets)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            heal_amount = 15
            env.deal_healing(user, heal_amount)

class Berserker(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=4,
            name="狂戰士",
            base_hp=430,
            passive_name = "狂暴",
            passive_desc="若自身血量<50%時，攻擊增加失去生命值的30%的傷害。",
            baseAtk=1.0,
            baseDef=0.76
        )

    def passive(self, user, dmg, env):
        if user["hp"] < (user["max_hp"] * 0.5):
            loss_hp = user["max_hp"] - user["hp"]
            dmg += loss_hp * 0.3
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「狂暴」觸發，攻擊時增加了 {int(loss_hp * 0.35)} 點傷害。"))
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 12:
            # 技能 12 => 對單體造成30點傷害，並自身反嗜15%造成的傷害。
            dmg = 30 * self.baseAtk 
            dmg = self.passive(user, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 自身反噬
            heal = dmg * 0.15
    
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 受到反噬。"))
            env.deal_healing(user, heal, self_mutilation = True)
            
        elif skill_id == 13:
            # 技能 13 => 消耗150點血量，接下來5回合每回合回復40點生命值。冷卻5回合。
            if user["hp"] > 150:
                env.deal_healing(user, 150,self_mutilation = True)
                heal_effect = HealthPointRecover(hp_recover=40, duration=5, stackable=False,source=skill_id,env=env)
                env.apply_status(user, heal_effect)
            else: 
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 嘗試使用「熱血」，但血量不足。"))


                
        elif skill_id == 14:
            # 技能 14 => 犧牲30點血量，接下來2回合免控，並提升35%防禦力。
            if user["hp"] > 30:
                env.deal_healing(user, 30,self_mutilation = True)
                immune_control = ImmuneControl(duration=2, stackable=False)
                env.apply_status(user, immune_control)
                def_buff = DefenseMultiplier(multiplier=1.35, duration=2, stackable=False,source=skill_id)
                env.apply_status(user, def_buff)
            else:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 嘗試使用「血怒」，但血量不足。"))

class DragonGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=5,
            name="龍神",
            base_hp=315,
            passive_name = "龍血",
            passive_desc="每回合疊加一個龍神狀態，龍神狀態每層增加 5% 攻擊力、增加 5% 防禦力",
            baseAtk=1.05,
            baseDef=1.05
        )
        self.dragon_soul_stacks = 0

    def passive(self, user, targets, env):
        # 在env中的passive 實作
        pass
    def on_turn_start(self, user, targets, env,id):
        env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「龍血」觸發！"))
        
        deffect = DefenseMultiplier(multiplier=1.05,duration=99,stacks=1,source=self.default_passive_id,stackable=True,max_stack=99)
        heffect = DamageMultiplier(multiplier=1.05,duration=99,stacks=1,source=self.default_passive_id,stackable=True,max_stack=99)
        # hpeffect = MaxHPmultiplier(multiplier=1.02,duration=99,stacks=1,source=passive_id,stackable=True,max_stack=99)
        track = Track(name="龍神buff",duration=99,stacks=1,source=self.default_passive_id,stackable=True,max_stack=99)
        env.apply_status(user,heffect)
        env.apply_status(user,deffect)
        env.apply_status(user,track)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 15:
            # 技能 15 => 對單體造成25點傷害，每層龍血狀態增加5點傷害。
            base_dmg = 30 * self.baseAtk 
            # get name="龍血buff"的效果
            dragon_soul_effect = user["effect_manager"].get_effects("龍神buff")[0]
            # get stacks
            if dragon_soul_effect:
                stacks = dragon_soul_effect.stacks
            else:
                stacks = 0
            dmg = base_dmg + stacks * 5
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            
        elif skill_id == 16:
            # 技能 16 => 回復100點血量，接下來3回合每回合扣除30點血量，冷卻4回合。
            heal_amount = 120
            env.deal_healing(user, heal_amount)
            bleed_effect = HealthPointRecover(hp_recover=30, duration=3, stackable=False,source=skill_id,env=env,self_mutilation=True)
            env.apply_status(user, bleed_effect)
        elif skill_id == 17:
            # 技能 17 => 消除一半的龍神狀態的層數，造成層數*25的傷害。
            # get name="龍血buff"的效果
            dragon_soul_effect = user["effect_manager"].get_effects("龍神buff")[0]
            # get stacks
            if dragon_soul_effect:
                stacks = dragon_soul_effect.stacks
            else:
                stacks = 0
            if stacks // 2 > 0:
                consume_stack = stacks // 2
                damage = consume_stack * 35
                # 消耗了X層龍神狀態
                env.deal_damage(user, targets[0], damage, can_be_blocked=True)

                env.add_event(event = BattleEvent(type="text",text=f"「神龍燎原」消耗了 {consume_stack} 層龍神狀態。"))
                # 1 2 and 12 and 999
                # 1: DamageMultiplier
                # 2: DefenseMultiplier
                # 12: Max HP Increase
                # 999: DragonSoul Tracker
                source = self.default_passive_id
                env.set_status(user, "攻擊力" , consume_stack,source = source)
                env.set_status(user, "防禦力" , consume_stack,source = source)
                dragon_soul_effect.stacks = consume_stack

            else:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 嘗試使用「荒龍燎原」，但沒有龍神狀態。"))


class BloodGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=6,
            name="血神",
            base_hp=296,
            passive_name = "出血",
            passive_desc="攻擊時 50% 機率對敵方附加流血狀態。流血狀態會對敵方造成 1 點傷害，最多可以疊加10層(流血傷害持續5回合)。",
            baseAtk=1.15,
            baseDef=1.15

        )
        self.bleed_stacks = 0

    def passive(self, user, targets, env):
        if random.random() < 0.25:
            env.apply_status(targets[0], BleedEffect(duration=5, stacks=1))
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「出血」觸發，對 {targets[0]['profession'].name} 造成了流血狀態。"))

        
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
    
        if skill_id == 18:
            # 技能 18 => 血斬：造成25傷害，疊加一層流血狀態。
            dmg = 45 * self.baseAtk 
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 被動技能：血神流血附加
            self.passive(user, targets, env)
            
        elif skill_id == 19:
            # 技能 19 => 飲血：消耗敵方現在一半的流血狀態，每層消耗的流血狀態對敵方造成5點傷害，並回復5點血量。
            target = targets[0]
            bleed_effects = target["effect_manager"].get_effects("流血")
            total_bleed = sum(eff.stacks for eff in bleed_effects)
            if total_bleed > 0:
                consumed_bleed = total_bleed // 2
                damage = consumed_bleed * 15
                heal_amount = consumed_bleed * 15
                env.deal_damage(user, target, damage, can_be_blocked=True)
                env.deal_healing(user, heal_amount)
                # set bleed stacks to half
                self.passive(user, targets, env)
                
            else:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 嘗試使用「飲血」，但目標沒有流血狀態。"))
                
        elif skill_id == 20:
            # 技能 20 => 赤紅之災：隨機對敵方疊加3~5層流血狀態
            target =targets[0]
            add_stacks = random.randint(1, 3)
            bleed_effect = BleedEffect(duration=5, stacks=add_stacks)
            env.apply_status(target, bleed_effect)
            
class SteadfastWarrior(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=7,
            name="剛毅武士",
            base_hp=294,
            passive_name = "堅韌壁壘",
            passive_desc="每回合開始時恢復已損生命值的10%。",
            baseAtk=1.0,
            baseDef=1.35
        )
    # 
    def passive(self, user, targets, env):
        # 已在on_turn_start 中實作
        pass
    
    def on_turn_start(self, user, targets, env , id):
        heal = int(self.max_hp - user["hp"] * 0.1)
        env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「堅韌壁壘」觸發。"))
        env.deal_healing(user, heal)

    
    def on_turn_end(self, user, targets, env, id):
        if id == 23 :
            if user["last_attacker"] :
                    dmg = user["last_damage_taken"] * 3
                    env.deal_damage(user, user["last_attacker"], dmg, can_be_blocked=True)
            else :
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「絕地反擊」沒有對象反擊。"))

        
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

class Devour(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=8,
            name="鯨吞",
            base_hp=800,
            passive_name = "巨鯨",
            passive_desc="攻擊時會消耗8%當前生命值。",
            baseAtk=1.0,
            baseDef=1.0
        )
    
    def passive(self, user,dmg, env):
        # 巨鯨：攻擊時會消耗5%當前生命值。
        env.deal_healing(user, int(user["hp"] * 0.08),self_mutilation = True)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 24:
            # 吞裂", "造成65點傷害，50%機率使用失敗。
            dmg = 65 * self.baseAtk
            if random.random() < 0.5:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的技能「吞裂」使用失敗。"))

            else:
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, dmg, env)

        elif skill_id == 25:
            # 巨口吞世", "當敵方當前血量比例較我方高時，對敵方造成已損血量15%的傷害，否則，造成當前血量10%的傷害。冷卻3回合。"
            if targets[0]["hp"] > user["hp"]:
                dmg = int(user["max_hp"] * 0.15) * self.baseAtk
            else:
                dmg = int(user["hp"] * 0.1) * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, dmg, env)
        elif skill_id == 26:
            # "堅硬皮膚", "提升25%防禦力，持續3回合。"
            def_buff = DefenseMultiplier(multiplier=1.35, duration=3, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)
    
class Ranger(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=9,
            name="荒原遊俠",
            base_hp=269,
            passive_name = "冷箭",
            passive_desc="冷箭：受到攻擊時，25% 機率反擊對敵方造成 35 點傷害。",
            baseAtk=1.2,
            baseDef=0.9
        )

    def passive(self, user, targets, env):
        # 已在 damege_taken 中實作
        pass
    def damage_taken(self, user, target, env, dmg):
        if random.random() < 0.25:

            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「冷箭」觸發！"))
            env.deal_damage(user, target, 35, can_be_blocked=True)
    
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 27:
            # 技能 27 => 續戰：造成35傷害，每次連續使用攻擊技能時多增加15點傷害。
            times_used = user.get("skills_used", {}).get(skill_id, 0)
            dmg = (35 + (15 * times_used)) * self.baseAtk 
            
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            user["skills_used"][skill_id] = times_used + 1
        elif skill_id == 28:
            # 技能 28 => 埋伏：3回合內，提升20%防禦。
            def_buff = DefenseMultiplier(multiplier=1.2, duration=3, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)

        elif skill_id == 29:
            # 技能 29 => 荒原：消耗15點生命力，免疫一回合的傷害。
            if user["hp"] > 15:
                env.deal_healing(user, 15,self_mutilation = True)
                immune_damage = ImmuneDamage(duration=1, stackable=False)
                env.apply_status(user, immune_damage)
            else:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 嘗試使用「荒原」，但血量不足。"))


class ElementalMage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=10,
            name="元素法師",
            base_hp=228,
            passive_name = "元素之力",
            passive_desc="攻擊時 30% 機率造成(麻痺、冰凍、燃燒)其中之一",
            baseAtk=1.45,
            baseDef=0.95
        )

    def passive(self, user, targets, env):
        # 元素之力：攻擊時30%造成(麻痺、冰凍、燃燒)其中之一
        if random.random() < 0.3:
            effect = random.choice([Burn(duration=3, stacks=1), Freeze(duration=3, stacks=1), Paralysis(duration=2)])

            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「元素之力」觸發！"))
            env.apply_status(targets[0], effect)
    
    def on_turn_end(self, user, targets, env,id):
        super().on_turn_end(user, targets, env, id)
        pass
    
    def damage_taken(self, user, target, env, dmg):
        # if 自身有雷霆護甲
        if user["effect_manager"].get_effects("雷霆護甲"):
            # 30% 機率直接麻痺敵人
            if random.random() < 0.3:
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「雷霆護甲」觸發！"))
                env.apply_status(target, Paralysis(duration=2))
    
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == 30:
            # 技能 30 => 雷霆護甲：2回合內，受到傷害15%機率直接麻痺敵人。
            def_buff = DefenseMultiplier(multiplier=1.5, duration=2, stackable=False,source=skill_id)
            env.apply_status(user, def_buff)
            # 雷霆護甲", "2 回合內，受到傷害時有 30% 機率直接麻痺敵人，並增加 50% 防禦力，回復最大生命的 5% ", 'effect'))
            heal = user["max_hp"] * 0.05
            env.deal_healing(user, heal)
            # 
            teff = Track(name = "雷霆護甲",duration=2,stacks=1,source=skill_id,stackable=False,max_stack=99)
            # add effect
            env.apply_status(user,teff)

        elif skill_id == 31:
            # 技能 31 => 凍燒雷：造成55點傷害，每層麻痺、冰凍、燃燒，額外造成20點傷害
            dmg = 55 * self.baseAtk 
            target = targets[0]   
   
            # 計算所有異常狀態的堆疊數總和
            total_layers = 0
            for effects in target["effect_manager"].active_effects.values():
                for eff in effects:
                    if eff.name in ["麻痺", "凍結", "燃燒"]:
                        total_layers += eff.stacks
            extra_dmg = 20 * total_layers + dmg  
            env.deal_damage(user, target, extra_dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == 32:
            # 技能 32 => 造成15傷害，50% 機率使敵方暈眩 1~3 回合。
            dmg = 35 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            target = targets[0]
            # 70% 2 25% 3 10% 4
            para_duration = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1], k=1)[0]
            if random.random() < 0.35:
                stun_effect = Paralysis(duration=para_duration)
                env.apply_status(target, stun_effect)
            self.passive(user, targets, env)
       
class HuangShen(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=11,  # 確保職業ID唯一
            name="荒神",
            base_hp=230,
            baseAtk=1.18,
            baseDef=1.3,
            passive_name="枯萎之刃",
            passive_desc="隨著造成傷害次數增加，攻擊時額外進行隨機追打，每造成兩次傷害增加一次最高追打機會，追打造成敵方當前生命的5%血量；額外追打不會累積傷害次數。",
        )

    def passive(self, user, targets, env):
        bonus = user.get("skills_used", {}).get(33, 0) // 2
        if bonus > 0:
            for i in range(bonus):
                env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「枯萎之刃」觸發！"))
                env.deal_damage(user, targets[0], int(targets[0]["hp"] * 0.05), can_be_blocked=True)
    
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 33:
            # "枯骨的永恆", "隨機造成 1 ~ 3 次傷害。
            user["skills_used"][skill_id] = user.get("skills_used", {}).get(skill_id, 0)
            times = random.randint(1, 3)
            for i in range(times):
                dmg = 30 * self.baseAtk * (1-i*0.25)
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                self.passive(user, targets, env)
                user["skills_used"][skill_id] += 1
                
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
            # 生命逆流", "回復造成傷害次數300%的血量，冷卻3回合", 'damage'),3)
            heal_amount = user.get("skills_used", {}).get(33, 0) * 5
            env.deal_healing(user, heal_amount)
            
class GodOfStar(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=12,
            name="星神",
            base_hp=295,
            passive_name="天啟星盤",
            passive_desc="星神在戰鬥中精通增益與減益效果的能量運用。每當場上有一層「能力值增益」或「減益」效果時，每回合會額外對敵方造成 5 點傷害 並恢復 5 點生命值。",
            baseAtk=1.08,
            baseDef=1.08
        )

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
        env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的被動技能「天啟星盤」觸發！"))
        return bounous_damage, bounous_heal
#  sm.add_skill(Skill(36, "光輝流星", "對敵方單體造成 15 點傷害，並隨機為自身附加以下一種增益效果，持續 3 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。", 'damage'))
# sm.add_skill(Skill(37, "災厄隕星", "為自身恢復 15 點生命值，並隨機為敵方附加以下一種減益效果，持續 3 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。", 'damage'))
# sm.add_skill(Skill(38, "虛擬創星圖", "對敵方單體造成 45 點傷害。", 'damage'))
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        dmg, heal = self.passive(user, targets, env)
        if skill_id == 36:
            # 技能 36 => 對單體造成30點傷害
            dmg += 25 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            # 隨機為自身附加以下一種增益效果，持續 3 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。
            buff = random.choice([DamageMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id), DefenseMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id), HealMultiplier(multiplier=1.05, duration=3, stackable=False,source=skill_id)])
            env.apply_status(user, buff)
            env.deal_healing(user, heal)
        elif skill_id == 37:
            # 技能 37 => 為自身恢復 15 點生命值
            heal += 25
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            env.deal_healing(user, heal)
            # 隨機為敵方附加以下一種減益效果，持續 3 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。
            debuff = random.choice([DamageMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id), DefenseMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id), HealMultiplier(multiplier=0.95, duration=3, stackable=False,source=skill_id)])
            env.apply_status(targets[0], debuff)
        elif skill_id == 38:
            # 技能 38 => 對敵方單體造成 45 點傷害
            dmg *=1.5
            heal *=1.5
            env.add_event(event = BattleEvent(type="text",text=f"{self.name} 的技能「虛擬創星圖」強化了天啟星盤的力量，增加了傷害和回復。"))
            dmg += 50 * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            env.deal_healing(user, heal)