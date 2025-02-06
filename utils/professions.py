# professions.py

# module import
from .status_effects import (
    Burn, Poison, Freeze,
    DamageMultiplier, DefenseMultiplier, HealMultiplier, HealthPointRecover, Paralysis,
    ImmuneDamage, ImmuneControl, BleedEffect, Track
)
from .skills import SkillManager, sm
from .profession_var import * 
from .battle_event import BattleEvent

# others import
import random
from .profession_var import *

def build_professions():
    return [
        Paladin(PROFESSION_VAR = PALADIN_VAR),
        Mage(PROFESSION_VAR = MAGE_VAR),
        Assassin(PROFESSION_VAR = ASSASSIN_VAR),
        Archer(PROFESSION_VAR = ARCHER_VAR),
        Berserker(PROFESSION_VAR = BERSERKER_VAR),
        DragonGod(PROFESSION_VAR = DRAGONGOD_VAR),
        BloodGod(PROFESSION_VAR = BLOODGOD_VAR),
        SteadfastWarrior(PROFESSION_VAR = STEADFASTWARRIOR_VAR),
        Devour(PROFESSION_VAR = DEVOUR_VAR),
        Ranger(PROFESSION_VAR = RANGER_VAR),
        ElementalMage(PROFESSION_VAR = ELEMENTALMAGE_VAR),
        HuangShen(PROFESSION_VAR = HUANGSHEN_VAR),
        GodOfStar(PROFESSION_VAR = GODOFSTAR_VAR)
    ]

class BattleProfession:

    def __init__(self, profession_id = -1, name = '初始化失敗', base_hp=-1, passive_name='初始化失敗', passive_desc="初始化失敗", baseAtk=1.0, baseDef=1.0,PROFESSION_VAR=None):
        self.profession_id = profession_id
        self.name = name
        self.base_hp = base_hp
        self.max_hp = base_hp
        self.passive_name = passive_name
        self.passive_desc = passive_desc
        self.baseAtk = baseAtk
        self.baseDef = baseDef
        self.default_passive_id = (profession_id * -1)-1
        self.PROFESSION_VAR = PROFESSION_VAR
    def get_available_skill_ids(self, cooldowns: dict):
        # get cooldowns
        ok_skills = []
        if cooldowns[0] == 0:
            ok_skills.append(self.profession_id * 4)
        if cooldowns[1] == 0:
            ok_skills.append(self.profession_id*4 + 1)
        if cooldowns[2] == 0:
            ok_skills.append(self.profession_id*4 + 2)
        if cooldowns[3] == 0:
            ok_skills.append(self.profession_id*4 + 3)
        
        # return is real skill id
        return ok_skills

    def damage_taken(self, user, targets, env, dmg):
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
        cooldowns_skill_id = skill_id - self.profession_id * 4
        env.add_event(user=user, event=BattleEvent(type="skill", appendix={
                      "skill_id": skill_id, "relatively_skill_id": cooldowns_skill_id}))
        # check cooldown
        # get skill id's cooldown

        if skill_id in self.get_available_skill_ids(user["cooldowns"]):
            # set cooldown
            local_skill_id = skill_id - self.profession_id * 4
            user["cooldowns"][local_skill_id] = sm.get_skill_cooldown(skill_id)
            if sm.get_skill_cooldown(skill_id) > 0:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」進入冷卻 {sm.get_skill_cooldown(skill_id)} 回合。"))

            return -1
        else:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」冷卻中。"))
            return -1

class Paladin(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=0)
        self.PALADIN_VAR = PROFESSION_VAR
        self.name = "聖騎士"
        self.base_hp = self.PALADIN_VAR['PALADIN_BASE_HP'][0]
        self.passive_name = "聖光"
        self.passive_desc = f"攻擊時，{int(self.PALADIN_VAR['PALADIN_PASSIVE_TRIGGER_RATE'][0]*100)}% 機率恢復最大血量的 {int(self.PALADIN_VAR['PALADIN_PASSIVE_HEAL_RATE'][0]*100)}% 的生命值，回復超出最大生命時，對敵方造成 {int(self.PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0]*100)}% 的回復傷害"
        self.baseAtk = self.PALADIN_VAR['PALADIN_BASE_ATK'][0]
        self.baseDef = self.PALADIN_VAR['PALADIN_BASE_DEF'][0]
        
        self.heal_counts = {}
        
    def damage_taken(self, user, targets, env, dmg):
        super().damage_taken(user, targets, env, dmg)
        #  這邊處理的是技能 3 的效果
        if user["effect_manager"].has_effect("決一死戰") and user["hp"] == 0:
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 整理好自己的狀態，並開始進行最後的決戰。"))
            user["hp"] = int(user["max_hp"] * self.PALADIN_VAR['PALADIN_SKILL_3_MAX_HP_HEAL'][0])
            # b
            buff = DamageMultiplier(multiplier=self.PALADIN_VAR['PALADIN_SKILL_3_DAMAGE_BUFF'][0], duration=3, stackable=False, source=self.profession_id * 4 + 3)
            debuff = DefenseMultiplier(multiplier=self.PALADIN_VAR['PALADIN_SKILL_3_DEFENSE_DEBUFF'][0], duration=3, stackable=False, source=self.profession_id * 4 + 3)
            env.apply_status(user, buff)
            env.apply_status(user, debuff)
            user["effect_manager"].remove_all_effects("決一死戰")
            


    def passive(self, user, targets, env):
        # 被動技能：15%機率回復最大血量的15%，超出部分造成100%回復傷害
        super().passive(user, targets, env)

        if random.random() < self.PALADIN_VAR['PALADIN_PASSIVE_TRIGGER_RATE'][0]:
            heal_amount = int(
                self.max_hp * self.PALADIN_VAR['PALADIN_PASSIVE_HEAL_RATE'][0])
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 聖光觸發，恢復了血量。"))
            env.deal_healing(
                user, heal_amount, rate=self.PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == self.profession_id * 4:
            # 技能 0 => 對單體造成40點傷害
            dmg = self.baseAtk * self.PALADIN_VAR['PALADIN_SKILL_0_DAMAGE'][0]
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == self.profession_id * 4 + 1:
            # 技能 1 => 本回合迴避攻擊，回復10點血量。冷卻3回合。
            user["is_defending"] = True
            heal_amount = self.PALADIN_VAR['PALADIN_SKILL_1_HEAL'][0]

            env.deal_healing(
                user, heal_amount, rate=self.PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])

        elif skill_id == self.profession_id * 4 + 2:
            # 技能 2 => 恢復血量，第一次50, 第二次35, 第三次及以後15
            times_healed = user.get("times_healed", 0)
            if times_healed == 0:
                heal_amount = self.PALADIN_VAR['PALADIN_SKILL_2_FIRST_HEAL'][0]
            elif times_healed == 1:
                heal_amount = self.PALADIN_VAR['PALADIN_SKILL_2_SECOND_HEAL'][0]
            else:
                heal_amount = self.PALADIN_VAR['PALADIN_SKILL_2_MORE_HEAL'][0]

            env.deal_healing(
                user, heal_amount, rate=self.PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])
            user["times_healed"] = times_healed + 1
        
        elif skill_id == self.profession_id * 4 + 3:
            # 2回合內若受到致死傷害時，回復至25%最大血量。同時 3 回合間攻擊力提升100%，防禦力降低50%。
            eff = Track(name = '決一死戰', duration=2, stacks=1, source=skill_id, stackable=False)
            env.apply_status(user, eff)
            # 剩餘部分在on_attack_end中處理
                 
class Mage(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=1)
        self.MAGE_VAR = PROFESSION_VAR
        self.name = "法師"
        self.base_hp = self.MAGE_VAR['MAGE_BASE_HP'][0]
        self.passive_name = "魔力充盈"
        self.passive_desc = f"攻擊造成異常狀態時，{int(self.MAGE_VAR['MAGE_PASSIVE_TRIGGER_RATE'][0]*100)}% 機率額外疊加一層異常狀態（燃燒或冰凍）。"
        self.baseAtk = self.MAGE_VAR['MAGE_BASE_ATK'][0]
        self.baseDef = self.MAGE_VAR['MAGE_BASE_DEF'][0]

    def passive(self, user, targets, env):
        # 被動技能：攻擊造成異常狀態時，有機率額外疊加異常狀態
        if random.random() < self.MAGE_VAR['MAGE_PASSIVE_TRIGGER_RATE'][0]:
            extra_status = random.choice([
                Burn(duration=3, stacks=1),
                Freeze(duration=3, stacks=1)
            ])
            target = targets[0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「魔力充盈」觸發，對 {target['profession'].name} 施加了額外的 {extra_status.name}。"))
            env.apply_status(target, extra_status)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == self.profession_id * 4:
            # 技能 0：火焰之球
            dmg = self.MAGE_VAR['MAGE_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            burn_effect = Burn(duration=3, stacks=1)
            env.apply_status(targets[0], burn_effect)
            self.passive(user, targets, env)

        elif skill_id == self.profession_id * 4 + 1:
            # 技能 1：冰霜箭
            dmg = self.MAGE_VAR['MAGE_SKILL_1_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            freeze_effect = Freeze(duration=3, stacks=1)
            env.apply_status(targets[0], freeze_effect)
            self.passive(user, targets, env)

        elif skill_id == self.profession_id * 4 + 2:
            # 技能 2：全域爆破
            base_dmg = self.MAGE_VAR['MAGE_SKILL_2_BASE_DAMAGE'][0] * self.baseAtk
            total_layers = sum(
                eff.stacks for effects in targets[0]["effect_manager"].active_effects.values()
                for eff in effects if isinstance(eff, (Burn, Freeze))
            )
            dmg = base_dmg + \
                self.MAGE_VAR['MAGE_SKILL_2_STATUS_MULTIPLIER'][0] * total_layers
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            targets[0]["effect_manager"].remove_all_effects("燃燒")
            targets[0]["effect_manager"].remove_all_effects("凍結")
            
        elif skill_id == self.profession_id*4 +3:
            # 技能3 無詠唱魔法
            base_dmg = self.MAGE_VAR['MAGE_SKILL_3_BASE_DAMAGE'][0] * self.baseAtk
            total_layers = sum(
                eff.stacks for effects in targets[0]["effect_manager"].active_effects.values()
                for eff in effects if isinstance(eff, (Burn, Freeze))
            )
            dmg = base_dmg + \
                self.MAGE_VAR['MAGE_SKILL_3_STATUS_MULTIPLIER'][0] * total_layers
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            targets[0]["effect_manager"].remove_all_effects("燃燒")
            targets[0]["effect_manager"].remove_all_effects("凍結")
                
class Assassin(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=2)
        self.ASSASSIN_VAR = PROFESSION_VAR
        self.name = "刺客"
        self.base_hp = self.ASSASSIN_VAR['ASSASSIN_BASE_HP'][0]
        self.passive_name = "刺殺"
        self.passive_desc = f"攻擊時額外造成敵方當前 {int(self.ASSASSIN_VAR['ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE'][0] * 100)}% 生命值的傷害，並且 {int(self.ASSASSIN_VAR['ASSASSIN_PASSIVE_TRIGGER_RATE'][0] * 100)}% 機率對敵方造成中毒狀態。"
        self.baseAtk = self.ASSASSIN_VAR['ASSASSIN_BASE_ATK'][0]
        self.baseDef = self.ASSASSIN_VAR['ASSASSIN_BASE_DEF'][0]
        

    def passive(self, targets, damage, env):
        # 被動技能：額外傷害和中毒效果
        if random.random() < self.ASSASSIN_VAR['ASSASSIN_PASSIVE_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"刺客對敵方造成額外中毒效果！"))
            effect = Poison(duration=3, stacks=1)
            env.apply_status(targets[0], effect)
        return damage + int(targets[0]["hp"] * self.ASSASSIN_VAR['ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE'][0])

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == self.profession_id * 4:
            # 技能 0：致命暗殺
            dmg = self.ASSASSIN_VAR['ASSASSIN_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(targets, dmg, env)
            if random.random() < self.ASSASSIN_VAR['ASSASSIN_SKILL_0_CRIT_RATE'][0]:
                env.add_event(event=BattleEvent(type="text", text=f"擊中要害！"))
                dmg *= 2
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

        elif skill_id == self.profession_id * 4 + 1:
            # 技能 1：毒爆
            target = targets[0]
            total_layers = 0
            for effects in target["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, Poison):
                        total_layers += eff.stacks
            if total_layers > 0:
                env.add_event(event=BattleEvent(type="text", text=f"引爆了中毒效果！"))
                dmg = self.ASSASSIN_VAR['ASSASSIN_SKILL_1_DAMAGE_PER_LAYER'][0] * total_layers
                dmg = self.passive(targets, dmg, env)
                heal = self.ASSASSIN_VAR['ASSASSIN_SKILL_1_HEAL_PER_LAYER'][0] * total_layers
                env.deal_damage(user, target, dmg, can_be_blocked=True)
                env.deal_healing(user, heal)
                target["effect_manager"].remove_all_effects("中毒")
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"無法引爆中毒效果，對方並未中毒。"))

        elif skill_id == self.profession_id * 4 + 2:
            # 技能 2：毒刃襲擊
            dmg = self.ASSASSIN_VAR['ASSASSIN_SKILL_2_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(targets, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

            # 計算隨機疊加的中毒層數
            weights = [
                self.ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_1_WEIGHT'][0],
                self.ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_2_WEIGHT'][0],
                self.ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_3_WEIGHT'][0],
                self.ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_4_WEIGHT'][0],
                self.ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_5_WEIGHT'][0]
            ]
            add_stacks = random.choices(
                [1, 2, 3, 4, 5], weights=weights, k=1)[0]
            effect = Poison(duration=3, stacks=add_stacks)
            env.apply_status(targets[0], effect)
        elif skill_id == self.profession_id * 4 + 3:
            # get posion stacks and duration
            # 致命藥劑 根據敵方當前中毒層數降低治癒力，敵方每層中毒降低30%的治癒力。持續時間等同於中毒的剩餘持續時間
            total_layers = 0
            for effects in targets[0]["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, Poison):
                        total_layers += eff.stacks
            total_duration = 0
            for effects in targets[0]["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, Poison):
                        total_duration += eff.duration
            debuff = HealMultiplier(multiplier= self.ASSASSIN_VAR['ASSASSIN_SKILL_3_DEBUFF_MULTIPLIER'][0], duration=total_duration,stacks=total_layers,max_stack=5 ,stackable=True, source=skill_id)
            env.apply_status(targets[0], debuff)

class Archer(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=3)
        self.ARCHER_VAR = PROFESSION_VAR
        self.name = "弓箭手"
        self.base_hp = self.ARCHER_VAR['ARCHER_BASE_HP'][0]
        self.passive_name = "鷹眼"
        self.passive_desc = f"攻擊時 {int(self.ARCHER_VAR['ARCHER_PASSIVE_BASE_TRIGGER_RATE'][0] * 100)}% 機率造成 2 倍傷害；敵方防禦基值越高時，額外增加觸發機率。"
        self.baseAtk = self.ARCHER_VAR['ARCHER_BASE_ATK'][0]
        self.baseDef = self.ARCHER_VAR['ARCHER_BASE_DEF'][0]
        

    def passive(self, env, dmg, tar):
        target = tar[0]
        prob = self.ARCHER_VAR['ARCHER_PASSIVE_BASE_TRIGGER_RATE'][0]
        toatal_def = target["profession"].baseDef * target["defend_multiplier"]
        if toatal_def > 1:
            prob += (toatal_def - 1) * self.ARCHER_VAR['ARCHER_PASSIVE_TRIGGER_RATE_BONUS'][0]
            prob = min(prob, self.ARCHER_VAR['ARCHER_PASSIVE_TRIGGER_RATE_MAX'][0])
        if random.random() < prob:
            env.add_event(event=BattleEvent(
                type="text", text=f"被動技能「鷹眼」觸發，攻擊造成{self.ARCHER_VAR['ARCHER_PASSIVE_DAMAGE_MULTIPLIER'][0]} 倍傷害！"))
            return dmg * self.ARCHER_VAR['ARCHER_PASSIVE_DAMAGE_MULTIPLIER'][0]
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == self.profession_id * 4:
            # 技能 0：五連矢
            dmg = self.ARCHER_VAR['ARCHER_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg /= 5 
            for i in range(5):
                tmp = dmg
                dmg = self.passive(env, dmg, targets)
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                if i==0:
                    def_buff = DefenseMultiplier(
                    multiplier=self.ARCHER_VAR['ARCHER_SKILL_0_DEFENSE_DEBUFF'][0],
                    duration=self.ARCHER_VAR['ARCHER_SKILL_0_DURATION'][0],
                    stackable=False,
                    source=skill_id)
                    env.apply_status(targets[0], def_buff)
                dmg = tmp
                    

        elif skill_id == self.profession_id * 4 + 1:
            # 技能 1：箭矢補充
            if random.random() < self.ARCHER_VAR['ARCHER_SKILL_1_SUCESS_RATIO'][0]:
                dmg_multiplier = self.ARCHER_VAR['ARCHER_SKILL_1_DAMAGE_MULTIPLIER'][0]
                dmg_buff = DamageMultiplier(
                    multiplier=dmg_multiplier,
                    duration=self.ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.add_event(event=BattleEvent(
                    type="text", text=f"箭矢補充成功，攻擊力大幅提升！"))
                env.apply_status(user, dmg_buff)
            else:
                def_multiplier = self.ARCHER_VAR['ARCHER_SKILL_1_DEFENSE_MULTIPLIER'][0]
                def_debuff = DefenseMultiplier(
                    multiplier=def_multiplier,
                    duration=self.ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.add_event(event=BattleEvent(
                    type="text", text=f"箭矢補充失敗，防禦力下降！"))
                env.apply_status(user, def_debuff)

        elif skill_id == self.profession_id * 4 + 2:
            # 技能 2：吸血箭
            dmg = self.ARCHER_VAR['ARCHER_SKILL_2_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(env, dmg, targets)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            heal_amount = self.ARCHER_VAR['ARCHER_SKILL_2_HEAL'][0]
            env.deal_healing(user, heal_amount)
        elif skill_id == self.profession_id * 4 + 3:
            dmg = self.ARCHER_VAR['ARCHER_SKILL_3_DAMAGE'][0] * self.baseAtk / 5
            for i in range(5):
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True, ignore_defense=self.ARCHER_VAR['ARCHER_SKILL_3_IGN_DEFEND'][0])

class Berserker(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=4)
        self.BERSERKER_VAR = PROFESSION_VAR
        self.name = "狂戰士"
        self.base_hp = self.BERSERKER_VAR['BERSERKER_BASE_HP'][0]
        self.passive_name = "狂暴"
        self.passive_desc = f"若自身血量低於 {int(self.BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD'][0] * 100)}%，攻擊增加失去生命值的 {int(self.BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0] * 100)}% 的傷害。"
        self.baseAtk = self.BERSERKER_VAR['BERSERKER_BASE_ATK'][0]
        self.baseDef = self.BERSERKER_VAR['BERSERKER_BASE_DEF'][0]

    def passive(self, user, dmg, env):
        if user["hp"] < (user["max_hp"] * self.BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD'][0]):
            loss_hp = user["max_hp"] - user["hp"]
            dmg += loss_hp * self.BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「狂暴」觸發，攻擊時增加了 {int(loss_hp * self.BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0])} 點傷害。"))
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        
        if skill_id == self.profession_id * 4:  # 對應 BERSERKER_SKILL_0
            dmg = self.BERSERKER_VAR['BERSERKER_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(user, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self_mutilation = dmg * self.BERSERKER_VAR['BERSERKER_SKILL_0_SELF_MUTILATION_RATE'][0]
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 受到反噬。"))
            env.deal_healing(user, self_mutilation, self_mutilation=True)
            
            # check if user 吸血狀態
            if user["effect_manager"].has_effect("吸血"):
                heal = self.BERSERKER_VAR['BERSERKER_SKILL_3_BASE_HEAL_RATE'][0] * dmg
                max_mul = self.BERSERKER_VAR['BERSERKER_SKILL_3_BONUS_RATE'][0]
                hp_rate = user["hp"] / user["max_hp"]
                bonus_mul = (1-hp_rate) * max_mul
                heal *= (1 + bonus_mul)
                env.deal_healing(user, heal)
                
        elif skill_id == self.profession_id *4 +1:  # 對應 BERSERKER_SKILL_1
            if user["hp"] > self.BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0]:
                env.deal_healing(user, self.BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0], self_mutilation=True)
                heal_effect = HealthPointRecover(
                    hp_recover=self.BERSERKER_VAR['BERSERKER_SKILL_1_HEAL_PER_TURN'][0],
                    duration=self.BERSERKER_VAR['BERSERKER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id,
                    env=env
                )
                env.apply_status(user, heal_effect)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「熱血」，但血量不足。"))

        elif skill_id == self.profession_id * 4 +2:  # 對應 BERSERKER_SKILL_2
            if user["hp"] > self.BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0]:
                env.deal_healing(user, self.BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0], self_mutilation=True)
                immune_control = ImmuneControl(duration=self.BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0], stackable=False)
                env.apply_status(user, immune_control)
                def_buff = DefenseMultiplier(
                    multiplier=self.BERSERKER_VAR['BERSERKER_SKILL_2_DEFENSE_BUFF'][0],
                    duration=self.BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.apply_status(user, def_buff)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「血怒」，但血量不足。"))
        elif skill_id == self.profession_id * 4 +3:  # 對應 BERSERKER_SKILL_3
            tr = Track(name = "吸血", duration = 3, stacks = 1, source = skill_id, stackable = False)
            env.apply_status(user, tr)
            
class DragonGod(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=5)
        self.DRAGONGOD_VAR = PROFESSION_VAR
        self.name = "龍神"
        self.base_hp = self.DRAGONGOD_VAR['DRAGONGOD_BASE_HP'][0]
        self.passive_name = "龍血"
        self.passive_desc = f"每回合疊加一個龍神狀態，龍神狀態每層增加 {int((self.DRAGONGOD_VAR['DRAGONGOD_PASSIVE_ATK_MULTIPLIER'][0] - 1) * 100)}% 攻擊力、增加 {int((self.DRAGONGOD_VAR['DRAGONGOD_PASSIVE_DEF_MULTIPLIER'][0] - 1) * 100)}% 防禦力。"
        self.baseAtk = self.DRAGONGOD_VAR['DRAGONGOD_BASE_ATK'][0]
        self.baseDef = self.DRAGONGOD_VAR['DRAGONGOD_BASE_DEF'][0]

    def on_turn_start(self, user, targets, env, id):
        super().on_turn_start(user, targets, env, id)
        # 每回合觸發被動技能，疊加龍血狀態

        if user["effect_manager"].has_effect("預借",source = self.profession_id*4 +3):
            env.add_event(event=BattleEvent(
                type="text", text=f"預借效果中，不會疊加龍血狀態。"))
            return
        
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「龍血」觸發！"))

        atk_buff = DamageMultiplier(
            multiplier=self.DRAGONGOD_VAR['DRAGONGOD_PASSIVE_ATK_MULTIPLIER'][0],
            duration=99,
            stacks=1,
            source=self.default_passive_id,
            stackable=True,
            max_stack=99
        )
        def_buff = DefenseMultiplier(
            multiplier=self.DRAGONGOD_VAR['DRAGONGOD_PASSIVE_DEF_MULTIPLIER'][0],
            duration=99,
            stacks=1,
            source=self.default_passive_id,
            stackable=True,
            max_stack=99
        )
        track = Track(
            name="龍血",
            duration=99,
            stacks=1,
            source=self.default_passive_id,
            stackable=True,
            max_stack=99
        )
        env.apply_status(user, atk_buff)
        env.apply_status(user, def_buff)
        env.apply_status(user, track)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == self.profession_id *4:  # 對應 DRAGONGOD_SKILL_0
            base_dmg = self.DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BASE_DAMAGE'][0] * self.baseAtk
            dragon_soul_effect = user["effect_manager"].get_effects("龍血")[0]
            stacks = dragon_soul_effect.stacks if dragon_soul_effect else 0
            bonus_dmg = stacks * self.DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK'][0]
            dmg = base_dmg + bonus_dmg
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

        elif skill_id == self.profession_id*4 +1:  # 對應 DRAGONGOD_SKILL_1
            heal_amount = self.DRAGONGOD_VAR['DRAGONGOD_SKILL_1_HEAL_AMOUNT'][0]
            env.deal_healing(user, heal_amount)
            bleed_effect = HealthPointRecover(
                hp_recover=self.DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_PER_TURN'][0],
                duration=self.DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_DURATION'][0],
                stackable=False,
                source=skill_id,
                env=env,
                self_mutilation=True
            )
            env.apply_status(user, bleed_effect)

        elif skill_id == self.profession_id*4 +2:  # 對應 DRAGONGOD_SKILL_2
            dragon_soul_effect = user["effect_manager"].get_effects("龍血")[0]
            stacks = dragon_soul_effect.stacks if dragon_soul_effect else 0
            consume_ratio = self.DRAGONGOD_VAR['DRAGONGOD_SKILL_2_STACK_CONSUMPTION'][0]
            consume_stack = int(stacks * consume_ratio)
            if consume_stack > 0:
                damage = consume_stack * self.DRAGONGOD_VAR['DRAGONGOD_SKILL_2_DAMAGE_PER_STACK'][0]
                env.add_event(event=BattleEvent(
                    type="text", text=f"「神龍燎原」消耗了 {consume_stack} 層龍神狀態。"))
                env.deal_damage(user, targets[0], damage, can_be_blocked=True)
                finstack = dragon_soul_effect.stacks - consume_stack
                   
                env.set_status(user, "攻擊力" , finstack,source = self.default_passive_id)
                env.set_status(user, "防禦力" , finstack,source = self.default_passive_id)
                dragon_soul_effect.stacks = finstack
                         
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「神龍燎原」，但沒有足夠的龍神狀態。"))
        
        elif skill_id ==self.profession_id*4 +3:
            # 龍神層數立即疊加 4 層，但在接下來的4回合內不會疊加層數。
            tr = Track(name = "預借", duration = self.DRAGONGOD_VAR['DRAGONGOD_SKILL_3_ADD_STACK'][0]+1,stacks=1,max_stack=1, source = skill_id, stackable = False)

            for _ in range(self.DRAGONGOD_VAR['DRAGONGOD_SKILL_3_ADD_STACK'][0]):
                self.on_turn_start(user, targets, env, -1)
            env.apply_status(user, tr)
            
            
class BloodGod(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=6)
        self.BLOODGOD_VAR = PROFESSION_VAR
        self.name = "血神"
        self.base_hp = self.BLOODGOD_VAR['BLOODGOD_BASE_HP'][0]
        self.passive_name = "血脈"
        self.passive_desc = f"每回合會累積所受傷害至血脈裡面，所受傷害累積越多會降低血脈的強度。每受到最大血量的 {int(self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0] * 100)}% 傷害，則自身攻擊力、防禦力、治癒力降低 {int((1 - self.BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0]) * 100)}%。"
        self.baseAtk = self.BLOODGOD_VAR['BLOODGOD_BASE_ATK'][0]
        self.baseDef = self.BLOODGOD_VAR['BLOODGOD_BASE_DEF'][0]

    def passive(self, user, targets, env):
        pass

    def on_turn_end(self, user, targets, env, id):
        super().on_turn_end(user, targets, env, id)
        if 'total_accumulated_damage' in user['private_info']:
            user['private_info']['total_accumulated_damage'] += user['accumulated_damage']
        else:
            user['private_info']['total_accumulated_damage'] = user['accumulated_damage']

        # 每受到最大血量的指定比例傷害，降低攻擊、防禦、治癒力
        threshold = user['max_hp'] * self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]
        stack = int(user['private_info']['total_accumulated_damage'] / threshold)

        if stack > 0:
            if user["effect_manager"].has_effect("攻擊力", source=self.default_passive_id):
                eff = user["effect_manager"].get_effects("攻擊力", source=self.default_passive_id)[0]
                if eff.stacks != stack:
                    env.set_status(user, "攻擊力", stack, source=self.default_passive_id)
                    env.set_status(user, "防禦力", stack, source=self.default_passive_id)
                    env.set_status(user, "治癒力", stack, source=self.default_passive_id)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 血脈混濁，降低了自身的攻擊力、防禦力、治癒力。"))
                atffect = DamageMultiplier(
                    multiplier=self.BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
                    duration=99, stacks=stack,
                    source=self.default_passive_id, stackable=True, max_stack=99
                )
                deffect = DefenseMultiplier(
                    multiplier=self.BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
                    duration=99, stacks=stack,
                    source=self.default_passive_id, stackable=True, max_stack=99
                )
                heffect = HealMultiplier(
                    multiplier=self.BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
                    duration=99, stacks=stack,
                    source=self.default_passive_id, stackable=True, max_stack=99
                )
                env.apply_status(user, atffect)
                env.apply_status(user, deffect)
                env.apply_status(user, heffect)
        elif stack == 0 and user["effect_manager"].has_effect("攻擊力", source=self.default_passive_id):
            user["effect_manager"].remove_all_effects("攻擊力", source=self.default_passive_id)
            user["effect_manager"].remove_all_effects("防禦力", source=self.default_passive_id)
            user["effect_manager"].remove_all_effects("治癒力", source=self.default_passive_id)

    def damage_taken(self, user, targets, env, dmg):
        super().damage_taken(user, targets, env, dmg)
        if 'total_accumulated_damage' in user['private_info']:
            user['private_info']['total_accumulated_damage'] += user['accumulated_damage']
        else:
            user['private_info']['total_accumulated_damage'] = user['accumulated_damage']
        
        if user["effect_manager"].has_effect("轉生之印"):
            eff = user["effect_manager"].get_effects("轉生之印")[0]
            if eff.stacks > 0 and user["hp"] == 0:
                eff.stacks -= 1
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的「轉生之印」觸發，神秘的力量使自身免於死亡。"))
                user["hp"] = int(user["max_hp"] * self.BLOODGOD_VAR['BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO'][0])
                user["private_info"]["total_accumulated_damage"] += dmg * self.BLOODGOD_VAR['BLOODGOD_SKILL_2_BLOOD_ACCUMULATION_MULTIPLIER'][0]
                user["effect_manager"].remove_all_effects("轉生之印")

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == self.profession_id * 4:
            dmg = self.BLOODGOD_VAR['BLOODGOD_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            bleed_effect = BleedEffect(duration=self.BLOODGOD_VAR['BLOODGOD_SKILL_0_BLEED_DURATION'][0], stacks=1)
            env.apply_status(targets[0], bleed_effect)

            bleed_effects = targets[0]["effect_manager"].get_effects("流血")
            if bleed_effects:
                stack = bleed_effects[0].stacks
                heal_amount = stack * self.BLOODGOD_VAR['BLOODGOD_SKILL_0_HEAL_PER_BLEED_STACK'][0]
                env.deal_healing(user, heal_amount)

        elif skill_id == self.profession_id * 4 + 1:
            bleed_effects = targets[0]["effect_manager"].get_effects("流血")
            if bleed_effects:
                stack = bleed_effects[0].stacks
                user['private_info']['total_accumulated_damage'] = max(
                    user['private_info']['total_accumulated_damage'] - stack * self.BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER'][0], 0)
                

                if user['private_info']['total_accumulated_damage'] < user['max_hp'] * self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]:
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有完美的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*2):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有上等的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*3):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有普通的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (self.BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*4):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有混濁的血脈，並使敵方流血更加嚴重！"))
                else:
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有拙劣的血脈，並使敵方流血更加嚴重！"))
                    
                heal_amount = stack * self.BLOODGOD_VAR['BLOODGOD_SKILL_1_HEAL_MULTIPLIER'][0]
                env.deal_healing(user, heal_amount)
                env.set_status(targets[0], "流血", stacks=stack * self.BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_STACK_MULTIPLIER'][0])
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「血脈祭儀」，但是流血層數不夠發動血脈祭儀。"))

        elif skill_id == self.profession_id * 4 + 2:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 啟用了久遠的未知印記，神秘的力量開始聚於自身之上。"))
            env.deal_healing(user, int(user["hp"] * self.BLOODGOD_VAR['BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO'][0]), self_mutilation=True)
            eff = Track(name="轉生之印", duration=self.BLOODGOD_VAR['BLOODGOD_SKILL_2_DURATION'][0], stacks=1,
                        source=skill_id, stackable=False)
            env.apply_status(user, eff)
            # 剩下部分在 damage_taken 處理
        elif skill_id == self.profession_id * 4 + 3:
            if 'total_accumulated_damage' not in user['private_info']:

                user['private_info']['total_accumulated_damage'] = user['accumulated_damage']
            # reduce the accumulated damage
            user['private_info']['total_accumulated_damage'] *= (1-self.BLOODGOD_VAR['BLOODGOD_SKILL_3_REDUCE_DAMAGE'][0])
            dam_debuff = DamageMultiplier(multiplier = self.BLOODGOD_VAR['BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER'][0], duration = 99,stacks = 1 ,max_stack= 5 ,stackable = True, source = skill_id)
            def_debuff = DefenseMultiplier(multiplier = self.BLOODGOD_VAR['BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER'][0], duration = 99,stacks = 1 ,max_stack= 5 ,stackable = True, source = skill_id)
            hel_debuff = HealMultiplier(multiplier = self.BLOODGOD_VAR['BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER'][0], duration = 99,stacks = 1 ,max_stack= 5 ,stackable = True, source = skill_id)
            env.apply_status(user, dam_debuff)
            env.apply_status(user, def_debuff)
            env.apply_status(user, hel_debuff)
            # 立即更新血脈
            self.on_turn_end(user, targets, env, -1)
                            

class SteadfastWarrior(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=7)
        self.STEADFASTWARRIOR_VAR = PROFESSION_VAR
        self.name = "剛毅武士"
        self.base_hp = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_HP'][0]
        self.passive_name = "堅韌壁壘"
        self.passive_desc = f"每回合開始時恢復已損生命值的 {int(self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT'][0] * 100)}%。"
        self.baseAtk = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_ATK'][0]
        self.baseDef = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_DEF'][0]

    def on_turn_start(self, user, targets, env, id):
        heal = int((self.max_hp - user["hp"]) * self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT'][0])
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「堅韌壁壘」觸發。"))
        env.deal_healing(user, heal)

    def on_turn_end(self, user, targets, env, id):
        if id == self.profession_id * 4 + 2:
            if user["last_attacker"]:
                dmg = user["last_damage_taken"] * self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER'][0]
                env.deal_damage(user, user["last_attacker"], dmg, can_be_blocked=True)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的被動技能「絕地反擊」沒有對象反擊。"))

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == self.profession_id * 4:
            # 剛毅打擊
            dmg = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            def_buff = DefenseMultiplier(
                multiplier=self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF'][0],
                duration=self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DURATION'][0],
                stackable=False, source=skill_id)
            env.apply_status(targets[0], def_buff)

        elif skill_id == self.profession_id * 4 + 1:
            # 不屈意志
            def_buff = DefenseMultiplier(
                multiplier=self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF'][0],
                duration=self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DURATION'][0],
                stackable=False, source=skill_id)
            env.apply_status(user, def_buff)
            heal_amount = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT'][0]
            actual_heal = env.deal_healing(user, heal_amount)

        elif skill_id == self.profession_id * 4 + 2:
            # 絕地反擊
            pass
        elif skill_id == self.profession_id * 4 + 3:
            dmg = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_DAMAGE'][0] * self.baseAtk
            # check 敵方 buff狀態
            effects_cnt = 0
            effects_copy_1 = list(user["effect_manager"].active_effects.values())
            effects_copy_2 = list(targets[0]["effect_manager"].active_effects.values())
            deleted = 0
            for eff in effects_copy_2:
                for effect in eff:
                    if effect.type == "buff" and effect.multiplier > 1 and deleted == 0:
                        effects_cnt += 1
                        # del this effect
                        targets[0]["effect_manager"].remove_all_effects(effect.name, effect.source)
                        deleted  = 1
                    
            deleted = 0   
            # check 自己的debuff狀態
            for eff in effects_copy_1:
                for effect in eff:
                    if effect.type == "buff" and effect.multiplier < 1 and deleted == 0:
                        effects_cnt += 1
                        # del this effect
                        user["effect_manager"].remove_all_effects(effect.name, effect.source)
                        deleted = 1
            dmg = self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_DAMAGE'][0] * self.baseAtk
            dmg += effects_cnt * self.STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_BONUS_DAMAGE'][0]
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
             

class Devour(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=8)
        self.DEVOUR_VAR = PROFESSION_VAR
        self.name = "鯨吞"
        self.base_hp = self.DEVOUR_VAR['DEVOUR_BASE_HP'][0]
        self.passive_name = "巨鯨"
        self.passive_desc = f"攻擊時會消耗 {int(self.DEVOUR_VAR['DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT'][0] * 100)}% 當前生命值。"
        self.baseAtk = self.DEVOUR_VAR['DEVOUR_BASE_ATK'][0]
        self.baseDef = self.DEVOUR_VAR['DEVOUR_BASE_DEF'][0]

    def passive(self, user, dmg, env):
        # 巨鯨：攻擊時會消耗指定比例當前生命值
        self_damage = int(user["hp"] * self.DEVOUR_VAR['DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT'][0])
        env.deal_healing(user, self_damage, self_mutilation=True)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if 'continous_fail_times' not in user['private_info'] :
            user['private_info']['continous_fail_times'] = 0
    
        if skill_id == self.profession_id * 4:
            # 吞裂
            dmg = self.DEVOUR_VAR['DEVOUR_SKILL_0_DAMAGE'][0] * self.baseAtk
            #  real failure rate = DEVOUR_SKILL_0_FAILURE_RATE + DEVOUR_SKILL_0_FAILURE_RATE * continous_fail_times
            real_failure_rate = self.DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_RATE'][0] - self.DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_RATE'][0] * user['private_info']['continous_fail_times']
            # if user exsit "觸電反應" => failure rate = 0
            if user["effect_manager"].has_effect("觸電反應"):
                real_failure_rate = 0
            if random.random() < real_failure_rate:
                env.add_event(event=BattleEvent(type="text", text=f"{self.name} 的技能「吞裂」使用失敗。"))
                user['private_info']['continous_fail_times'] += 1
            else:
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                user['private_info']['continous_fail_times'] = 0
            self.passive(user, dmg, env)

        elif skill_id == self.profession_id * 4 + 1:
            # 巨口吞世
            if targets[0]["hp"] > user["hp"]:
                dmg = int((user["max_hp"] - user["hp"]) * self.DEVOUR_VAR['DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER'][0])
            else:
                dmg = int(user["hp"] * self.DEVOUR_VAR['DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER'][0])
            env.deal_damage(user, targets[0], dmg * self.baseAtk, can_be_blocked=True)
            self.passive(user, dmg, env)

        elif skill_id == self.profession_id * 4 + 2:
            # 堅硬皮膚
            def_buff = DefenseMultiplier(
                multiplier=self.DEVOUR_VAR['DEVOUR_SKILL_2_DEFENSE_MULTIPLIER'][0],
                duration=self.DEVOUR_VAR['DEVOUR_SKILL_2_DURATION'][0],
                stackable=False,
                source=skill_id
            )
            env.apply_status(user, def_buff)
        elif skill_id == self.profession_id * 4 + 3:
            para = Paralysis(duration=self.DEVOUR_VAR['DEVOUR_SKILL_3_PARALYSIS_DURATION'][0])
            tr = Track(name = "觸電反應", duration = self.DEVOUR_VAR['DEVOUR_SKILL_3_MUST_SUCCESS_DURATAION'][0], source = skill_id, stacks= 1,max_stack= 1,stackable = False)
            env.apply_status(user, para)
            env.apply_status(user, tr)

class Ranger(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=9)
        self.RANGER_VAR = PROFESSION_VAR
        self.name = "荒原遊俠"
        self.base_hp = self.RANGER_VAR['RANGER_BASE_HP'][0]
        self.passive_name = "冷箭"
        self.passive_desc = f"冷箭：受到攻擊時，{int(self.RANGER_VAR['RANGER_PASSIVE_TRIGGER_RATE'][0] * 100)}% 機率反擊對敵方造成 {self.RANGER_VAR['RANGER_PASSIVE_DAMAGE'][0]} 點傷害。"
        self.baseAtk = self.RANGER_VAR['RANGER_BASE_ATK'][0]
        self.baseDef = self.RANGER_VAR['RANGER_BASE_DEF'][0]

    def passive(self, user, targets, env):
        pass

    def damage_taken(self, user, target, env, dmg):
        super().damage_taken(user, target, env, dmg)
        # 被動觸發反擊
        if random.random() < self.RANGER_VAR['RANGER_PASSIVE_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「冷箭」觸發！"))
            env.deal_damage(user, target, self.RANGER_VAR['RANGER_PASSIVE_DAMAGE'][0], can_be_blocked=True)

        # 埋伏觸發反擊
        if user["effect_manager"].has_effect("埋伏") and random.random() < self.RANGER_VAR['RANGER_SKILL_1_AMBUSH_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 埋伏成功，向敵人發動反擊！"))
            env.deal_damage(user, target, dmg * self.RANGER_VAR['RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER'][0], can_be_blocked=True)

    def on_turn_end(self, user, target, env, id):
        super().on_turn_end(user, target, env, id)
        if 'mines_accumulated' not in user['private_info'] and user['private_info']['mines'] == True:
            user['private_info']['mines_accumulated'] = target['accumulated_damage']
        elif user['private_info']['mines'] == True:
            user['private_info']['mines_accumulated'] += target['accumulated_damage']
        # 引爆處理
        mine_exist = target['effect_manager'].has_effect("地雷")
        # round out bomb!
        if mine_exist==False and user['private_info']['mines'] == True:
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 設下的地雷因時間到而引爆！"))
            user['private_info']['mines'] = False
            # 計算傷害
            dmg = user['private_info']['mines_accumulated'] * self.RANGER_VAR['RANGER_SKILL_3_DAMAGE_RATE_FAIL'][0]
            debuff = DefenseMultiplier(
                multiplier=self.RANGER_VAR['RANGER_SKILL_3_DEBUFF_MULTIPLIER_FAIL'][0],
                duration= 2,
                stackable=False,
                source = self.profession_id * 4 + 3
            )
            
            env.deal_damage(user, target, dmg, can_be_blocked=True)
            env.apply_status(target, debuff)
            target['effect_manager'].remove_all_effects("地雷")
            
            user['private_info']['mines_accumulated'] = 0
        # accumulate damage bomb!
        elif mine_exist==True and user['private_info']['mines'] == True and user['private_info']['mines_accumulated'] >= user['private_info']['cur_hp'] * self.RANGER_VAR['RANGER_SKILL_3_HP_THRESHOLD'][0]:
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 設下的地雷累積傷害達到門檻，即將引爆！"))
            user['private_info']['mines'] = False
            # 計算傷害
            dmg = user['private_info']['mines_accumulated'] * self.RANGER_VAR['RANGER_SKILL_3_DAMAGE_RATE_SUCCESS'][0]
            debuff = DefenseMultiplier(
                multiplier=self.RANGER_VAR['RANGER_SKILL_3_DEBUFF_MULTIPLIER_SUCCESS'][0],
                duration= 2,
                stackable=False,
                source = self.profession_id * 4 + 3
            )
            env.deal_damage(user, target, dmg, can_be_blocked=True)
            env.apply_status(target, debuff)
            target['effect_manager'].remove_all_effects("地雷")
            
            user['private_info']['mines_accumulated'] = 0
                
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if 'mines' not in user['private_info']:
            user['private_info']['mines'] = False
            user['private_info']['cur_hp'] = 0
        if 'continous_bonus' not in user['private_info']:
            user['private_info']['continous_bonus'] = 0
            


        if skill_id == self.profession_id * 4:
            # 續戰攻擊
            times_used = user.get("skills_used", {}).get(skill_id, 0)
            dmg = (self.RANGER_VAR['RANGER_SKILL_0_DAMAGE'][0] + (self.RANGER_VAR['RANGER_SKILL_0_BONUS_DAMAGE_PER_USE'][0] * user['private_info']['continous_bonus'] )) * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            user['private_info']['continous_bonus'] +=1

        elif skill_id == self.profession_id * 4 + 1:
            # 埋伏防禦
            def_buff = DefenseMultiplier(
                multiplier=self.RANGER_VAR['RANGER_SKILL_1_DEFENSE_BUFF'][0],
                duration=self.RANGER_VAR['RANGER_SKILL_1_DURATION'][0],
                stackable=False,
                source=skill_id
            )
            counter_track = Track(
                name="埋伏",
                duration=self.RANGER_VAR['RANGER_SKILL_1_DURATION'][0],
                stacks=1,
                source=skill_id,
                stackable=False,
                max_stack=1
            )
            env.apply_status(user, counter_track)
            env.apply_status(user, def_buff)

        elif skill_id == self.profession_id * 4 + 2:
            # 荒原抗性
            if user["hp"] > self.RANGER_VAR['RANGER_SKILL_2_HP_COST'][0]:
                env.deal_healing(user, self.RANGER_VAR['RANGER_SKILL_2_HP_COST'][0], self_mutilation=True)
                immune_damage = ImmuneDamage(
                    duration=self.RANGER_VAR['RANGER_SKILL_2_DURATION'][0],
                    stackable=False
                )
                immune_control = ImmuneControl(
                    duration=self.RANGER_VAR['RANGER_SKILL_2_DURATION'][0],
                    stackable=False
                )
                env.apply_status(user, immune_damage)
                env.apply_status(user, immune_control)
            else:
                env.add_event(event=BattleEvent(type="text", text=f"{self.name} 嘗試使用「荒原」，但血量不足。"))
        elif skill_id == self.profession_id * 4 + 3:
            tr = Track(name = "地雷", duration = self.RANGER_VAR['RANGER_SKILL_3_DURATION'][0], source = skill_id, stackable = False,stacks=1,max_stack=1)
            env.apply_status(targets[0], tr)
            user['private_info']['mines'] = True
            user['private_info']['cur_hp']  = user['hp']
                  

class ElementalMage(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=10)
        self.ELEMENTALMAGE_VAR = PROFESSION_VAR
        self.name = "元素法師"
        self.base_hp = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_HP'][0]
        self.passive_name = "元素反應"
        self.passive_desc = f"使用不同元素的技能會使技能強化，元素強化只會保留前 2 次使用的元素屬性。消耗元素強化可使使用技能增強，當元素強化有兩個不同屬性的效果時才套用雙屬性強化。"
        self.baseAtk = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_ATK'][0]
        self.baseDef = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_DEF'][0]
            

    def passive(self, user, targets, env):
        if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_TRIGGER_RATE'][0]:
            effect = random.choice([Burn(duration=3, stacks=1), Freeze(duration=3, stacks=1), Paralysis(duration=2)])
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「元素之力」觸發！"))
            env.apply_status(targets[0], effect)

    def damage_taken(self, user, target, env, dmg):
        if user["effect_manager"].get_effects("雷霆護甲") and random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「雷霆護甲」觸發！"))
            env.apply_status(target, Paralysis(duration=2))
        if user["effect_manager"].has_effect("雷霆護甲．神火") :
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「雷霆護甲．神火」觸發！"))
            # 燃燒反擊處理
            # get target's burn effect
            beff = Burn(duration=3, stacks=1)
            env.apply_status(target, beff)
            burn_effects = target["effect_manager"].get_effects("燃燒")
            if burn_effects:
                stack = burn_effects[0].stacks
                # 燃燒傷害固定為5
                dmg = stack * 5
                env.deal_damage(user, target, dmg)
            # 麻痺處理
            if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_TRIGGER_RATE'][0]:
                env.apply_status(target, Burn(duration=3, stacks=1))
    def add_to_private_info(self,status , private_info):
        # 這邊可以用 queue 來做比較好 但是因為只有兩個元素所以直接用 if else好了
        # if passive 0 = empty
        if private_info['passive_0'] == None:
            private_info['passive_0'] = status
        # if passive 1 = empty
        elif private_info['passive_1'] == None:
            private_info['passive_1'] = status
        # if both are not empty => remove passive 0 and passive1 to passive 0
        else:
            private_info['passive_0'] = private_info['passive_1']
            private_info['passive_1'] = status
    def remove_from_private_info(self,status , private_info):
        if private_info['passive_0'] == status:
            private_info['passive_0'] = None
        elif private_info['passive_1'] == status:
            private_info['passive_1'] = None
        else:
            pass

    def set_tar_cool_down(self,env ,target):

        tar_profession_id = target['profession'].profession_id
        skillmgr = env.skill_mgr
        # the four skills is pid*4 +0/1/2/3
        t = [tar_profession_id * 4 + i for i in range(4)]
        # shuffle the skills
        random.shuffle(t)
        for i in range(4):
            if skillmgr.get_skill_cooldown(t[i]) != 0:
                # local skill id
                lsid = t[i] - tar_profession_id * 4
                # set the cool down
                target['cooldowns'][lsid] = skillmgr.get_skill_cooldown(t[i])
                # only set one skill
                break
        
    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        # 元素反應
        if 'passive_0' not in user['private_info'] or 'passive_1' not in user['private_info']:
            user['private_info']['passive_0'] = None
            user['private_info']['passive_1'] = None
            user['private_info']['real_statues'] = '無狀態'
        
        if skill_id == self.profession_id * 4:
            base_dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DAMAGE'][0] * self.baseAtk
            if user['private_info']['real_statues'] == "雷":
                #  15% para dur = 2
                par = Paralysis(duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_DURATION'][0])
                if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_TRIGGER_RATE'][0]:
                    env.apply_status(targets[0], par)
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_SIGLE_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("雷", user['private_info'])
            elif user['private_info']['real_statues'] == "火":
                # add burn effect
                beff = Burn(duration=3, stacks=1)
                env.apply_status(targets[0], beff)
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_SIGLE_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("火", user['private_info'])
            elif user['private_info']['real_statues'] == "冰":
                # add freeze effect
                fr = Freeze(duration=3, stacks=1)
                env.apply_status(targets[0], fr)
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_SIGLE_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("冰", user['private_info'])
            elif user['private_info']['real_statues'] == "冰雷":
                # para BASE dur = ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION  and ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE
                # ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE
                # 麻痺處理
                if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_TRIGGER_RATE'][0]:
                    if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE'][0]:
                        par = Paralysis(duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION'][0] + 1)
                    else:
                        par = Paralysis(duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION'][0])
                    env.apply_status(targets[0], par)
                    
                # add freeze effect
                fr = Freeze(duration=3, stacks=1)
                env.apply_status(targets[0], fr)       
                # cool down set
                self.set_tar_cool_down(env,targets[0])
     
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_MULTI_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("冰", user['private_info'])
                self.remove_from_private_info("雷", user['private_info'])
            elif user['private_info']['real_statues'] == "冰火":
                # 冰處理
                # add freeze effect
                fr = Freeze(duration=3, stacks=1)
                env.apply_status(targets[0], fr)       
                # cool down set
                self.set_tar_cool_down(env,targets[0])
                
                # 火處理
                # add burn effect
                beff = Burn(duration=3, stacks=1)
                env.apply_status(targets[0], beff)
                # 
                burn_effects = targets[0]["effect_manager"].get_effects("燃燒")
                if burn_effects:
                    stack = burn_effects[0].stacks
                    # 燃燒傷害固定為5
                    dmg = stack * 5
                    env.deal_damage(user, targets[0], dmg)
                
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_MULTI_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("冰", user['private_info'])
                self.remove_from_private_info("火", user['private_info'])
            elif user['private_info']['real_statues'] == "雷火":
                # 雷處理
                if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_TRIGGER_RATE'][0]:
                    if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE'][0]:
                        par = Paralysis(duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION'][0] + 1)
                    else:
                        par = Paralysis(duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION'][0])
                    env.apply_status(targets[0], par)
                # 火處理
                # 火處理
                # add burn effect
                beff = Burn(duration=3, stacks=1)
                env.apply_status(targets[0], beff)
                # 
                burn_effects = targets[0]["effect_manager"].get_effects("燃燒")
                if burn_effects:
                    stack = burn_effects[0].stacks
                    # 燃燒傷害固定為5
                    dmg = stack * 5
                    env.deal_damage(user, targets[0], dmg)
                
                dmg = base_dmg * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_MULTI_ELEMENT_BONOUS'][0]
                self.remove_from_private_info("雷", user['private_info'])
                self.remove_from_private_info("火", user['private_info'])
            else:
                dmg = base_dmg
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

        if skill_id == self.profession_id * 4 +1:
            # 雷霆護甲
            if user['private_info']['real_statues'] == "冰火" :
                env.add_event(event=BattleEvent(type="text", text=f"{self.name} 的「雷霆護甲」強化為「雷霆護甲．神火」！"))
                self.remove_from_private_info("冰", user['private_info'])
                self.remove_from_private_info("火", user['private_info'])
                def_buff = DefenseMultiplier(
                    multiplier=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DEFENSE_BUFF'][0],
                    duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                track = Track(
                    name="雷霆護甲．神火",
                    duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DURATION'][0],
                    stacks=1,
                    source=skill_id,
                    stackable=False,
                    max_stack=99
                )

            else:
                # 無強化
                def_buff = DefenseMultiplier(
                    multiplier=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DEFENSE_BUFF'][0],
                    duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                track = Track(
                    name="雷霆護甲",
                    duration=self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DURATION'][0],
                    stacks=1,
                    source=skill_id,
                    stackable=False,
                    max_stack=99
                )
            env.apply_status(user, def_buff)
            env.apply_status(user, track)
            self.add_to_private_info("雷", user['private_info'])
            
        if skill_id == self.profession_id * 4 + 2:
            # 為自身疊加冰元素，對敵方造成25傷害，立即疊加1~3層冰凍效果，並使敵方的隨機一個技能進入冷卻。
            
            if user['private_info']['real_statues'] == "雷火":
                self.remove_from_private_info("雷", user['private_info'])
                self.remove_from_private_info("火", user['private_info'])
                
                env.add_event(event=BattleEvent(type="text", text=f"{self.name} 的「寒星墜落」強化為「寒星墜落．雷霆」！"))
                # ELEMENTALMAGE_SKILL_2_BONOUS_DAMAGE_MULTIPLIER
                dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_DAMAGE'][0] * self.baseAtk * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_BONOUS_DAMAGE_MULTIPLIER'][0]
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                
                # para = 25% para dur = 2
                if random.random() < self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE'][0]:
                    env.apply_status(targets[0], Paralysis(duration=2))
                # # freeze effect = 1~3
                freeze = Freeze(duration=3, stacks=random.randint(1,3))
                env.apply_status(targets[0], freeze)                
                # cool down set
                for _ in range(2):
                    self.set_tar_cool_down(env,targets[0])
                pass
            else: 
                dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_DAMAGE'][0] * self.baseAtk
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

                # freeze effect = 1~3
                freeze = Freeze(duration=3, stacks=random.randint(1,3))
                env.apply_status(targets[0], freeze)
                self.set_tar_cool_down(env,targets[0])
            
            self.add_to_private_info("冰", user['private_info'])
        if skill_id == self.profession_id * 4 + 3:
            dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_DAMAGE'][0] * self.baseAtk
            if user['private_info']['real_statues'] == "冰雷":
                self.remove_from_private_info("冰", user['private_info'])
                self.remove_from_private_info("雷", user['private_info'])
                
                env.add_event(event=BattleEvent(type="text", text=f"{self.name} 的「焚天」強化為「焚天．寒焰」！"))
                dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_DAMAGE'][0] * self.baseAtk * self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_BONOUS_DAMAGE_MULTIPLIER'][0]
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                env.apply_status(targets[0], Freeze(duration=3, stacks=1))
                env.apply_status(targets[0], Burn(duration=3, stacks=1))
                self.set_tar_cool_down(env,targets[0])
                burn_effects = targets[0]["effect_manager"].get_effects("燃燒")
                if burn_effects:
                    stack = burn_effects[0].stacks
                    # 燃燒傷害固定為5
                    dmg = stack * 5
                    env.deal_damage(user, targets[0], dmg)
                
            else :
                dmg = self.ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_DAMAGE'][0] * self.baseAtk
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                env.apply_status(targets[0], Burn(duration=3, stacks=1))
                burn_effects = targets[0]["effect_manager"].get_effects("燃燒")
                if burn_effects:
                    stack = burn_effects[0].stacks
                    # 燃燒傷害固定為5
                    dmg = stack * 5
                    env.deal_damage(user, targets[0], dmg)
            self.add_to_private_info("火", user['private_info'])
                
        # end of elemental check
        havestatus = False
        if (user['private_info']['passive_0'] == '雷' and user['private_info']['passive_1'] == '火') or (user['private_info']['passive_0'] == '火' and user['private_info']['passive_1'] == '雷'):
            user['private_info']['real_statues'] = "雷火"
            havestatus = True
        elif (user['private_info']['passive_0'] == '雷' and user['private_info']['passive_1'] == '冰') or (user['private_info']['passive_0'] == '冰' and user['private_info']['passive_1'] == '雷'):
            user['private_info']['real_statues'] = "冰雷"
            havestatus = True
        elif (user['private_info']['passive_0'] == '火' and user['private_info']['passive_1'] == '冰') or (user['private_info']['passive_0'] == '冰' and user['private_info']['passive_1'] == '火'):
            user['private_info']['real_statues'] = "冰火"
            havestatus = True
        elif user['private_info']['passive_0'] == "雷" or user['private_info']['passive_1'] == "雷":
            user['private_info']['real_statues'] = "雷"
            havestatus = True
        elif user['private_info']['passive_0'] == "火" or user['private_info']['passive_1'] == "火":
            user['private_info']['real_statues'] = "火"
            havestatus = True
        elif user['private_info']['passive_0'] == "冰" or user['private_info']['passive_1'] == "冰":
            user['private_info']['real_statues'] = "冰"
            havestatus = True
        if havestatus == True:
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 持有 {user['private_info']['real_statues']} 元素強化。"))
        
            


class HuangShen(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=11)
        self.HUANGSHEN_VAR = PROFESSION_VAR
        self.name = "荒神"
        self.base_hp = self.HUANGSHEN_VAR['HUANGSHEN_BASE_HP'][0]
        self.passive_name = "枯萎之刃"
        self.passive_desc = f"隨著造成傷害次數增加，攻擊時額外進行隨機追打，每造成兩次傷害增加一次最高追打機會，追打造成敵方當前生命的 {self.HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT'][0]*100}% 血量；額外追打不會累積傷害次數。"
        self.baseAtk = self.HUANGSHEN_VAR['HUANGSHEN_BASE_ATK'][0]
        self.baseDef = self.HUANGSHEN_VAR['HUANGSHEN_BASE_DEF'][0]
        

    def passive(self, user, targets, env):
        bonus_hits = user['private_info']['hit_times'] // self.HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_THRESHOLD'][0]
        if bonus_hits > 0:
            for _ in range(bonus_hits):
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的被動技能「枯萎之刃」觸發！"))
                env.deal_damage(user, targets[0], int(
                    targets[0]["hp"] * self.HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT'][0]), can_be_blocked=True)
                # if have "風化"
                if user["effect_manager"].has_effect("風化"):
                    env.add_event(event=BattleEvent(
                        type="text", text=f"枯萎之刃對敵方造成風化，降低敵方的防禦及治癒力！"))
                    def_debuff = DefenseMultiplier(
                        multiplier=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_3_REDUCE_MULTIPLIER'][0],
                        duration=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_3_DURATION'][0],
                        stackable=True,
                        stacks=1,
                        max_stack=99,
                        source=self.profession_id * 4 + 3
                    )
                    heal_debuff = HealMultiplier(
                        multiplier=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_3_REDUCE_MULTIPLIER'][0],
                        duration=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_3_DURATION'][0],
                        stackable=True,
                        stacks=1,
                        max_stack=99,
                        source=self.profession_id * 4 + 3
                    )
                    env.apply_status(targets[0], def_debuff)
                    env.apply_status(targets[0], heal_debuff)
                    
                # heal
                if user["effect_manager"].has_effect("風化"):
                    heal = bonus_hits * self.HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_HEAL'][0]
                    env.deal_healing(user, heal)
                    
                    

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if 'hit_times' not in user['private_info']:
            user['private_info']['hit_times'] = 0
        if 'skill2_used' not in user['private_info']:
            user['private_info']['skill2_used'] = 0

        if skill_id == self.profession_id * 4:
            # 枯骨

            times = random.randint(*self.HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'])
            for i in range(times):
                dmg = self.HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE'][0] * self.baseAtk * (1 - i * self.HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE_REDUCTION_PER_HIT'][0])
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                self.passive(user, targets, env)
                user['private_info']['hit_times'] += 1

        elif skill_id == self.profession_id * 4 + 1:
            # 荒原
            time_used = user["private_info"]["skill2_used"] 
            if time_used % 3 == 0:
                atk_buff = DamageMultiplier(
                    multiplier=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_ATK_BUFF'][0],
                    duration=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, atk_buff)
            elif time_used % 3 == 1:
                heal_buff = HealMultiplier(
                    multiplier=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_HEAL_BUFF'][0],
                    duration=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, heal_buff)
            elif time_used % 3 == 2:
                def_buff = DefenseMultiplier(
                    multiplier=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_DEF_BUFF'][0],
                    duration=self.HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, def_buff)
            user["private_info"]["skill2_used"] += 1

        elif skill_id == self.profession_id * 4 + 2:
            # 生命逆流
            heal_amount = user['private_info']['hit_times']* self.HUANGSHEN_VAR['HUANGSHEN_SKILL_2_HEAL_MULTIPLIER'][0]
            env.deal_healing(user, heal_amount)
            
        elif skill_id == self.profession_id * 4 + 3:
            tr = Track(name = "風化", duration = self.HUANGSHEN_VAR['HUANGSHEN_SKILL_3_DURATION'][0], source = skill_id, stackable = False,stacks=1,max_stack=1)
            env.apply_status(user, tr)

class GodOfStar(BattleProfession):
    def __init__(self,PROFESSION_VAR):
        super().__init__(profession_id=12)
        self.GODOFSTAR_VAR = PROFESSION_VAR
        self.name = "星神"
        self.base_hp = self.GODOFSTAR_VAR['GODOFSTAR_BASE_HP'][0]
        self.passive_name = "天啟星盤"
        self.passive_desc = f"星神在戰鬥中精通增益與減益效果的能量運用。每當場上有一層「能力值增益」或「減益」效果時，每回合會額外對敵方造成 {self.GODOFSTAR_VAR['GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT'][0]} 點傷害 並恢復 {self.GODOFSTAR_VAR['GODOFSTAR_PASSIVE_HEAL_PER_EFFECT'][0]} 點生命值。"
        self.baseAtk = self.GODOFSTAR_VAR['GODOFSTAR_BASE_ATK'][0]
        self.baseDef = self.GODOFSTAR_VAR['GODOFSTAR_BASE_DEF'][0]
        

    def passive(self, user, targets, env):
        buff_effects = []
        for eff in user["effect_manager"].active_effects.values():
            for effect in eff:
                if effect.type == "buff":
                    buff_effects.append(effect)
        for eff in targets[0]["effect_manager"].active_effects.values():
            for effect in eff:
                if effect.type == "buff":
                    buff_effects.append(effect)
        buff_effect_count = len(buff_effects)
        bonus_damage = self.GODOFSTAR_VAR['GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT'][0] * buff_effect_count
        bonus_heal = self.GODOFSTAR_VAR['GODOFSTAR_PASSIVE_HEAL_PER_EFFECT'][0] * buff_effect_count
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「天啟星盤」觸發！"))
        return bonus_damage, bonus_heal

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        dmg, heal = self.passive(user, targets, env)
        if skill_id == self.profession_id * 4:
            dmg += self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            buff = random.choice([
                DamageMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id),
                DefenseMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id),
                HealMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                
            ])
            env.apply_status(targets[0], buff)
            env.deal_healing(user, heal)

        elif skill_id == self.profession_id * 4 + 1:
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            heal += self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_HEAL'][0]
            env.deal_healing(user, heal)
            debuff = random.choice([
                DamageMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id),
                DefenseMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id),
                HealMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id)
                
            ])
            env.apply_status(user, debuff)

        elif skill_id == self.profession_id * 4 + 2:
            dmg *= self.GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0]
            heal *= self.GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的技能「虛擬創星圖」強化了天啟星盤的力量，增加了傷害和回復。"))
            dmg += self.GODOFSTAR_VAR['GODOFSTAR_SKILL_2_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            env.deal_healing(user, heal)
        elif skill_id == self.profession_id * 4 + 3:
            dmg *= self.GODOFSTAR_VAR['GODOFSTAR_SKILL_3_PASSIVE_MULTIPLIER'][0]
            heal *= self.GODOFSTAR_VAR['GODOFSTAR_SKILL_3_PASSIVE_MULTIPLIER'][0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的技能「聯星」強化了天啟星盤的力量，增加了傷害和回復。"))
            # find user buff with type = buff and id = self.profession_id * 4 + 1
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            env.deal_healing(user, heal)
            
            effects_copy_1 = list(user["effect_manager"].active_effects.values())
            effects_copy_2 = list(targets[0]["effect_manager"].active_effects.values())
            # 這邊是buff
            for eff in effects_copy_1:
                for effect in eff:
                    if effect.source == self.profession_id * 4 + 1:
                        if effect.name =='攻擊力':
                            debuff = DamageMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(targets[0], debuff)
                        elif effect.name == '防禦力':
                            debuff = DefenseMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(targets[0], debuff)

                        elif effect.name == '治療力':
                            debuff = HealMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(targets[0], debuff)
                        
            # find target debuff with type = buff and id = self.profession_id * 4 + 1
            for eff in effects_copy_2:
                for effect in eff:
                    if effect.source == self.profession_id * 4 :
                        # check its name
                        if effect.name =='攻擊力':
                            buff = DamageMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(user, buff)
                        elif effect.name == '防禦力':
                            buff = DefenseMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(user, buff)
                        elif effect.name == '治療力':
                            buff = HealMultiplier(multiplier=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=self.GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
                            env.apply_status(user, buff)
            
            
            
            
