# professions.py

from .status_effects import (
    Burn, Poison, Freeze,
    DamageMultiplier, DefenseMultiplier, HealMultiplier, HealthPointRecover, Paralysis,
    ImmuneDamage, ImmuneControl, BleedEffect, Track
)
import random
from .skills import SkillManager, sm

from .profession_var import *

from .battle_event import BattleEvent
import math


class BattleProfession:
    def __init__(self, profession_id, name, base_hp, passive_name, passive_desc="", baseAtk=1.0, baseDef=1.0):
        self.profession_id = profession_id
        self.name = name
        self.base_hp = base_hp
        self.max_hp = base_hp
        self.passive_name = passive_name
        self.passive_desc = passive_desc
        self.baseAtk = baseAtk
        self.baseDef = baseDef
        self.default_passive_id = (profession_id * -1)-1

    def get_available_skill_ids(self, cooldowns: dict):
        # get cooldowns
        ok_skills = []
        if cooldowns[0] == 0:
            ok_skills.append(self.profession_id * 3)
        if cooldowns[1] == 0:
            ok_skills.append(self.profession_id*3 + 1)
        if cooldowns[2] == 0:
            ok_skills.append(self.profession_id*3 + 2)
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
        cooldowns_skill_id = skill_id - self.profession_id * 3
        env.add_event(user=user, event=BattleEvent(type="skill", appendix={
                      "skill_id": skill_id, "relatively_skill_id": cooldowns_skill_id}))
        # check cooldown
        # get skill id's cooldown

        if skill_id in self.get_available_skill_ids(user["cooldowns"]):
            # set cooldown
            local_skill_id = skill_id - self.profession_id * 3
            user["cooldowns"][local_skill_id] = sm.get_skill_cooldown(skill_id)
            if sm.get_skill_cooldown(skill_id) > 0:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」進入冷卻 {sm.get_skill_cooldown(skill_id)} 回合。"))

            return -1
        else:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的技能「{sm.get_skill_name(skill_id)}」還在冷卻中。"))
            return -1


class Paladin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=0,
            name="聖騎士",
            base_hp=PALADIN_VAR['PALADIN_BASE_HP'][0],
            passive_name="聖光",
            passive_desc=f"攻擊時，{int(PALADIN_VAR['PALADIN_PASSIVE_TRIGGER_RATE'][0]*100)}% 機率恢復最大血量的 {int(PALADIN_VAR['PALADIN_PASSIVE_HEAL_RATE'][0]*100)}% 的生命值，回復超出最大生命時，對敵方造成 {int(PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0]*100)}% 的回復傷害",
            baseAtk=PALADIN_VAR['PALADIN_BASE_ATK'][0],
            baseDef=PALADIN_VAR['PALADIN_BASE_DEF'][0]
        )
        self.heal_counts = {}

    def passive(self, user, targets, env):
        # 被動技能：15%機率回復最大血量的15%，超出部分造成100%回復傷害
        super().passive(user, targets, env)

        if random.random() < PALADIN_VAR['PALADIN_PASSIVE_TRIGGER_RATE'][0]:
            heal_amount = int(
                self.max_hp * PALADIN_VAR['PALADIN_PASSIVE_HEAL_RATE'][0])
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 聖光觸發，恢復了血量。"))
            env.deal_healing(
                user, heal_amount, rate=PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 0:
            # 技能 0 => 對單體造成40點傷害
            dmg = self.baseAtk * PALADIN_VAR['PALADIN_SKILL_0_DAMAGE'][0]
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == 1:
            # 技能 1 => 本回合迴避攻擊，回復10點血量。冷卻3回合。
            user["is_defending"] = True
            heal_amount = PALADIN_VAR['PALADIN_SKILL_1_HEAL'][0]

            env.deal_healing(
                user, heal_amount, rate=PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])

        elif skill_id == 2:
            # 技能 2 => 恢復血量，第一次50, 第二次35, 第三次及以後15
            times_healed = user.get("times_healed", 0)
            if times_healed == 0:
                heal_amount = PALADIN_VAR['PALADIN_SKILL_2_FIRST_HEAL'][0]
            elif times_healed == 1:
                heal_amount = PALADIN_VAR['PALADIN_SKILL_2_SECOND_HEAL'][0]
            else:
                heal_amount = PALADIN_VAR['PALADIN_SKILL_2_MORE_HEAL'][0]

            env.deal_healing(
                user, heal_amount, rate=PALADIN_VAR['PALADIN_PASSIVE_OVERHEADLINGE_RATE'][0], heal_damage=True, target=targets[0])
            user["times_healed"] = times_healed + 1


class Mage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=1,
            name="法師",
            base_hp=MAGE_VAR['MAGE_BASE_HP'][0],
            passive_name="魔力充盈",
            passive_desc=f"攻擊造成異常狀態時，{int(MAGE_VAR['MAGE_PASSIVE_TRIGGER_RATE'][0]*100)}% 機率額外疊加一層異常狀態（燃燒或冰凍）。",
            baseAtk=MAGE_VAR['MAGE_BASE_ATK'][0],
            baseDef=MAGE_VAR['MAGE_BASE_DEF'][0]
        )

    def passive(self, user, targets, env):
        # 被動技能：攻擊造成異常狀態時，有機率額外疊加異常狀態
        if random.random() < MAGE_VAR['MAGE_PASSIVE_TRIGGER_RATE'][0]:
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
        if skill_id == 3:
            # 技能 0：火焰之球
            dmg = MAGE_VAR['MAGE_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            burn_effect = Burn(duration=3, stacks=1)
            env.apply_status(targets[0], burn_effect)
            self.passive(user, targets, env)

        elif skill_id == 4:
            # 技能 1：冰霜箭
            dmg = MAGE_VAR['MAGE_SKILL_1_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            freeze_effect = Freeze(duration=3, stacks=1)
            env.apply_status(targets[0], freeze_effect)
            self.passive(user, targets, env)

        elif skill_id == 5:
            # 技能 2：全域爆破
            base_dmg = MAGE_VAR['MAGE_SKILL_2_BASE_DAMAGE'][0] * self.baseAtk
            total_layers = sum(
                eff.stacks for effects in targets[0]["effect_manager"].active_effects.values()
                for eff in effects if isinstance(eff, (Burn, Freeze))
            )
            dmg = base_dmg + \
                MAGE_VAR['MAGE_SKILL_2_STATUS_MULTIPLIER'][0] * total_layers
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            targets[0]["effect_manager"].remove_all_effects("燃燒")
            targets[0]["effect_manager"].remove_all_effects("凍結")


class Assassin(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=2,
            name="刺客",
            base_hp=ASSASSIN_VAR['ASSASSIN_BASE_HP'][0],
            passive_name="刺殺",
            passive_desc=f"攻擊時額外造成敵方當前 {int(ASSASSIN_VAR['ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE'][0] * 100)}% 生命值的傷害，並且 {int(ASSASSIN_VAR['ASSASSIN_PASSIVE_TRIGGER_RATE'][0] * 100)}% 機率對敵方造成中毒狀態。",
            baseAtk=ASSASSIN_VAR['ASSASSIN_BASE_ATK'][0],
            baseDef=ASSASSIN_VAR['ASSASSIN_BASE_DEF'][0]
        )

    def passive(self, targets, damage, env):
        # 被動技能：額外傷害和中毒效果
        if random.random() < ASSASSIN_VAR['ASSASSIN_PASSIVE_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"刺客對敵方造成額外中毒效果！"))
            effect = Poison(duration=3, stacks=1)
            env.apply_status(targets[0], effect)
        return damage + int(targets[0]["hp"] * ASSASSIN_VAR['ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE'][0])

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 6:
            # 技能 0：致命暗殺
            dmg = ASSASSIN_VAR['ASSASSIN_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(targets, dmg, env)
            if random.random() < ASSASSIN_VAR['ASSASSIN_SKILL_0_CRIT_RATE'][0]:
                env.add_event(event=BattleEvent(type="text", text=f"擊中要害！"))
                dmg *= 2
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

        elif skill_id == 7:
            # 技能 1：毒爆
            target = targets[0]
            total_layers = 0
            for effects in target["effect_manager"].active_effects.values():
                for eff in effects:
                    if isinstance(eff, Poison):
                        total_layers += eff.stacks
            if total_layers > 0:
                env.add_event(event=BattleEvent(type="text", text=f"引爆了中毒效果！"))
                dmg = ASSASSIN_VAR['ASSASSIN_SKILL_1_DAMAGE_PER_LAYER'][0] * total_layers
                dmg = self.passive(targets, dmg, env)
                heal = ASSASSIN_VAR['ASSASSIN_SKILL_1_HEAL_PER_LAYER'][0] * total_layers
                env.deal_damage(user, target, dmg, can_be_blocked=True)
                env.deal_healing(user, heal)
                target["effect_manager"].remove_all_effects("中毒")
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"無法引爆中毒效果，對方並未中毒。"))

        elif skill_id == 8:
            # 技能 2：毒刃襲擊
            dmg = ASSASSIN_VAR['ASSASSIN_SKILL_2_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(targets, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

            # 計算隨機疊加的中毒層數
            weights = [
                ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_1_WEIGHT'][0],
                ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_2_WEIGHT'][0],
                ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_3_WEIGHT'][0],
                ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_4_WEIGHT'][0],
                ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_STACKS_5_WEIGHT'][0]
            ]
            add_stacks = random.choices(
                [1, 2, 3, 4, 5], weights=weights, k=1)[0]
            effect = Poison(duration=3, stacks=add_stacks)
            env.apply_status(targets[0], effect)


class Archer(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=3,
            name="弓箭手",
            base_hp=ARCHER_VAR['ARCHER_BASE_HP'][0],
            passive_name="鷹眼",
            passive_desc=f"攻擊時 {int(ARCHER_VAR['ARCHER_PASSIVE_BASE_TRIGGER_RATE'][0] * 100)}% 機率造成 2 倍傷害；敵方防禦基值越高時，額外增加觸發機率。",
            baseAtk=ARCHER_VAR['ARCHER_BASE_ATK'][0],
            baseDef=ARCHER_VAR['ARCHER_BASE_DEF'][0]
        )

    def passive(self, env, dmg, tar):
        target = tar[0]
        prob = ARCHER_VAR['ARCHER_PASSIVE_BASE_TRIGGER_RATE'][0]
        if target["profession"].baseDef > 1:
            prob += (target["profession"].baseDef - 1) * \
                ARCHER_VAR['ARCHER_PASSIVE_TRIGGER_RATE_BONUS'][0]
            prob = min(prob, ARCHER_VAR['ARCHER_PASSIVE_TRIGGER_RATE_MAX'][0])
        if random.random() < prob:
            env.add_event(event=BattleEvent(
                type="text", text=f"被動技能「鷹眼」觸發，攻擊造成{ARCHER_VAR['ARCHER_PASSIVE_DAMAGE_MULTIPLIER'][0]} 倍傷害！"))
            return dmg * ARCHER_VAR['ARCHER_PASSIVE_DAMAGE_MULTIPLIER'][0]
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 9:
            # 技能 0：五連矢
            dmg = ARCHER_VAR['ARCHER_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg /= 5 
            for i in range(5):
                dmg = self.passive(env, dmg, targets)
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                if i==0:
                    def_buff = DefenseMultiplier(
                    multiplier=ARCHER_VAR['ARCHER_SKILL_0_DEFENSE_DEBUFF'][0],
                    duration=ARCHER_VAR['ARCHER_SKILL_0_DURATION'][0],
                    stackable=False,
                    source=skill_id)
                    env.apply_status(targets[0], def_buff)
                    
                
            

        elif skill_id == 10:
            # 技能 1：箭矢補充
            if random.random() < ARCHER_VAR['ARCHER_SKILL_1_SUCESS_RATIO'][0]:
                dmg_multiplier = ARCHER_VAR['ARCHER_SKILL_1_DAMAGE_MULTIPLIER'][0]
                dmg_buff = DamageMultiplier(
                    multiplier=dmg_multiplier,
                    duration=ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.add_event(event=BattleEvent(
                    type="text", text=f"箭矢補充成功，攻擊力大幅提升！"))
                env.apply_status(user, dmg_buff)
            else:
                def_multiplier = ARCHER_VAR['ARCHER_SKILL_1_DEFENSE_MULTIPLIER'][0]
                def_debuff = DefenseMultiplier(
                    multiplier=def_multiplier,
                    duration=ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.add_event(event=BattleEvent(
                    type="text", text=f"箭矢補充失敗，防禦力下降！"))
                env.apply_status(user, def_debuff)

        elif skill_id == 11:
            # 技能 2：吸血箭
            dmg = ARCHER_VAR['ARCHER_SKILL_2_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(env, dmg, targets)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            heal_amount = ARCHER_VAR['ARCHER_SKILL_2_HEAL'][0]
            env.deal_healing(user, heal_amount)

class Berserker(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=4,
            name="狂戰士",
            base_hp=BERSERKER_VAR['BERSERKER_BASE_HP'][0],
            passive_name="狂暴",
            passive_desc=f"若自身血量低於 {int(BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD'][0] * 100)}%，攻擊增加失去生命值的 {int(BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0] * 100)}% 的傷害。",
            baseAtk=BERSERKER_VAR['BERSERKER_BASE_ATK'][0],
            baseDef=BERSERKER_VAR['BERSERKER_BASE_DEF'][0]
        )

    def passive(self, user, dmg, env):
        if user["hp"] < (user["max_hp"] * BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD'][0]):
            loss_hp = user["max_hp"] - user["hp"]
            dmg += loss_hp * BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「狂暴」觸發，攻擊時增加了 {int(loss_hp * BERSERKER_VAR['BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE'][0])} 點傷害。"))
        return dmg

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        if skill_id == 12:  # 對應 BERSERKER_SKILL_0
            dmg = BERSERKER_VAR['BERSERKER_SKILL_0_DAMAGE'][0] * self.baseAtk
            dmg = self.passive(user, dmg, env)
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self_mutilation = dmg * BERSERKER_VAR['BERSERKER_SKILL_0_SELF_MUTILATION_RATE'][0]
            env.add_event(event=BattleEvent(type="text", text=f"{self.name} 受到反噬。"))
            env.deal_healing(user, self_mutilation, self_mutilation=True)

        elif skill_id == 13:  # 對應 BERSERKER_SKILL_1
            if user["hp"] > BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0]:
                env.deal_healing(user, BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0], self_mutilation=True)
                heal_effect = HealthPointRecover(
                    hp_recover=BERSERKER_VAR['BERSERKER_SKILL_1_HEAL_PER_TURN'][0],
                    duration=BERSERKER_VAR['BERSERKER_SKILL_1_DURATION'][0],
                    stackable=False,
                    source=skill_id,
                    env=env
                )
                env.apply_status(user, heal_effect)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「熱血」，但血量不足。"))

        elif skill_id == 14:  # 對應 BERSERKER_SKILL_2
            if user["hp"] > BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0]:
                env.deal_healing(user, BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0], self_mutilation=True)
                immune_control = ImmuneControl(duration=BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0], stackable=False)
                env.apply_status(user, immune_control)
                def_buff = DefenseMultiplier(
                    multiplier=BERSERKER_VAR['BERSERKER_SKILL_2_DEFENSE_BUFF'][0],
                    duration=BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0],
                    stackable=False,
                    source=skill_id
                )
                env.apply_status(user, def_buff)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「血怒」，但血量不足。"))


class DragonGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=5,
            name="龍神",
            base_hp=DRAGONGOD_VAR['DRAGONGOD_BASE_HP'][0],
            passive_name="龍血",
            passive_desc=f"每回合疊加一個龍神狀態，龍神狀態每層增加 {int((DRAGONGOD_VAR['DRAGONGOD_PASSIVE_ATK_MULTIPLIER'][0] - 1) * 100)}% 攻擊力、增加 {int((DRAGONGOD_VAR['DRAGONGOD_PASSIVE_DEF_MULTIPLIER'][0] - 1) * 100)}% 防禦力。",
            baseAtk=DRAGONGOD_VAR['DRAGONGOD_BASE_ATK'][0],
            baseDef=DRAGONGOD_VAR['DRAGONGOD_BASE_DEF'][0]
        )

    def on_turn_start(self, user, targets, env, id):
        # 每回合觸發被動技能，疊加龍血狀態
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「龍血」觸發！"))

        atk_buff = DamageMultiplier(
            multiplier=DRAGONGOD_VAR['DRAGONGOD_PASSIVE_ATK_MULTIPLIER'][0],
            duration=99,
            stacks=1,
            source=self.default_passive_id,
            stackable=True,
            max_stack=99
        )
        def_buff = DefenseMultiplier(
            multiplier=DRAGONGOD_VAR['DRAGONGOD_PASSIVE_DEF_MULTIPLIER'][0],
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

        if skill_id == 15:  # 對應 DRAGONGOD_SKILL_0
            base_dmg = DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BASE_DAMAGE'][0] * self.baseAtk
            dragon_soul_effect = user["effect_manager"].get_effects("龍血")[0]
            stacks = dragon_soul_effect.stacks if dragon_soul_effect else 0
            bonus_dmg = stacks * DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK'][0]
            dmg = base_dmg + bonus_dmg
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)

        elif skill_id == 16:  # 對應 DRAGONGOD_SKILL_1
            heal_amount = DRAGONGOD_VAR['DRAGONGOD_SKILL_1_HEAL_AMOUNT'][0]
            env.deal_healing(user, heal_amount)
            bleed_effect = HealthPointRecover(
                hp_recover=DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_PER_TURN'][0],
                duration=DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_DURATION'][0],
                stackable=False,
                source=skill_id,
                env=env,
                self_mutilation=True
            )
            env.apply_status(user, bleed_effect)

        elif skill_id == 17:  # 對應 DRAGONGOD_SKILL_2
            dragon_soul_effect = user["effect_manager"].get_effects("龍血")[0]
            stacks = dragon_soul_effect.stacks if dragon_soul_effect else 0
            consume_ratio = DRAGONGOD_VAR['DRAGONGOD_SKILL_2_STACK_CONSUMPTION'][0]
            consume_stack = int(stacks * consume_ratio)
            if consume_stack > 0:
                damage = consume_stack * DRAGONGOD_VAR['DRAGONGOD_SKILL_2_DAMAGE_PER_STACK'][0]
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


class BloodGod(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=6,
            name="血神",
            base_hp=BLOODGOD_VAR['BLOODGOD_BASE_HP'][0],
            passive_name="血脈",
            passive_desc=f"每回合會累積所受傷害至血脈裡面，所受傷害累積越多會降低血脈的強度。每受到最大血量的 {int(BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0] * 100)}% 傷害，則自身攻擊力、防禦力、治癒力降低 {int((1 - BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0]) * 100)}%。",
            baseAtk=BLOODGOD_VAR['BLOODGOD_BASE_ATK'][0],
            baseDef=BLOODGOD_VAR['BLOODGOD_BASE_DEF'][0]
        )
        self.bleed_stacks = 0

    def passive(self, user, targets, env):
        pass

    def on_turn_end(self, user, targets, env, id):
        if 'total_accumulated_damage' in user['private_info']:
            user['private_info']['total_accumulated_damage'] += user['accumulated_damage']
        else:
            user['private_info']['total_accumulated_damage'] = user['accumulated_damage']

        # 每受到最大血量的指定比例傷害，降低攻擊、防禦、治癒力
        threshold = user['max_hp'] * BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]
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
                    multiplier=BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
                    duration=99, stacks=stack,
                    source=self.default_passive_id, stackable=True, max_stack=99
                )
                deffect = DefenseMultiplier(
                    multiplier=BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
                    duration=99, stacks=stack,
                    source=self.default_passive_id, stackable=True, max_stack=99
                )
                heffect = HealMultiplier(
                    multiplier=BLOODGOD_VAR['BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION'][0],
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
                user["hp"] = int(user["max_hp"] * BLOODGOD_VAR['BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO'][0])
                user["private_info"]["total_accumulated_damage"] += dmg * BLOODGOD_VAR['BLOODGOD_SKILL_2_BLOOD_ACCUMULATION_MULTIPLIER'][0]
                user["effect_manager"].remove_all_effects("轉生之印")

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 18:
            dmg = BLOODGOD_VAR['BLOODGOD_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            bleed_effect = BleedEffect(duration=BLOODGOD_VAR['BLOODGOD_SKILL_0_BLEED_DURATION'][0], stacks=1)
            env.apply_status(targets[0], bleed_effect)

            bleed_effects = targets[0]["effect_manager"].get_effects("流血")
            if bleed_effects:
                stack = bleed_effects[0].stacks
                heal_amount = stack * BLOODGOD_VAR['BLOODGOD_SKILL_0_HEAL_PER_BLEED_STACK'][0]
                env.deal_healing(user, heal_amount)

        elif skill_id == 19:
            bleed_effects = targets[0]["effect_manager"].get_effects("流血")
            if bleed_effects:
                stack = bleed_effects[0].stacks
                user['private_info']['total_accumulated_damage'] = max(
                    user['private_info']['total_accumulated_damage'] - stack * BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER'][0], 0)
                

                if user['private_info']['total_accumulated_damage'] < user['max_hp'] * BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]:
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有完美的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*2):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有上等的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*3):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有普通的血脈，並使敵方流血更加嚴重！"))
                elif user['private_info']['total_accumulated_damage'] < user['max_hp'] * (BLOODGOD_VAR['BLOODGOD_PASSIVE_DAMAGE_THRESHOLD'][0]*4):
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有混濁的血脈，並使敵方流血更加嚴重！"))
                else:
                    env.add_event(event = BattleEvent(type="text",text=f"{self.name} 發動血脈祭儀來純化血脈，現在擁有拙劣的血脈，並使敵方流血更加嚴重！"))
                    
                heal_amount = stack * BLOODGOD_VAR['BLOODGOD_SKILL_1_HEAL_MULTIPLIER'][0]
                env.deal_healing(user, heal_amount)
                env.set_status(targets[0], "流血", stacks=stack * BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_STACK_MULTIPLIER'][0])
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「血脈祭儀」，但是流血層數不夠發動血脈祭儀。"))

        elif skill_id == 20:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 啟用了久遠的未知印記，神秘的力量開始聚於自身之上。"))
            env.deal_healing(user, int(user["hp"] * BLOODGOD_VAR['BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO'][0]), self_mutilation=True)
            eff = Track(name="轉生之印", duration=BLOODGOD_VAR['BLOODGOD_SKILL_2_DURATION'][0], stacks=1,
                        source=skill_id, stackable=False)
            env.apply_status(user, eff)


class SteadfastWarrior(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=7,
            name="剛毅武士",
            base_hp=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_HP'][0],
            passive_name="堅韌壁壘",
            passive_desc=f"每回合開始時恢復已損生命值的 {int(STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT'][0] * 100)}%。",
            baseAtk=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_ATK'][0],
            baseDef=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_BASE_DEF'][0]
        )

    def passive(self, user, targets, env):
        pass

    def on_turn_start(self, user, targets, env, id):
        heal = int((self.max_hp - user["hp"]) * STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT'][0])
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「堅韌壁壘」觸發。"))
        env.deal_healing(user, heal)

    def on_turn_end(self, user, targets, env, id):
        if id == 23:
            if user["last_attacker"]:
                dmg = user["last_damage_taken"] * STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER'][0]
                env.deal_damage(user, user["last_attacker"], dmg, can_be_blocked=True)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的被動技能「絕地反擊」沒有對象反擊。"))

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 21:
            # 剛毅打擊
            dmg = STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            def_buff = DefenseMultiplier(
                multiplier=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF'][0],
                duration=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DURATION'][0],
                stackable=False, source=skill_id)
            env.apply_status(targets[0], def_buff)

        elif skill_id == 22:
            # 不屈意志
            def_buff = DefenseMultiplier(
                multiplier=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF'][0],
                duration=STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DURATION'][0],
                stackable=False, source=skill_id)
            env.apply_status(user, def_buff)
            heal_amount = STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT'][0]
            actual_heal = env.deal_healing(user, heal_amount)

        elif skill_id == 23:
            # 絕地反擊
            pass

class Devour(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=8,
            name="鯨吞",
            base_hp=DEVOUR_VAR['DEVOUR_BASE_HP'][0],
            passive_name="巨鯨",
            passive_desc=f"攻擊時會消耗 {int(DEVOUR_VAR['DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT'][0] * 100)}% 當前生命值。",
            baseAtk=DEVOUR_VAR['DEVOUR_BASE_ATK'][0],
            baseDef=DEVOUR_VAR['DEVOUR_BASE_DEF'][0]
        )

    def passive(self, user, dmg, env):
        # 巨鯨：攻擊時會消耗指定比例當前生命值
        self_damage = int(user["hp"] * DEVOUR_VAR['DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT'][0])
        env.deal_healing(user, self_damage, self_mutilation=True)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 24:
            # 吞裂
            dmg = DEVOUR_VAR['DEVOUR_SKILL_0_DAMAGE'][0] * self.baseAtk
            if random.random() < DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_RATE'][0]:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的技能「吞裂」使用失敗。"))
            else:
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            self.passive(user, dmg, env)

        elif skill_id == 25:
            # 巨口吞世
            if targets[0]["hp"] > user["hp"]:
                dmg = int((user["max_hp"] - user["hp"]) * DEVOUR_VAR['DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER'][0])
            else:
                dmg = int(user["hp"] * DEVOUR_VAR['DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER'][0])
            env.deal_damage(user, targets[0], dmg * self.baseAtk, can_be_blocked=True)
            self.passive(user, dmg, env)

        elif skill_id == 26:
            # 堅硬皮膚
            def_buff = DefenseMultiplier(
                multiplier=DEVOUR_VAR['DEVOUR_SKILL_2_DEFENSE_MULTIPLIER'][0],
                duration=DEVOUR_VAR['DEVOUR_SKILL_2_DURATION'][0],
                stackable=False,
                source=skill_id
            )
            env.apply_status(user, def_buff)


class Ranger(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=9,
            name="荒原遊俠",
            base_hp=RANGER_VAR['RANGER_BASE_HP'][0],
            passive_name="冷箭",
            passive_desc=f"冷箭：受到攻擊時，{int(RANGER_VAR['RANGER_PASSIVE_TRIGGER_RATE'][0] * 100)}% 機率反擊對敵方造成 {RANGER_VAR['RANGER_PASSIVE_DAMAGE'][0]} 點傷害。",
            baseAtk=RANGER_VAR['RANGER_BASE_ATK'][0],
            baseDef=RANGER_VAR['RANGER_BASE_DEF'][0]
        )

    def passive(self, user, targets, env):
        pass

    def damage_taken(self, user, target, env, dmg):
        # 被動觸發反擊
        if random.random() < RANGER_VAR['RANGER_PASSIVE_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「冷箭」觸發！"))
            env.deal_damage(user, target, RANGER_VAR['RANGER_PASSIVE_DAMAGE'][0], can_be_blocked=True)

        # 埋伏觸發反擊
        if user["effect_manager"].get_effects("埋伏") and random.random() < RANGER_VAR['RANGER_SKILL_1_AMBUSH_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 埋伏成功，向敵人發動反擊！"))
            env.deal_damage(user, target, dmg * RANGER_VAR['RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER'][0], can_be_blocked=True)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 27:
            # 續戰攻擊
            times_used = user.get("skills_used", {}).get(skill_id, 0)
            dmg = (RANGER_VAR['RANGER_SKILL_0_DAMAGE'][0] + (RANGER_VAR['RANGER_SKILL_0_BONUS_DAMAGE_PER_USE'][0] * times_used)) * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            user["skills_used"][skill_id] = times_used + 1

        elif skill_id == 28:
            # 埋伏防禦
            def_buff = DefenseMultiplier(
                multiplier=RANGER_VAR['RANGER_SKILL_1_DEFENSE_BUFF'][0],
                duration=RANGER_VAR['RANGER_SKILL_1_DURATION'][0],
                stackable=False,
                source=skill_id
            )
            counter_track = Track(
                name="埋伏",
                duration=RANGER_VAR['RANGER_SKILL_1_DURATION'][0],
                stacks=1,
                source=skill_id,
                stackable=False,
                max_stack=1
            )
            env.apply_status(user, counter_track)
            env.apply_status(user, def_buff)

        elif skill_id == 29:
            # 荒原抗性
            if user["hp"] > RANGER_VAR['RANGER_SKILL_2_HP_COST'][0]:
                env.deal_healing(user, RANGER_VAR['RANGER_SKILL_2_HP_COST'][0], self_mutilation=True)
                immune_damage = ImmuneDamage(
                    duration=RANGER_VAR['RANGER_SKILL_2_DURATION'][0],
                    stackable=False
                )
                immune_control = ImmuneControl(
                    duration=RANGER_VAR['RANGER_SKILL_2_DURATION'][0],
                    stackable=False
                )
                env.apply_status(user, immune_damage)
                env.apply_status(user, immune_control)
            else:
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 嘗試使用「荒原」，但血量不足。"))


class ElementalMage(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=10,
            name="元素法師",
            base_hp=ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_HP'][0],
            passive_name="元素之力",
            passive_desc=f"攻擊時 {int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_TRIGGER_RATE'][0] * 100)}% 機率造成麻痺、冰凍、燃燒其中之一。",
            baseAtk=ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_ATK'][0],
            baseDef=ELEMENTALMAGE_VAR['ELEMENTALMAGE_BASE_DEF'][0]
        )

    def passive(self, user, targets, env):
        if random.random() < ELEMENTALMAGE_VAR['ELEMENTALMAGE_PASSIVE_TRIGGER_RATE'][0]:
            effect = random.choice([Burn(duration=3, stacks=1), Freeze(duration=3, stacks=1), Paralysis(duration=2)])
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「元素之力」觸發！"))
            env.apply_status(targets[0], effect)

    def damage_taken(self, user, target, env, dmg):
        if user["effect_manager"].get_effects("雷霆護甲") and random.random() < ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_PARALYSIS_TRIGGER_RATE'][0]:
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的被動技能「雷霆護甲」觸發！"))
            env.apply_status(target, Paralysis(duration=2))

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 30:
            # 雷霆護甲
            def_buff = DefenseMultiplier(
                multiplier=ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DEFENSE_BUFF'][0],
                duration=ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DURATION'][0],
                stackable=False,
                source=skill_id
            )
            heal = user["max_hp"] * ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_HEAL_PERCENT'][0]
            env.deal_healing(user, heal)
            track = Track(
                name="雷霆護甲",
                duration=ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DURATION'][0],
                stacks=1,
                source=skill_id,
                stackable=False,
                max_stack=99
            )
            env.apply_status(user, def_buff)
            env.apply_status(user, track)

        elif skill_id == 31:
            # 凍燒雷
            dmg = ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DAMAGE'][0] * self.baseAtk
            total_layers = 0
            for effects in targets[0]["effect_manager"].active_effects.values():
                for eff in effects:
                    if eff.name in ["麻痺", "凍結", "燃燒"]:
                        total_layers += eff.stacks
            extra_dmg = ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_ADDITIONAL_DAMAGE'][0] * total_layers + dmg
            env.deal_damage(user, targets[0], extra_dmg, can_be_blocked=True)
            self.passive(user, targets, env)

        elif skill_id == 32:
            # 雷擊術
            dmg = ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            if random.random() < ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE'][0]:
                para_duration = random.choices(
                    ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_DURATIONS'],
                    weights=ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_DURATIONS_WEIGHTS'],
                    k=1
                )[0]
                stun_effect = Paralysis(duration=para_duration)
                env.apply_status(targets[0], stun_effect)
            self.passive(user, targets, env)



class HuangShen(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=11,
            name="荒神",
            base_hp=HUANGSHEN_VAR['HUANGSHEN_BASE_HP'][0],
            baseAtk=HUANGSHEN_VAR['HUANGSHEN_BASE_ATK'][0],
            baseDef=HUANGSHEN_VAR['HUANGSHEN_BASE_DEF'][0],
            passive_name="枯萎之刃",
            passive_desc="隨著造成傷害次數增加，攻擊時額外進行隨機追打，每造成兩次傷害增加一次最高追打機會，追打造成敵方當前生命的 5% 血量；額外追打不會累積傷害次數。",
        )

    def passive(self, user, targets, env):
        bonus_hits = user.get("skills_used", {}).get(33, 0) // HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_THRESHOLD'][0]
        if bonus_hits > 0:
            for _ in range(bonus_hits):
                env.add_event(event=BattleEvent(
                    type="text", text=f"{self.name} 的被動技能「枯萎之刃」觸發！"))
                env.deal_damage(user, targets[0], int(
                    targets[0]["hp"] * HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT'][0]), can_be_blocked=True)

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)

        if skill_id == 33:
            # 枯骨
            user["skills_used"][skill_id] = user.get("skills_used", {}).get(skill_id, 0)
            times = random.randint(*HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'])
            for i in range(times):
                dmg = HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE'][0] * self.baseAtk * (1 - i * HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE_REDUCTION_PER_HIT'][0])
                env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
                self.passive(user, targets, env)
                user["skills_used"][skill_id] += 1

        elif skill_id == 34:
            # 荒原
            time_used = user.get("skills_used", {}).get(skill_id, 0)
            if time_used % 3 == 0:
                atk_buff = DamageMultiplier(
                    multiplier=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_ATK_BUFF'][0],
                    duration=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, atk_buff)
            elif time_used % 3 == 1:
                heal_buff = HealMultiplier(
                    multiplier=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_HEAL_BUFF'][0],
                    duration=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, heal_buff)
            elif time_used % 3 == 2:
                def_buff = DefenseMultiplier(
                    multiplier=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_DEF_BUFF'][0],
                    duration=HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0],
                    stackable=False, source=skill_id)
                env.apply_status(user, def_buff)
            user["skills_used"][skill_id] = time_used + 1

        elif skill_id == 35:
            # 生命逆流
            heal_amount = user.get("skills_used", {}).get(33, 0) * HUANGSHEN_VAR['HUANGSHEN_SKILL_2_HEAL_MULTIPLIER'][0]
            env.deal_healing(user, heal_amount)

class GodOfStar(BattleProfession):
    def __init__(self):
        super().__init__(
            profession_id=12,
            name="星神",
            base_hp=GODOFSTAR_VAR['GODOFSTAR_BASE_HP'][0],
            baseAtk=GODOFSTAR_VAR['GODOFSTAR_BASE_ATK'][0],
            baseDef=GODOFSTAR_VAR['GODOFSTAR_BASE_DEF'][0],
            passive_name="天啟星盤",
            passive_desc="星神在戰鬥中精通增益與減益效果的能量運用。每當場上有一層「能力值增益」或「減益」效果時，每回合會額外對敵方造成 5 點傷害 並恢復 5 點生命值。"
        )

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
        bonus_damage = GODOFSTAR_VAR['GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT'][0] * buff_effect_count
        bonus_heal = GODOFSTAR_VAR['GODOFSTAR_PASSIVE_HEAL_PER_EFFECT'][0] * buff_effect_count
        env.add_event(event=BattleEvent(
            type="text", text=f"{self.name} 的被動技能「天啟星盤」觸發！"))
        return bonus_damage, bonus_heal

    def apply_skill(self, skill_id, user, targets, env):
        super().apply_skill(skill_id, user, targets, env)
        dmg, heal = self.passive(user, targets, env)
        if skill_id == 36:
            dmg += GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            buff = random.choice([
                DamageMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id),
                DefenseMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id),
                HealMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0], stackable=False, source=skill_id)
            ])
            env.apply_status(targets[0], buff)
            env.deal_healing(user, heal)

        elif skill_id == 37:
            heal += GODOFSTAR_VAR['GODOFSTAR_SKILL_1_HEAL'][0]
            env.deal_healing(user, heal)
            debuff = random.choice([
                DamageMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id),
                DefenseMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id),
                HealMultiplier(multiplier=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER'][0], duration=GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0], stackable=False, source=skill_id)
            ])
            env.apply_status(user, debuff)

        elif skill_id == 38:
            dmg *= GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0]
            heal *= GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0]
            env.add_event(event=BattleEvent(
                type="text", text=f"{self.name} 的技能「虛擬創星圖」強化了天啟星盤的力量，增加了傷害和回復。"))
            dmg += GODOFSTAR_VAR['GODOFSTAR_SKILL_2_DAMAGE'][0] * self.baseAtk
            env.deal_damage(user, targets[0], dmg, can_be_blocked=True)
            env.deal_healing(user, heal)
