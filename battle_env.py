# battle_env.py

import gym
import numpy as np
from status_effects import (
    Burn, Poison, Freeze, DamageMultiplier, DefenseMultiplier, HealMultiplier,
    ImmuneDamage, ImmuneControl, BleedEffect, Stun, StatusEffect,HealthPointRecover,MaxHPmultiplier, Track,Paralysis
    , EFFECT_NAME_MAPPING
)
from professions import (
    Paladin, Mage,Assassin,Archer, Berserker,DragonGod,BloodGod,
    SteadfastWarrior,SunWarrior,Ranger,ElementalMage,HuangShen,
    GodOfStar
)
from effect_manager import EffectManager
from skills import SkillManager
import random
from effect_mapping import EFFECT_VECTOR_LENGTH  # 引入效果向量長度
from gym import spaces

class BattleEnv(gym.Env):
    """
    支援 1v1 的雙方AI控制環境。
    - obs shape = (team_size + enemy_team_size) * per_character_obs + 1
        每個角色包含:
            hp, profession_id, max_hp, base_hp, is_defending,
            damage_multiplier, defend_multiplier, heal_multiplier,
            cooldowns (3個技能),
            effect_manager 輸出 (固定長度，來自 effect_mapping.py)
    - action shape = [3] * (team_size + enemy_team_size)
        每個角色有3個選項 (0, 1, 2)
    - 20 回合後毒圈 => 每回合扣 10*(回合-20).
    - 每回合結束 => is_defending 重置
    - 戰鬥超過10回合後，每過2回合，雙方的傷害係數增加5%
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, team_size, enemy_team_size, max_rounds,
                    player_team, enemy_team, skill_mgr, show_battle_log=True):
            super(BattleEnv, self).__init__()
            self.team_size = team_size
            self.enemy_team_size = enemy_team_size
            self.max_rounds = max_rounds
            self.player_team = player_team
            self.enemy_team = enemy_team
            self.skill_mgr = skill_mgr
            self.show_battle_log = show_battle_log 
            self.round_count = 1
            self.done = False
            self.battle_log = []
            self.damage_coefficient = 1.0

            # (1) 這裡只是示範: 把 obs_space 變成 Dict，裡面含 2 個 Box(玩家、敵人) + 2 個 Box(action_mask)
            # 你也可以用原本的 obs_space = spaces.Box(...) 方式，但要確保 step() 回傳 shape 對得上
            self.observation_space = spaces.Dict({
                "player_obs": spaces.Box(low=-100000, high=100000, shape=(10,), dtype=np.float32),
                "opponent_obs": spaces.Box(low=-100000, high=100000, shape=(10,), dtype=np.float32),
                "player_action_mask": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
                "opponent_action_mask": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
            })

            # 動作空間(每個智能體可以選 [0,1,2])
            self.action_space = spaces.Dict({
                "player_action": spaces.Discrete(3),
                "opponent_action": spaces.Discrete(3)
            })
    def reset(self):
        self.round_count = 1
        self.done = False
        self.battle_log.clear()
        self.damage_coefficient = 1.0

        # 重置我方隊伍
        for m in self.player_team:
            prof = m["profession"]
            m["hp"] = prof.base_hp
            m["max_hp"] = prof.base_hp
            m["times_healed"] = 0
            m["skip_turn"] = False
            m["is_defending"] = False
            m["damage_multiplier"] = 1.0
            m["defend_multiplier"] = 1.0
            m["heal_multiplier"] = 1.0
            m["battle_log"] = []
            m["skills_used"] = {} 
            m["cooldowns"] = {0:0, 1:0, 2:0}
            
            m["skills_used"] = {}  # 用於追蹤技能使用次數
            m["effect_manager"] = EffectManager(m)  # 初始化 EffectManager
            m["last_attacker"] = None
            m["last_damage"] = 0
            m['accumulated_damage'] = 0

        # 重置敵方
        for e in self.enemy_team:
            prof = e["profession"]
            e["hp"] = prof.base_hp
            e["max_hp"] = prof.base_hp
            e["skip_turn"] = False
            e["is_defending"] = False
            e["times_healed"] = 0
            e["damage_multiplier"] = 1.0
            e["defend_multiplier"] = 1.0
            e["heal_multiplier"] = 1.0
            e["battle_log"] = []
            e["cooldowns"] = {0:0, 1:0, 2:0}

            e["skills_used"] = {} 
            e["effect_manager"] = EffectManager(e)  # 初始化 EffectManager
            e["last_attacker"] = None
            e["last_damage"] = 0
            e['accumulated_damage'] = 0

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """
        回傳同時給AI1、AI2的觀測。 
        同時要加上 action_mask，表示哪些動作可用(沒進CD) => 1/0
        """
        # 這裡示範做得很簡單：只回傳 [ player_hp, enemy_hp, round_count ] 等資訊
        # 你可以自行加回你原本的觀察資訊(例如各種buff等)
        player_m = self.player_team[0]
        enemy_m = self.enemy_team[0]

        # 觀察值(可自行擴充)
        player_obs = np.array([
            player_m["hp"],
            player_m["max_hp"],
            enemy_m["hp"],
            enemy_m["max_hp"],
            self.round_count,
            # ... 你想加的資訊 ...
            0,0,0,0,0  # filler for shape=(10,) 
        ], dtype=np.float32)

        opponent_obs = np.array([
            enemy_m["hp"],
            enemy_m["max_hp"],
            player_m["hp"],
            player_m["max_hp"],
            self.round_count,
            # ... 你想加的資訊 ...
            0,0,0,0,0
        ], dtype=np.float32)

        # 行動遮罩(3招)
        player_mask = self._get_action_mask(player_m)
        opponent_mask = self._get_action_mask(enemy_m)

        return {
            "player_obs": player_obs,
            "opponent_obs": opponent_obs,
            "player_action_mask": player_mask,
            "opponent_action_mask": opponent_mask
        }
        
        
    def _get_action_mask(self, member):
      """
      回傳 [3]，其中 1 代表該技能可用，0 代表在冷卻或其他條件無法使用
      """
      mask = np.array([1,1,1], dtype=np.int8)
      for skill_id, cd_val in member["cooldowns"].items():
          if cd_val > 0:
              mask[skill_id] = 0
      return mask
  
    def step(self, actions):
        self._print_round_header()
        self._print_teams_hp()
        
        pa = actions['player_action']
        ea = actions['opponent_action']
        
        player_actions = [pa]
        enemy_actions = [ea]
   

        # 回合開始時處理被動技能
        self._process_passives()
        
        # 回合初始化
        for m in self.player_team + self.enemy_team:
            m["last_damage"] = 0
            m["last_attacker"] = None
            m["accumulated_damage"] = 0
            
        # 分類技能
        player_effect_skills = []
        player_heal_skills = []
        player_damage_skills = []

        enemy_effect_skills = []
        enemy_heal_skills = []
        enemy_damage_skills = []
        

        
        # 分類我方技能
        for i in range(self.team_size):
            user = self.player_team[i]
            # 需要檢查沒有免疫控制
            if user["hp"] > 0 and not user["effect_manager"].has_effect('免疫控制') :
                # 麻痺 or 暈眩
                if user["skip_turn"] and (user["effect_manager"].has_effect('麻痺') or user["effect_manager"].has_effect('暈眩')):
                    self.battle_log.append(f"【{user['profession'].name}】，因異常狀態跳過行動")
                    continue
                # 冰凍的跳過
                elif user["skip_turn"]:
                    self.battle_log.append(f"【{user['profession'].name}】，因異常狀態跳過行動")
                    user["skip_turn"] = False
                    # remove freeze effect
                    user["effect_manager"].remove_all_effects('凍結')
                    continue

            skill_index = int(player_actions[i])
            # 根據職業ID計算實際技能ID
            profession_id = user["profession"].profession_id
            skill_id = 3 * profession_id + skill_index

            available_skills = user["profession"].get_available_skill_ids(user["cooldowns"])
            if skill_id not in available_skills:
                # 如果選擇的技能不可用，隨機選擇可用技能
                penalty -= 5
                available_skill_indices = [sid - 3 * profession_id for sid in available_skills]
                if available_skill_indices:
                    skill_index = random.choice(available_skill_indices)
                    skill_id = 3 * profession_id + skill_index
                    skill_name = self.skill_mgr.get_skill_name(skill_id)
                    self.battle_log.append(
                        f"P隊伍{i}({user['profession'].name}) 選擇的技能不可用，隨機選擇 {skill_name}"
                    )
                else:
                    # 如果沒有可用技能，跳過行動
                    self.battle_log.append(
                        f"P隊伍{i}({user['profession'].name}) 沒有可用技能，跳過行動"
                    )
                    continue
            else:
                skill_name = self.skill_mgr.get_skill_name(skill_id)

            # 分類技能
            skill_type = self.skill_mgr.get_skill_type(skill_id)
            if skill_type == 'effect':
                player_effect_skills.append((i, skill_id))
            elif skill_type == 'heal':
                player_heal_skills.append((i, skill_id))
            elif skill_type == 'damage':
                player_damage_skills.append((i, skill_id))
            else:
                # 未知類型，預設為傷害類
                player_damage_skills.append((i, skill_id))

        # 分類敵方技能
        for j in range(self.enemy_team_size):
            e = self.enemy_team[j]
            # 需要檢查沒有免疫控制
            if e["hp"] > 0 and not e["effect_manager"].has_effect('免疫控制') :
                # 麻痺 or 暈眩
                if e["skip_turn"] and (e["effect_manager"].has_effect('麻痺') or e["effect_manager"].has_effect('暈眩')):
                    self.battle_log.append(f"【{e['profession'].name}】，因異常狀態跳過行動")
                    continue
                # 冰凍的跳過
                elif e["skip_turn"]:
                    self.battle_log.append(f"【{e['profession'].name}】，因異常狀態跳過行動")
                    e["skip_turn"] = False
                    # remove freeze effect
                    e["effect_manager"].remove_all_effects('凍結')
                    
                    continue

            skill_index = int(enemy_actions[j])
            # 根據職業ID計算實際技能ID
            profession_id = e["profession"].profession_id
            skill_id = 3 * profession_id + skill_index

            available_skills = e["profession"].get_available_skill_ids(e["cooldowns"])
            if skill_id not in available_skills:
                # 如果選擇的技能不可用，隨機選擇可用技能
                available_skill_indices = [sid - 3 * profession_id for sid in available_skills]
                penalty -= 5
                if available_skill_indices:
                    skill_index = random.choice(available_skill_indices)
                    skill_id = 3 * profession_id + skill_index
                    skill_name = self.skill_mgr.get_skill_name(skill_id)
                    self.battle_log.append(
                        f"E隊伍{j}({e['profession'].name}) 選擇的技能不可用，隨機選擇 {skill_name}"
                    )
                else:
                    # 如果沒有可用技能，跳過行動
                    self.battle_log.append(
                        f"E隊伍{j}({e['profession'].name}) 沒有可用技能，跳過行動"
                    )
                    continue
            else:
                skill_name = self.skill_mgr.get_skill_name(skill_id)
    
            # 分類技能
            skill_type = self.skill_mgr.get_skill_type(skill_id)
            if skill_type == 'effect':
                enemy_effect_skills.append((j, skill_id))
            elif skill_type == 'heal':
                enemy_heal_skills.append((j, skill_id))
            elif skill_type == 'damage':
                enemy_damage_skills.append((j, skill_id))
            else:
                # 未知類型，預設為傷害類
                enemy_damage_skills.append((j, skill_id))

        # 處理技能順序：先效果類，再治療類，最後傷害類
        # 1. 我方效果技能
        for i, skill_id in player_effect_skills:
            user = self.player_team[i]
            targets = self._select_targets(self.enemy_team)
            if targets:
                user["profession"].apply_skill(skill_id, user, targets, self)
            else:
                # 敵方全滅，結束遊戲
                self.done = True

        # 2. 敵方效果技能
        for j, skill_id in enemy_effect_skills:
            e = self.enemy_team[j]
            targets = self._select_targets(self.player_team)
            if targets:
                e["profession"].apply_skill(skill_id, e, targets, self)
            else:
                # 我方全滅，結束遊戲
                self.done = True

        # 3. 我方治療技能
        for i, skill_id in player_heal_skills:
            user = self.player_team[i]
            targets = self._select_heal_targets(user)
            if targets:
                user["profession"].apply_skill(skill_id, user, targets, self)
            else:
                # 治療目標不存在，可能無需額外處理
                pass

        # 4. 敵方治療技能（如有）
        for j, skill_id in enemy_heal_skills:
            e = self.enemy_team[j]
            targets = self._select_heal_targets(e)
            if targets:
                e["profession"].apply_skill(skill_id, e, targets, self)
            else:
                # 治療目標不存在，可能無需額外處理
                pass

        # 5. 我方傷害技能
        for i, skill_id in player_damage_skills:
            user = self.player_team[i]
            targets = self._select_targets(self.enemy_team)
            if targets:
                user["profession"].apply_skill(skill_id, user, targets, self)
            else:
                # 敵方全滅，結束遊戲
                self.done = True

        # 6. 敵方傷害技能
        for j, skill_id in enemy_damage_skills:
            e = self.enemy_team[j]
            targets = self._select_targets(self.player_team)
            if targets:
                e["profession"].apply_skill(skill_id, e, targets, self)
            else:
                # 我方全滅，結束遊戲
                self.done = True



        # check 我方是否全滅
        if all(m["hp"] <= 0 for m in self.player_team):
            self.done = True
            self.battle_log.append("我方全滅，敵方獲勝！")
        # check 敵方是否全滅
        if all(e["hp"] <= 0 for e in self.enemy_team):
            self.done = True
            self.battle_log.append("敵方全滅，我方獲勝！")
        # check 雙方全滅?
        if self._check_both_defeated():
            self.done = True
            self.battle_log.append("雙方全滅，平手！")
        
        #  最後階段的處理(冷卻，回合狀態，回合末端技能)
        if not self.done:
            player_all_skill = player_effect_skills + player_heal_skills + player_damage_skills
            enemy_all_skill = enemy_effect_skills + enemy_heal_skills + enemy_damage_skills
            self._process_passives_end_of_turn(player_all_skill, enemy_all_skill)
            self._handle_status_end_of_turn()
            self._manage_cooldowns()
        # 增加戰鬥特性：超過10回合後，每過2回合，雙方的傷害係數增加10%
            if self.round_count > 10 and (self.round_count - 10) % 2 == 0:
                self.damage_coefficient *= 1.1
                self.battle_log.append(
                f"戰鬥超過10回合，雙方的傷害係數增加了10%，現在為 {self.damage_coefficient:.2f}。")
        # 強制停止
        self.round_count += 1
        if self.round_count > self.max_rounds:
                done = True
                

        self._print_round_footer()   

        reward_player = self._get_reward(player=True)
        reward_opponent = self._get_reward(player=False)
        obs = self._get_obs()
        return obs, {"player_reward": reward_player, "opponent_reward": reward_opponent}, self.done, {}

    def _check_both_defeated(self):
        """
        檢查雙方是否都已被擊敗
        """
        player_defeated = all(m["hp"] <= 0 for m in self.player_team)
        enemy_defeated = all(e["hp"] <= 0 for e in self.enemy_team)
        return player_defeated and enemy_defeated

    def deal_damage(self, user, target, dmg, can_be_blocked=True):
        """
        處理傷害邏輯，包括防禦、buff效果等
        """
        # 先處理凍結觸發
        if target["effect_manager"].has_effect('凍結'):
            freeze_effects = target["effect_manager"].get_effects('凍結')
            # freeze stacks
            total_freeze_layers = sum([e.stacks for e in freeze_effects])
            chance = 0.15 * total_freeze_layers
            # enemy not immune to control
            if random.random() < chance and not target["effect_manager"].has_effect('免疫控制'):
                self.battle_log.append(
                    f"{target['profession'].name} 被凍結，將跳過下一回合的行動。"
                )
                # 移除所有凍結
                target["effect_manager"].remove_all_effects('凍結')

        # 檢查目標是否免疫傷害
        if target["effect_manager"].has_effect('免疫傷害'):
            self.battle_log.append(f"{target['profession'].name} 免疫所有傷害!")
            return

        if can_be_blocked and target["is_defending"]:
            self.battle_log.append(f"{target['profession'].name} 擋下了攻擊!")
            return

        # 應用傷害增減比例
        dmg = int(dmg * user.get("damage_multiplier", 1.0))

        # 處理防禦增減
        dmg = int(dmg / target.get("defend_multiplier", 1.0))
        
        dmg *= self.damage_coefficient  # 考慮戰鬥特性：傷害係數
        
        dmg = max(1, dmg)  # 至少造成1點傷害
        
        dmg = int(dmg)
        # 實際減血
        target["hp"] = max(0, target["hp"] - dmg)
        target['accumulated_damage'] += dmg
        self.battle_log.append(
            f"{user['profession'].name} 對 {target['profession'].name} 造成 {dmg} 點傷害 (剩餘HP={int(target['hp'])})"
        )
        
        # 如果職業是荒原遊俠
        # 進行冷箭判定冷箭：受到攻擊時，20%機率反擊45點傷害。
        if target["profession"].name == "荒原遊俠":
            if random.random() < 0.25:
                self.battle_log.append(
                    f"{target['profession'].name} 發動「冷箭」反擊！"
                )
                self.deal_damage(target, user, 45, can_be_blocked=False)
        
        
        # update last attacker and last damage
        if user in self.player_team:
            target["last_attacker"] = user
            target["last_damage"] = dmg

    def deal_healing(self, user, heal_amount, rate = 0,heal_damage=False,target=None,self_mutilation = False):
        """
        處理治療邏輯，包括超過最大生命的處理
        # heal_damage 與 self_mutilation 不能同時為True
        """
        if user["hp"] <= 0 :
            self.battle_log.append(f"{user['profession'].name} 已被擊敗，無法恢復生命值。")
            return
        # 首先進行治療增減比例的應用
        if self_mutilation and heal_damage:
            raise ValueError("heal_damage 與 self_mutilation 不能同時為True")
        
        if not self_mutilation:
            heal_amount = int(heal_amount * user.get("heal_multiplier", 1.0))
            # 超出最大血量的治療量
            extra_heal = 0
            if user["hp"] + heal_amount > user["max_hp"]:
                extra_heal = user["hp"] + heal_amount - user["max_hp"]
                heal_amount = user["max_hp"] - user["hp"]
            # 實際治療
            user["hp"] = min(user["hp"] + heal_amount, user["max_hp"])
            self.battle_log.append(f"{user['profession'].name} 恢復了 {heal_amount} 點生命值 (剩餘HP={int(user['hp'])})")
        # 接著如果 heal_damage 為 True，則對目標造成治療量的rate%傷害
        if heal_damage and target:
            dmg = int(extra_heal * rate) 
            self.battle_log.append(f"{user['profession'].name} 治療溢出造成傷害！ ")
            self.deal_damage(user, target, dmg, can_be_blocked=False)
        # 最後如果 self_mutilation 為 True，則對自己造成治療量的傷害(但不會低於最低血量)
        if self_mutilation:
            dmg = int(heal_amount)
            user["hp"] = max(1, user["hp"] - dmg)
            self.battle_log.append(f"{user['profession'].name} 自傷 {dmg} 點生命值 (剩餘HP={int(user['hp'])})")
            # 累積傷害
            user['accumulated_damage'] += dmg

    def set_status(self, target, status_name,stacks,source = None):
        """
        設置特定status_id 的stack
        """
        target["effect_manager"].set_effect_stack(status_name,target,stacks,source)
        
    def apply_status(self, target, status_effect):
        """
        應用異常狀態
        """
        target["effect_manager"].add_effect(status_effect)

        # 特殊被動技能處理
        for m in self.player_team + self.enemy_team:
            if m["profession"].name == "烈陽勇士" and target in self.enemy_team:
                m["hp"] = min(m["hp"] + 5, m["max_hp"])
                self.battle_log.append(
                    f"{m['profession'].name} 被動技能「太陽庇佑」回復5點生命值。"
                )

    def _handle_status_end_of_turn(self):
        """
        處理每回合結束時的異常狀態效果
        """
        alist = self.player_team + self.enemy_team
        for m in alist:
            if m["hp"] <= 0:
                continue
            m["effect_manager"].tick_effects()

        # 重置一次性buff
        for m in alist:
            m["is_defending"] = False
            


        # 合併各角色的 battle_log 到 env.battle_log
        for m in alist:
            if m["battle_log"]:
                self.battle_log.extend(m["battle_log"])
                m["battle_log"] = []  # 清空角色的 battle_log

    def _process_passives(self):
        """
        處理每回合開始時的被動技能
        """
        for m in self.player_team + self.enemy_team:
            if m["hp"] <= 0:
                continue
            profession = m["profession"]
            if profession.name == "剛毅武士":
                # baattle log
                m["battle_log"].append(
                    f"{profession.name} 的被動技能「堅韌壁壘」發動！"
                )
                heal = int((profession.max_hp - m['hp']) * 0.1)
                self.deal_healing(m, heal)
            elif profession.name == "烈陽勇士":
                # 太陽庇佑被動：每當敵方單位進入異常狀態時，回復自身5點生命值。
                # 這部分已在 apply_status 中處理
                pass
            elif profession.name == "血神":
                # 血神被動：受到致死傷害時，5%機率免疫該次傷害
                # 在 deal_damage 方法中實作
                pass
            elif profession.name == "元素法師":
                # 元素法師被動已在技能中處理
                pass
            elif profession.name == "龍神":
                self.battle_log.append(f"{profession.name} 神血回歸，變得更加強大！")
                # add 3% max hp, 3% attack, 3% defense for each stacks of dragon god
                # source = -1 是此被動技能的skill_id
                passive_id = profession.default_passive_id
                deffect = DefenseMultiplier(multiplier=1.03,duration=99,stacks=1,source=passive_id,stackable=True,max_stack=99)
                heffect = DamageMultiplier(multiplier=1.03,duration=99,stacks=1,source=passive_id,stackable=True,max_stack=99)
                # hpeffect = MaxHPmultiplier(multiplier=1.02,duration=99,stacks=1,source=passive_id,stackable=True,max_stack=99)
                track = Track(name="龍神buff",duration=99,stacks=1,source=passive_id,stackable=True,max_stack=99)
                self.apply_status(m,deffect)
                self.apply_status(m,heffect)
                self.apply_status(m,track)
     
    def _process_passives_end_of_turn(self,mSkill = None,eSkill = None):
        """
        # take all 
        """ 
        # 我方處理       
        for i, m in mSkill:
            # 假如是荒神
            p = self.player_team[i]
            if p["profession"].name == "荒神":
                # 如果p的累積傷害超過了自身最大生命值的35%，則觸發荒神的被動技能：在回合結束時回復已損生命的5%
                if p['accumulated_damage'] >= p['max_hp'] * 0.35:
                    heal = int(p['accumulated_damage'] * 0.05)
                    self.battle_log.append(
                        f"{p['profession'].name} 的被動技能「荒蕪」發動！"
                    )
                    self.deal_healing(p, heal)
                    p['accumulated_damage'] = 0
            
            
            # 烈陽勇士 技能 25 => 本回合防禦力增加30%，並對攻擊者附加1層燃燒
            if m == 25:
                # 取得當前玩家角色
                player = self.player_team[i]
                # 檢查該角色的最後攻擊者是否存在
                if player["last_attacker"]:
                    # 添加燃燒效果
                    burn = Burn(duration=3, stacks=1)
                    self.apply_status(player["last_attacker"], burn)
                    self.battle_log.append(
                        f"{player['profession'].name} 「日冕之盾」對攻擊者附加了燃燒效果！"
                    )
            
            # 雷霆護甲：2回合內，受到傷害時一定機率直接麻痺敵人。
            if m == 30:
                player = self.player_team[i]
                if player["last_attacker"]:
                    if random.random() < 0.3:
                        self.battle_log.append(
                            f"{player['profession'].name} 發動「雷霆護甲」麻痺了攻擊者！"
                        )
                        self.apply_status(player["last_attacker"], Paralysis(duration=2))
            
            # 絕地反擊 23 => 對攻擊者立即造成其本次攻擊傷害的200%，此技能需冷卻3回合
            if m == 23:
                player = self.player_team[i]
                if player["last_attacker"]:
                    dmg = player["last_damage"] * 2
                    self.battle_log.append(
                        f"{player['profession'].name} 啟動「絕地反擊」。"
                    )
                    self.deal_damage(player, player["last_attacker"], dmg)
                    # 進入冷卻
                    player["cooldowns"][2] = 3
                    self.battle_log.append(f"「絕地反擊」進入 3 回合的冷卻。")
                else:
                    self.battle_log.append(f"「絕地反擊」沒有攻擊者可反擊。")

        for i, e in eSkill:
            # 假如是荒神
            p = self.enemy_team[i]
            if p["profession"].name == "荒神":
                # 如果p的累積傷害超過了自身最大生命值的35%，則觸發荒神的被動技能：在回合結束時回復已損生命的5%
                if p['accumulated_damage'] >= p['max_hp'] * 0.35:
                    heal = int(p['accumulated_damage'] * 0.05)
                    self.battle_log.append(
                        f"{p['profession'].name} 的被動技能「荒蕪」發動！"
                    )
                    self.deal_healing(p, heal)
                    p['accumulated_damage'] = 0
            
            # 烈陽勇士 技能 25 => 本回合防禦力增加30%，並對攻擊者附加1層燃燒
            if e == 25:
                enemy = self.enemy_team[i]
                if enemy["last_attacker"]:
                    burn = Burn(duration=3, stacks=1)
                    self.apply_status(enemy["last_attacker"], burn)
                    self.battle_log.append(
                        f"{enemy['profession'].name} 「烈陽之盾」對攻擊者附加了燃燒效果！"
                    )
            
            # 雷霆護甲：2回合內，受到傷害時一定機率直接麻痺敵人。
            if e == 30:
                enemy = self.enemy_team[i]
                if enemy["last_attacker"]:
                    if random.random() < 0.3:
                        self.battle_log.append(
                            f"{enemy['profession'].name} 發動「雷霆護甲」麻痺了攻擊者！"
                        )
                        self.apply_status(enemy["last_attacker"], Paralysis(duration=2))
            
            # 絕地反擊 23 => 對攻擊者立即造成其本次攻擊傷害的200%，此技能需冷卻3回合
            if e == 23:
                enemy = self.enemy_team[i]
                if enemy["last_attacker"]:
                    dmg = enemy["last_damage"] * 2
                    self.battle_log.append(
                        f"{enemy['profession'].name} 啟動「絕地反擊」。"
                    )
                    self.deal_damage(enemy, enemy["last_attacker"], dmg)
                    # 進入冷卻
                    enemy["cooldowns"][2] = 3
                    self.battle_log.append(f"「絕地反擊」進入 3 回合的冷卻。")
                else:
                    self.battle_log.append(f"「絕地反擊」沒有攻擊者可反擊。")
                                 
    def _select_targets(self, team):
        """
        選擇目標：選擇HP最高的敵人
        """
        alive = [member for member in team if member["hp"] > 0]
        if not alive:
            return []
        # 選擇HP最高的
        target = max(alive, key=lambda x: x["hp"])
        return [target]

    def _select_heal_targets(self, user):
        """
        選擇治療目標：通常是自己或隊伍中HP最低的隊友
        這裡假設治療技能只治療自己
        """
        return [user]

    def _get_reward(self, player=True):
        """
        計算獎勵，根據玩家或對手
        """
        # 勝利/失敗獎勵
        if self.done:
            if player:
                if all(m["hp"] <= 0 for m in self.player_team) and all(e["hp"] <= 0 for e in self.enemy_team):
                    return 0
                elif all(m["hp"] <= 0 for m in self.player_team):
                    return -50
                elif all(e["hp"] <= 0 for e in self.enemy_team):
                    return 50
            else:
                if all(m["hp"] <= 0 for m in self.player_team) and all(e["hp"] <= 0 for e in self.enemy_team):
                    return 0
                elif all(m["hp"] <= 0 for m in self.player_team):
                    return 50
                elif all(e["hp"] <= 0 for e in self.enemy_team):
                    return -50
        else:
            # 可以根據每回合的表現給予獎勵
            # 例如，造成傷害給予正獎勵，受到傷害給予負獎勵
            reward = 0
            if player:
                # 計算玩家的傷害和受到的傷害
                # reward += player_damage_done * 0.1
                # reward -= player_damage_received * 0.1
                reward += sum([m['accumulated_damage'] for m in self.enemy_team]) * 0.1
                # 如果血量比對方高，給予額外獎勵 +2
                if sum([m['hp'] for m in self.player_team]) > sum([m['hp'] for m in self.enemy_team]):
                    reward += 2
 
            else:
                # 計算對手的傷害和受到的傷害
                # reward += opponent_damage_done * 0.1
                # reward -= opponent_damage_received * 0.1
                reward += sum([m['accumulated_damage'] for m in self.player_team]) * 0.1
                # 如果血量比對方高，給予額外獎勵 +2
                if sum([m['hp'] for m in self.enemy_team]) > sum([m['hp'] for m in self.player_team]):
                    reward += 2
            return reward
        return 0


    def _print_round_header(self):
        if self.show_battle_log:
            print(f"\n---\n回合 {self.round_count} :\n")

    def _print_round_footer(self):
        if self.show_battle_log:
            for line in self.battle_log:
                print(line)
            self.battle_log = []
            print("回合結束\n---")

    def _print_teams_hp(self):
        if not self.show_battle_log:
            return

        def hp_bar(hp, maxhp, length=20):
            ratio = hp / maxhp
            ratio = max(0, min(ratio, 1))
            bar_num = int(ratio * length)
            return "[" + "#" * bar_num + " " * (length - bar_num) + f"] {int(hp)}/{int(maxhp)}"

        def format_effect_details(effect_id, effs):
            """格式化每個效果的詳細資訊，包括名稱、層數、持續時間"""
            stacks = 0
            effect_name = EFFECT_NAME_MAPPING.get(effect_id, '未知效果')
            if  len(effs)>0:
                stacks = effs[0].stacks  # 當前層數
            else:
                stacks = 0
            durations = ", ".join([str(eff.duration) for eff in effs])  # 剩餘回合數
            return f"{effect_name}({stacks}層, 持續: {durations}回合)"

        for i, m in enumerate(self.player_team):
            maxhp = m["profession"].max_hp
            bar = hp_bar(m["hp"], maxhp)
            # 獲取所有狀態名稱、層數和持續回合
            status_details = [
                format_effect_details(effect_id, effs)
                for effect_id, effs in m["effect_manager"].active_effects.items()
            ]
            status_str = " | ".join([f"【{s}】" for s in status_details])
            print(f"P{i} {m['profession'].name} HP {bar} {status_str}")

        # 敵方隊伍
        for j, e in enumerate(self.enemy_team):
            maxhp = e["profession"].max_hp
            bar = hp_bar(e["hp"], maxhp)
            # 獲取所有狀態名稱、層數和持續回合
            status_details = [
                format_effect_details(effect_id, effs)
                for effect_id, effs in e["effect_manager"].active_effects.items()
            ]
            status_str = " | ".join([f"【{s}】" for s in status_details])
            print(f"E{j} {e['profession'].name} HP {bar} {status_str}")

        print()

    def _manage_cooldowns(self):
        """
        管理技能冷卻，每回合減少冷卻回合數
        """
        for m in self.player_team + self.enemy_team:
            for skill_id in m["cooldowns"]:
                if m["cooldowns"][skill_id] > 0:
                    m["cooldowns"][skill_id] -= 1

    def _check_both_defeated(self):
        """
        檢查雙方是否都已被擊敗
        """
        player_defeated = all(m["hp"] <= 0 for m in self.player_team)
        enemy_defeated = all(e["hp"] <= 0 for e in self.enemy_team)
        return player_defeated and enemy_defeated
