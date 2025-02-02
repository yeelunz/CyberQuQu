# battle_env.py

import gymnasium
import numpy as np
from .status_effects import (
    Burn, Poison, Freeze, DamageMultiplier, DefenseMultiplier, HealMultiplier,
    ImmuneDamage, ImmuneControl, BleedEffect, StatusEffect, HealthPointRecover, MaxHPmultiplier, Track, Paralysis,
    EFFECT_NAME_MAPPING
)

from .effect_manager import EffectManager
from .skills import SkillManager
import random
from .effect_mapping import EFFECT_VECTOR_LENGTH  # 引入效果向量長度

from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Dict
from ray.rllib.env import MultiAgentEnv

from .train_var import player_train_times, enemy_train_times
from .battle_event import BattleEvent

import sys

class BattleEnv(MultiAgentEnv):
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

    def __init__(self, config):
        # self, team_size, enemy_team_size, max_rounds,
        #             player_team, enemy_team, skill_mgr, show_battle_log=True
         
            super(BattleEnv, self).__init__()
            self.config = config
      
            # init from config
            self.team_size = config["team_size"]
            self.enemy_team_size = config["enemy_team_size"]
            self.max_rounds = config["max_rounds"]
            self.player_team = config["player_team"]
            self.enemy_team = config["enemy_team"]
            self.skill_mgr = config["skill_mgr"]
            self.show_battle_log = config["show_battle_log"]
            self.round_count = 1
            self.done = False
            self.battle_log = []
            self.damage_coefficient = 1.0
            self.size = 160
            self.train_mode = config["train_mode"]
            self.mpro = []
            self.epro = []
            all_professions = self.config["all_professions"]
            # 為了標準化觀察空間，需要知道最大的 base_hp
            self.maxProfession_hp = 0
            for m in all_professions:
                if m.base_hp > self.maxProfession_hp:
                    self.maxProfession_hp = m.base_hp
                

            self.agents = ["player", "enemy"]
            # 為每個代理配置觀察空間和動作空間
            self.observation_space = Box(low=-1e4, high=1e4, shape=(self.size,), dtype=np.float32)
            self.action_space = Discrete(3)

    def reset(self,*, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.round_count = 1
        self.done = False
        self.battle_log.clear()
        self.damage_coefficient = 1.0
        all_professions = self.config["all_professions"]
        if self.train_mode:
            for m in self.player_team + self.enemy_team:
                m["profession"] = random.choice(all_professions)
                
        # 重置我方隊伍
        for m in self.player_team:
            prof = m["profession"]
            # self.mpro.append(prof.name)
            player_train_times[prof.profession_id] += 1
            m["hp"] = prof.base_hp
            m["max_hp"] = prof.base_hp
            m["times_healed"] = 0
            m["skip_turn"] = False
            m["is_defending"] = False
            m["damage_multiplier"] = 1.0
            m["defend_multiplier"] = 1.0
            m["heal_multiplier"] = 1.0
            m["skills_used"] = {} 
            m["cooldowns"] = {0:0, 1:0, 2:0}
            m["last_skill_used"] = None
            m["skills_used"] = {}  # 用於追蹤技能使用次數
            m["effect_manager"] = EffectManager(target=m, env=self)  # 初始化 EffectManager
            m["last_attacker"] = None
            m["last_damage_taken"] = 0
            m['accumulated_damage'] = 0
            m['private_info'] = {}
            m['last_skill_used'] = [0,0,0]
        
        # 重置敵方
        for e in self.enemy_team:
            prof = e["profession"]
            # self.epro.append(prof.name)
            enemy_train_times[prof.profession_id] += 1
            e["hp"] = prof.base_hp
            e["max_hp"] = prof.base_hp
            e["skip_turn"] = False
            e["is_defending"] = False
            e["times_healed"] = 0
            e["damage_multiplier"] = 1.0
            e["defend_multiplier"] = 1.0
            e["heal_multiplier"] = 1.0
            e["cooldowns"] = {0:0, 1:0, 2:0}
            e["last_skill_used"] = None 
            e["skills_used"] = {} 
            e["effect_manager"] = EffectManager(target=e, env=self)  # 初始化 EffectManager
            e["last_attacker"] = None
            e["last_damage_taken"] = 0
            e['accumulated_damage'] = 0
            e['private_info'] = {}
            e['last_skill_used'] = [0,0,0]

        infos = {
             "player": {"professions":self.mpro},
            "enemy": {"professions":self.epro},
            }
        return {
            "player": self._get_obs("player"),
            "enemy": self._get_obs("enemy"),
        }, infos
        
    def add_event(self, user=None, target=None, event=None):
        """
        這個取代原本的add battle_log

        - type: str, 事件類型
            其中包含以下事件類型:
            - damage: 傷害事件
            - heal: 治療事件
            - self_mutilation: 自傷事件
            - status_apply: 異常狀態
            - status_tick: 異常狀態持續
            - status_remove: 異常狀態移除
            - status_set 異常狀態設置stack
            - status_duration_update: 異常狀態持續時間更新
            - status_apply_fail: 異常狀態施加失敗
            - status_stack_update: 異常狀態堆疊更新
            - skip_turn: 跳過行動
            - skill: 技能事件
            - text : 純文字事件
            - turn_start: 回合開始
            - turn_end: 回合結束
            - refresh_status : 刷新雙方血量條/ 異常狀態條
            - other: 其他事件
        - appendix: DICT, 附加資訊
        - user: str, 動畫使用者(left/right) 
            此項只在 damage/heal/skill中被使用，只有單目標有動畫則優先使用此項
        - target: str, 動畫目標(left/right)
            此項為動畫中的目標
        - animation: str, 動畫名稱
            如果沒有指定的話，則從type中自動選擇

        - text: str, 純文字事件的文字(這個會自動生成，但當指定時)，選擇"text" 時需要自己填寫"
        """
        if not self.show_battle_log:
            return
        
        if not event:
            raise ValueError("event is None")
        
        # 如果在 player team 中發現user => user = left
        # 如果在 enemy team 中發現user => user = right
        if user in self.player_team:
            event.user = "left"
            event.target = "right"
        else:
            event.user = "right"
            event.target = "left"
        
        
        match event.type:
            case "damage":
                # 文字格式為：【{user['profession'].name}】 對 【{target['profession'].name}】 造成 {event.appendix["amount"]} 點傷害
                if event.text is None:
                    event.text = f"【{user['profession'].name}】 對 【{target['profession'].name}】 造成 {event.appendix['amount']} 點傷害"
                # 動畫格式為：
                # 1. event.target 使用受傷動畫 (在前端js處理)
                # 2. 啟用傷害數字動畫，數字在event.applendix["amount"] (在前端js處理)
                self.battle_log.append(event)
                # 3. 啟用狀態刷新動畫
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "heal":
                # f"{user['profession'].name} 恢復了 {heal_amount} 點生命值
                if event.text is None:
                    event.text = f"{user['profession'].name} 恢復了 {event.appendix['amount']} 點生命值"
                # 動畫格式為：
                # 1. event.user 使用治療動畫 (在前端js處理)
                # 2. 啟用治療數字動畫，數字在event.applendix["amount"] (在前端js處理)
                # chane right to left
                # TODO 這邊不知道為啥邏輯相反了 但是先這樣處理
                t = event.user
                event.user = event.target
                event.target = t
                
                
                self.battle_log.append(event)
                # 3. 啟用狀態刷新動畫
                self.add_event(event=BattleEvent(type="refresh_status"))
            
            case "self_mutilation":
                # 文字格式為：【{user['profession'].name}】 自傷 {event.appendix["amount"]} 點生命值
                if event.text is None:
                    event.text = f"【{user['profession'].name}】 自傷 {event.appendix['amount']} 點生命值"
                # 動畫格式為：
                # 1. event.user 使用受傷動畫 (在前端js處理)
                # 2. 啟用傷害數字動畫，數字在event.applendix["amount"] (在前端js處理)
                self.battle_log.append(event)
                # 3. 啟用狀態刷新動畫
                
                self.add_event(event=BattleEvent(type="refresh_status"))

            case "status_apply":
                # 文字格式為：user 被施加了 {event.appendix["effect_name"]} 狀態
                if event.text is None:
                    event.text = f"{user['profession'].name} 被施加了 {event.appendix['effect_name']} 狀態"
                # 動畫格式為： 無
                self.battle_log.append(event)
                
                self.add_event(event=BattleEvent(type="refresh_status"))
                
                  
            case "status_tick":
                # 額外處理異常狀態持續事件
                # 需要status type
                # 公用部分：user 的 effect_name 持續中
                # 如果是Dot類型，則顯示持續傷害
                if event.appendix["effect_type"] == "dot":
                    # 文字格式為 user 的 {event.appendix["effect_name"]} 持續中，造成 {event.appendix["amount"]} 點傷害
                    if event.text is None:
                        event.text = f"{user['profession'].name} 的 {event.appendix['effect_name']} 持續中，造成 {event.appendix['amount']} 點傷害"
                    # 動畫格式為： event.target 使用受傷動畫 (在前端js處理)
                    self.battle_log.append(event)
                else:
                    # 文字格式為 user 的 {event.appendix["effect_name"]} 持續中
                    if event.text is None:
                        event.text = f"{user['profession'].name} 的 {event.appendix['effect_name']} 持續中"
                    # 動畫格式為：無
                    self.battle_log.append(event)
                    
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "status_apply_fail":
                # 處理異常狀態施加失敗事件
                # 文字格式為 user 被施加 {event.appendix["effect_name"]} 狀態失敗
                if event.text is None:
                    event.text = f"{user['profession'].name} 被施加 {event.appendix['effect_name']} 狀態失敗"
                self.battle_log.append(event)
                # 無動畫
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "status_remove":
                # 處理異常狀態移除事件
                # 文字格式為 user 的 {event.appendix["effect_name"]} 狀態被移除
                if event.text is None:
                    event.text = f"{user['profession'].name} 的 {event.appendix['effect_name']} 狀態被移除"
                self.battle_log.append(event)
                # 無動畫
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "status_duration_update":
                # 處理異常狀態持續時間更新事件
                # 文字格式為 user 的 {event.appendix["effect_name"]} 狀態剩餘回合數更新為 {event.appendix["duration"]}
                if event.text is None:
                    event.text = f"{user['profession'].name} 的 {event.appendix['effect_name']} 狀態剩餘回合數更新為 {event.appendix['duration']}"
                    
                self.battle_log.append(event)
                
                self.add_event(event=BattleEvent(type="refresh_status"))
            case "status_stack_update":
                # 處理異常狀態堆疊更新事件
                # 文字格式為 user 的 {event.appendix["effect_name"]} 效果堆疊數量更新為 {event.appendix["stack"]}
                if event.text is None:
                    event.text = f"{user['profession'].name} 的 {event.appendix['effect_name']} 效果堆疊數量更新為 {event.appendix['stacks']}"
                #  動畫格式為： 不顯示動畫
                self.battle_log.append(event) 
                
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "status_set":
                
                # 處理異常狀態堆疊設置事件
                self.battle_log.append(event)
                
                self.add_event(event=BattleEvent(type="refresh_status"))
                
                
            case "skip_turn":
                # 文字格式為： 【{user['profession'].name}】，因 {event.applendix["effect_name"]} 跳過行動
                if event.text is None:
                    event.text = f"【{user['profession'].name}】，因 {event.appendix['effect_name']} 跳過行動" 
                # 動畫格式為： 不顯示動畫
                self.battle_log.append(event)

            case "skill":
                # 文字格式為： f"{self.name} 使用了技能 {sm.get_skill_name(skill_id)}。"
                if event.text is None:
                    event.text = f"{user['profession'].name} 使用了技能 {self.skill_mgr.get_skill_name(event.appendix['skill_id'])}。"
                # 動畫格式為： event.user 使用技能動畫 (在前端js處理)
                self.battle_log.append(event)
            
            case "text":
                # 文字格式為： event.text
                # 動畫格式： 不顯示動畫
                if not event.text:
                    raise ValueError("text is None")
                self.battle_log.append(event)
                
            case "refresh_status":
                # 文字格式為： 不顯示文字
                # 動畫格式為： 刷新雙方血量條/ 異常狀態條
                # 將以下dict加入 event.appendix 以刷新前端的數字
                # 註：left = player, right = enemy
                
                # 構建 appendix 字典 
                # copy cooldowns
                pla = self.player_team[0]["cooldowns"].copy()
                ene = self.enemy_team[0]["cooldowns"].copy()
 
                
                appendix = {
                    "global": {
                        "round": self.round_count , "max_rounds": self.max_rounds,"left_profession": self.player_team[0]["profession"].name, "right_profession": self.enemy_team[0]["profession"].name,"damage_coefficient": self.damage_coefficient
                    },
                    "left": {
                        "hp": self.player_team[0]["hp"],
                        "max_hp": self.player_team[0]["max_hp"],
                        "effects": self.player_team[0]["effect_manager"].get_effect_vector(),
                        "multiplier": {
                            "damage": self.player_team[0]["damage_multiplier"],
                            "defend": self.player_team[0]["defend_multiplier"],
                            "heal": self.player_team[0]["heal_multiplier"]
                        },
                        "cooldowns": pla
                    },
                    "right": {
                        "hp": self.enemy_team[0]["hp"],
                        "max_hp": self.enemy_team[0]["max_hp"],
                        "effects": self.enemy_team[0]["effect_manager"].get_effect_vector(),
                        "multiplier": {
                            "damage": self.enemy_team[0]["damage_multiplier"],
                            "defend": self.enemy_team[0]["defend_multiplier"],
                            "heal": self.enemy_team[0]["heal_multiplier"]
                        },
                        "cooldowns": ene
                    }
                }
                
                # 設定 appendix
                event.appendix = appendix
                # 清除文字
                event.text = None
                # 動畫格式為： 刷新狀態動畫 (在前端js處理)
                self.battle_log.append(event)
            
            case "turn_start":
                # add refresh_status
                self.battle_log.append(event)
                self.add_event(event=BattleEvent(type="refresh_status"))
                
            case "turn_end":
                # add refresh_status
                self.battle_log.append(event)
                self.add_event(event=BattleEvent(type="refresh_status"))

            
            case "other":
                # 處理其他事件類型
                self.battle_log.append(event)


        
        
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
        
        # print("\n",actions,"\n")

        p_action = actions.get("player", 0)
        e_action = actions.get("enemy", 0)
        
        # last skill used by one-hot encoding
        m = self.player_team[0]
        e = self.enemy_team[0]
        m['last_skill_used'] = [0,0,0]
        e['last_skill_used'] = [0,0,0]
        m['last_skill_used'][p_action] = 1
        e['last_skill_used'][e_action] = 1

        player_actions = [p_action]
        enemy_actions = [e_action]
   

        # 回合開始時處理被動技能
        self._process_passives()
        
        # 回合初始化
        for m in self.player_team + self.enemy_team:
            m["last_damage_taken"] = 0
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
                    e = BattleEvent(type="skip_turn",appendix={"effect_name":"麻痺"})
                    self.add_event(user,event=e)
                    continue
                # 冰凍的跳過
                elif user["skip_turn"]:
                    e = BattleEvent(type="skip_turn",appendix={"effect_name":"凍結"})
                    self.add_event(user,event=e)

                    # remove freeze effect
                    user["skip_turn"] = False
                    user["effect_manager"].remove_all_effects('凍結')
                    continue

            skill_index = int(player_actions[i])
            # 根據職業ID計算實際技能ID
            profession_id = user["profession"].profession_id
            skill_id = 3 * profession_id + skill_index

            available_skills = user["profession"].get_available_skill_ids(user["cooldowns"])
            if skill_id not in available_skills:
                # 如果選擇的技能不可用，隨機選擇可用技能
                available_skill_indices = [sid - 3 * profession_id for sid in available_skills]
                if available_skill_indices:
                    skill_index = random.choice(available_skill_indices)
                    skill_id = 3 * profession_id + skill_index
                    skill_name = self.skill_mgr.get_skill_name(skill_id)
                    self.add_event(user,event=BattleEvent(type="text",text=f"P隊伍{i}({user['profession'].name}) 選擇的技能不可用，隨機選擇 {skill_name}"))
  
                else:
                    # 如果沒有可用技能，跳過行動
                    self.add_event(user,event=BattleEvent(type="text",text=f"P隊伍{i}({user['profession'].name}) 沒有可用技能，跳過行動"))
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
                    be = BattleEvent(type="skip_turn",appendix={"effect_name":"麻痺"})
                    self.add_event(e,event=be)

                    continue
                # 冰凍的跳過
                elif e["skip_turn"]:
                    be = BattleEvent(type="skip_turn",appendix={"effect_name":"凍結"})
                    self.add_event(e,event=be)
                    # remove freeze effect
                    e["skip_turn"] = False
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
                if available_skill_indices:
                    skill_index = random.choice(available_skill_indices)
                    skill_id = 3 * profession_id + skill_index
                    skill_name = self.skill_mgr.get_skill_name(skill_id)
                    self.add_event(e,event=BattleEvent(type="text",text=f"E隊伍{j}({e['profession'].name}) 選擇的技能不可用，隨機選擇 {skill_name}"))

                else:
                    # 如果沒有可用技能，跳過行動
                    self.add_event(e,event=BattleEvent(type="text",text=f"E隊伍{j}({e['profession'].name}) 沒有可用技能，跳過行動"))

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
        def effect_player():
            for i, skill_id in player_effect_skills:
                user = self.player_team[i]
                targets = self._select_targets(self.enemy_team)
                if targets:
                    user["profession"].apply_skill(skill_id, user, targets, self)
                else:
                    # 敵方全滅，結束遊戲
                    self.done = True

        # 2. 敵方效果技能
        def effect_enemy():
            for j, skill_id in enemy_effect_skills:
                e = self.enemy_team[j]
                targets = self._select_targets(self.player_team)
                if targets:
                    e["profession"].apply_skill(skill_id, e, targets, self)
                else:
                    # 我方全滅，結束遊戲
                    self.done = True

        # 3. 我方治療技能
        def heal_player():
            for i, skill_id in player_heal_skills:
                user = self.player_team[i]
                targets = self.enemy_team
                if targets:
                    user["profession"].apply_skill(skill_id, user, targets, self)
                else:
                    # 治療目標不存在，可能無需額外處理
                    pass

        # 4. 敵方治療技能（如有）
        def heal_enemy():
            for j, skill_id in enemy_heal_skills:
                e = self.enemy_team[j]
                targets = self.player_team
                if targets:
                    e["profession"].apply_skill(skill_id, e, targets, self)
                else:
                    # 治療目標不存在，可能無需額外處理
                    pass

        # 5. 我方傷害技能
        def damage_player():
            for i, skill_id in player_damage_skills:
                user = self.player_team[i]
                targets = self._select_targets(self.enemy_team)
                if targets:
                    user["profession"].apply_skill(skill_id, user, targets, self)
                else:
                    # 敵方全滅，結束遊戲
                    self.done = True

        # 6. 敵方傷害技能
        def damage_enemy():
            for j, skill_id in enemy_damage_skills:
                e = self.enemy_team[j]
                targets = self._select_targets(self.player_team)
                if targets:
                    e["profession"].apply_skill(skill_id, e, targets, self)
                else:
                    # 我方全滅，結束遊戲
                    self.done = True

        # 抽籤處理順序
        # 兩兩一組排序，但依然按照效果、治療、傷害的順序
        order = [[effect_player, effect_enemy], [heal_player, heal_enemy], [damage_player, damage_enemy]]
        for pair in order:
            # 組內隨機排序
            random.shuffle(pair)
            pair[0]()
            pair[1]()
            
        
        # check 我方是否全滅
        def check_end():
            if all(m["hp"] <= 0 for m in self.player_team):
                self.done = True
                self.add_event(event=BattleEvent(type="text",text="我方全滅，敵方獲勝！"))
            # check 敵方是否全滅
            if all(e["hp"] <= 0 for e in self.enemy_team):
                self.done = True
                self.add_event(event=BattleEvent(type="text",text="敵方全滅，我方獲勝！"))
            # check 雙方全滅?
            if self._check_both_defeated():
                self.done = True
                self.add_event(event=BattleEvent(type="text",text="雙方全滅，平手！"))

        
        check_end()
        
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
                self.add_event(event=BattleEvent(type="text",text=f"戰鬥超過10回合，雙方的傷害係數增加了10%，現在為 {self.damage_coefficient:.2f}。"))
        # 強制停止
        self.round_count += 1
        if self.round_count > self.max_rounds:
                self.done = True
        
        # 因為異常狀態有可能造成擊殺，所以要在最後檢查
        if not self.done:
            check_end()
        
        self._print_round_footer()   
            
        #  返回觀測、獎勵、終止條件、截斷條件、信息
        rewards = {
        "player": self._get_reward(player=True),
        "enemy": self._get_reward(player=False),
        }
        
        terminateds = {
        "player": self.done,
        "enemy": self.done,
        "__all__": self.done,
        }
        
        truncateds = {
            "player": False,  # 如果有截斷條件，根據情況設置
            "enemy": False,
            "__all__": False,
        }

        # result = 4 type
        # notdone, playerwin, enemywin, draw
        info_res  = 2
        if self.done:
            if all(m["hp"] <= 0 for m in self.player_team):
                info_res = -1
            elif all(e["hp"] <= 0 for e in self.enemy_team):
                info_res = 1
            else:
                info_res = 0

        infos = {
        "player": {},  # 玩家相關的空信息
        "enemy": {},   # 敵人相關的空信息
        "__common__": {"result": info_res},  # 全局信息
        }
    
        obs_next = {
            "player":self._get_obs("player"),
            "enemy":self. _get_obs("enemy"),
        }
        return obs_next, rewards, terminateds, truncateds, infos


    def _get_obs(self, agent):
        """
        回傳同時給 AI1、AI2 的觀測。
        同時要加上 action_mask，表示哪些動作可用(沒進CD) => 1/0
        """
        player_m = self.player_team[0]
        enemy_m = self.enemy_team[0]

        # 定義職業數量 (假設職業 ID 的範圍是 0 ~ 12)
        num_professions = 13

        # One-Hot 編碼職業 ID 的輔助函式
        def one_hot_encode(profession_id, num_classes):
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[profession_id] = 1.0
            return one_hot

        # 根據該職業計算真實技能 ID，並編碼技能類型
        def one_hot_encode_skill(profession):
            # 利用 self.skill_mgr (技能管理器) 取得技能類型
            sm = self.skill_mgr
            # 定義 mapping: damage->[1,0,0], effect->[0,1,0], heal->[0,0,1]
            mapping = {
                "damage": np.array([1, 0, 0], dtype=np.float32),
                "effect": np.array([0, 1, 0], dtype=np.float32),
                "heal":   np.array([0, 0, 1], dtype=np.float32)
            }
            one_hot_list = []
            # 這裡 local 技能 id 為 0,1,2
            for local_id in [0, 1, 2]:
                real_skill_id = profession.profession_id * 3 + local_id
                skill_type = sm.get_skill_type(real_skill_id)
                if skill_type in mapping:
                    one_hot_list.append(mapping[skill_type])
                else:
                    one_hot_list.append(np.zeros(3, dtype=np.float32))
            # 串接三個 one-hot 編碼，總長度 9
            return np.concatenate(one_hot_list)

        # 編碼公共觀測中的 last_skill_used
        # 輸入的值應為 0, 1, 2，分別對應編碼：[0,0], [1,0], [0,1]
        def encode_last_skill(last_skill_local):
            if last_skill_local == 0:
                return np.array([0, 0], dtype=np.float32)
            elif last_skill_local == 1:
                return np.array([1, 0], dtype=np.float32)
            elif last_skill_local == 2:
                return np.array([0, 1], dtype=np.float32)
            else:
                # 若不在 0,1,2 範圍內則回傳預設 [0,0]
                return np.array([0, 0], dtype=np.float32)

        # 編碼 Player 與 Enemy 的職業
        player_profession_one_hot = one_hot_encode(player_m["profession"].profession_id, num_professions)
        enemy_profession_one_hot = one_hot_encode(enemy_m["profession"].profession_id, num_professions)

        # 取得各角色的技能類型 one-hot 編碼 (長度 9)
        player_skill_one_hot = one_hot_encode_skill(player_m["profession"])
        enemy_skill_one_hot = one_hot_encode_skill(enemy_m["profession"])

        # 構建個體觀測值
        # 原本數值特徵有 8 項，加上職業 one-hot (13 維) 及技能編碼 (9 維) → 13+8+9 = 30 維
        player_obs = np.concatenate([
            player_profession_one_hot,  # 13 維
            [
                player_m["hp"] / player_m["max_hp"],
                player_m["max_hp"] / self.maxProfession_hp,
                player_m["damage_multiplier"],
                player_m["defend_multiplier"],
                player_m["heal_multiplier"],
                np.log(player_m["accumulated_damage"] + 1),
                player_m["profession"].baseAtk,
                player_m["profession"].baseDef,
            ],  # 8 維
            player_skill_one_hot  # 9 維
        ], dtype=np.float32)

        enemy_obs = np.concatenate([
            enemy_profession_one_hot,  # 13 維
            [
                enemy_m["hp"] / enemy_m["max_hp"],
                enemy_m["max_hp"] / self.maxProfession_hp,
                enemy_m["damage_multiplier"],
                enemy_m["defend_multiplier"],
                enemy_m["heal_multiplier"],
                np.log(enemy_m["accumulated_damage"] + 1),
                enemy_m["profession"].baseAtk,
                enemy_m["profession"].baseDef,
            ],  # 8 維
            enemy_skill_one_hot  # 9 維
        ], dtype=np.float32)

        # 行動遮罩 (假設共有 3 個特徵)
        player_mask = self._get_action_mask(player_m)
        enemy_mask = self._get_action_mask(enemy_m)

        # 合併個體觀測與行動遮罩
        # 原本個體觀測由 21 維升級到 30 維，加上 3 維遮罩 → 33 維
        flattened_pobs = np.concatenate([player_mask, player_obs])
        flattened_eobs = np.concatenate([enemy_mask, enemy_obs])

        # 公共觀測值：保留 damage_coefficient 與 round_count，並額外加入對方的 last_skill_used 編碼
        base_public_obs = np.array(
            [self.damage_coefficient, self.round_count / self.max_rounds],
            dtype=np.float32
        )
        if agent == "player":
            # 對方是 enemy，假設 enemy_m["last_skill_used"] 為 local 技能 id (0,1,2)
            last_skill_encoded = encode_last_skill(enemy_m["last_skill_used"])
        else:
            last_skill_encoded = encode_last_skill(player_m["last_skill_used"])
        public_obs = np.concatenate([base_public_obs, last_skill_encoded])

        # 特效管理器觀測值 (假設各有 42 維)
        pem = player_m["effect_manager"].export_obs()
        eem = enemy_m["effect_manager"].export_obs()

        # 最終觀測值拼接：
        # 例如：若 agent 為 "player"，順序為：
        # player_mask + player_obs (33 維) + pem (42 維) +
        # enemy_mask + enemy_obs (33 維) + eem (42 維) + public_obs (2+2=4 維)
        pout = np.concatenate([flattened_pobs, pem, flattened_eobs, eem, public_obs])
        eout = np.concatenate([flattened_eobs, eem, flattened_pobs, pem, public_obs])
        

        if agent == "player":
            return pout
        else:
            return eout



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
                self.add_event(event=BattleEvent(type="text",text=f"{target['profession'].name} 被凍結，將跳過下一回合的行動。"))
                target["skip_turn"] = True

        # 檢查目標是否免疫傷害
        if target["effect_manager"].has_effect('免疫傷害'):
            self.add_event(event=BattleEvent(type="text",text=f"{target['profession'].name} 免疫所有傷害!"))
            return

        if can_be_blocked and target["is_defending"]:
            self.add_event(event=BattleEvent(type="text",text=f"{target['profession'].name} 正在防禦，擋下了攻擊!"))
            return

        # 應用傷害增減比例
        dmg = int(dmg * user.get("damage_multiplier", 1.0))

        # 處理防禦增減
        dmg = int(dmg / target.get("defend_multiplier", 1.0))
        dmg = int(dmg/target['profession'].baseDef)
        
        dmg *= self.damage_coefficient  # 考慮戰鬥特性：傷害係數
        
        dmg = max(1, dmg)  # 至少造成1點傷害
        
        dmg = int(dmg)
        # 實際減血
        target["hp"] = max(0, target["hp"] - dmg)
        target['accumulated_damage'] += dmg
        self.add_event(user=user,target=target,event=BattleEvent(type="damage",appendix={"amount":dmg}))
        
        # process damage taken event
        target["profession"].damage_taken(target,user,self,dmg)
    
        
        # update last attacker and last damage

        target["last_attacker"] = user
        target["last_damage_taken"] = dmg
        

    def deal_healing(self, user, heal_amount, rate = 0,heal_damage=False,target=None,self_mutilation = False):
        """
        處理治療邏輯，包括超過最大生命的處理
        # heal_damage 與 self_mutilation 不能同時為True
        """
        if user["hp"] <= 0 :
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
            self.add_event(user = user,event=BattleEvent(type="heal",appendix={"amount":heal_amount}))
        # 接著如果 heal_damage 為 True，則對目標造成治療量的rate%傷害
        if heal_damage and target:
            dmg = int(extra_heal * rate)  
            if extra_heal > 0:
                self.add_event(event = BattleEvent(type="text",text=f"{user['profession'].name} 治療溢出造成傷害！"))
                self.deal_damage(user, target, dmg, can_be_blocked=False)
        # 最後如果 self_mutilation 為 True，則對自己造成治療量的傷害(但不會低於最低血量)
        if self_mutilation:
            dmg = int(heal_amount)
            user["hp"] = max(1, user["hp"] - dmg)

            self.add_event(user = user,event=BattleEvent(type="self_mutilation",appendix={"amount":dmg}))
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
        for member in self.player_team:

            left = member
        for member in self.enemy_team:

            right = member
        left["profession"].on_turn_start(left,right,self,-1)
        right["profession"].on_turn_start(right,left,self,-1)
        
    def _process_passives_end_of_turn(self,mSkill = None,eSkill = None):
        """
        # take all  on turn end skill
        """ 
        left_skill = None
        right_skill = None
        # 我方處理       
        for i, m in mSkill:   
            left_skill = m     

        for i, e in eSkill:
            right_skill = e
            
        left = self.player_team[0]
        right = self.enemy_team[0]
        if left_skill != None:
            left["profession"].on_turn_end(left,right,self,left_skill)
        if right_skill != None:
            right["profession"].on_turn_end(right,left,self,right_skill)

                 
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
        res = 0
        if self.done:
            if player:
                if all(m["hp"] <= 0 for m in self.player_team) and all(e["hp"] <= 0 for e in self.enemy_team):
                    return -0.3
                elif all(m["hp"] <= 0 for m in self.player_team):
                    return -1
                elif all(e["hp"] <= 0 for e in self.enemy_team):
                    return 1
            else:
                if all(m["hp"] <= 0 for m in self.player_team) and all(e["hp"] <= 0 for e in self.enemy_team):
                    return -0.3
                elif all(m["hp"] <= 0 for m in self.player_team):
                    return 1
                elif all(e["hp"] <= 0 for e in self.enemy_team):
                    return -1
        return res

    def _print_round_header(self):
        self.add_event(event=BattleEvent(type="turn_start"))
        if self.show_battle_log:
            print(f"\n---\n回合 {self.round_count} :\n")

    def _print_round_footer(self):
        if self.show_battle_log:
            for line in self.battle_log:
                pass
                # print(line.type, line.text)
            # self.battle_log = []
        self.add_event(event=BattleEvent(type="turn_end"))

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
