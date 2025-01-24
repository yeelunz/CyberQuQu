# backend/main.py
import sys 
import numpy as np
import random
import pandas as pd
from stable_baselines3 import PPO

from .battle_env import BattleEnv
from .skills import SkillManager, sm
from .professions import (
    Paladin, Mage, Assassin, Archer, Berserker, DragonGod, BloodGod,
    SteadfastWarrior, Devour, Ranger, ElementalMage, HuangShen,
    GodOfStar
)
from .status_effects import StatusEffect, DamageMultiplier
from .effect_mapping import EFFECT_MAPPING, EFFECT_VECTOR_LENGTH
from .train_methods import make_env_config
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo  # 或其他使用的算法
import os
from ray.rllib.utils.checkpoints import get_checkpoint_info
import ray

# 本檔案最重要的是「只留以下幾個功能選單」
# 1) 交叉疊代訓練 (多智能體)
# 2) 版本環境測試 (雙方隨機 => 交叉戰鬥100場，並給統計)
# 3) 高段環境測試 (雙方都是 AI => 交叉戰鬥100場，並給統計)
# 4) AI ELO (與隨機電腦比較，基準分設 1000)
# 5) 電腦 VS 電腦
# 6) AI VS 電腦
# 7) AI VS AI
# 8) 各職業介紹

from .train_methods import (
    multi_agent_cross_train,        # (1) 交叉疊代訓練
    version_test_random_vs_random,  # (2) 版本環境測試
    compute_ai_elo,                 # (4) AI ELO
)



#---------------------------------------
# (8) 各職業介紹
#---------------------------------------
def show_profession_info(profession_list, skill_mgr):
    print("\n========== 職業介紹 ==========")
    for p in profession_list:
        print(f"{p.name} (HP={p.base_hp})")
        print(f"攻擊係數{p.baseAtk} 防禦係數{p.baseDef}")
        print(f"  被動: {p.passive_desc}")
        skill_ids = p.get_available_skill_ids({0:0,1:0,2:0})
        for sid in skill_ids:
            desc = skill_mgr.get_skill_desc(sid)
            skill = skill_mgr.skills[sid]
            print(f"    技能 {skill.name} => {desc}")
        print("----------------------------------")
    input("按Enter返回主選單...")

#---------------------------------------
# (5) 電腦 VS 電腦
#---------------------------------------
def computer_vs_computer(skill_mgr, professions,pr1,pr2):
    # 這邊的 pr1, pr2 是職業物件
    env = BattleEnv(
        config=make_env_config(skill_mgr=skill_mgr, professions=professions,show_battlelog=True,pr1=pr1,pr2=pr2),
    )
    done = False
    obs, _ = env.reset()
    
    while not done:
        pmask = obs["player"][0:3]
        emask = obs["enemy"][0:3]
        p_actions = np.where(pmask == 1)[0]
        e_actions = np.where(emask == 1)[0]
        p_act = random.choice(p_actions) if len(p_actions) > 0 else 0
        e_act = random.choice(e_actions) if len(e_actions) > 0 else 0

        actions = {
            "player": p_act,
            "enemy": e_act
        }
        obs, rew, done_, tru, info = env.step(actions)
        
        done = done_["__all__"]

    res = info["__common__"]["result"]
    
    if res == 1:
        print("電腦(左)贏了!")
    elif res == -1:
        print("電腦(右)贏了!")
    else:
        print("平手或回合耗盡")
    

    return env.battle_log


current_loaded_model_path_1 = None
current_loaded_model_path_2 = None
current_trainer_1 = None
current_trainer_2 = None



def ai_vs_ai(skill_mgr, professions, model_path_1, model_path_2, pr1, pr2,SameModel=False):
    global current_loaded_model_path_1, current_loaded_model_path_2
    global current_trainer_1, current_trainer_2
    
    env = BattleEnv(
        config=make_env_config(skill_mgr=skill_mgr, professions=professions,show_battlelog=True,pr1=pr1,pr2=pr2),
    )
    done = False
    obs, _ = env.reset()
    
    # 初始化 AI ELO
    beconfig = make_env_config(skill_mgr=skill_mgr, professions=professions,show_battlelog=True,pr1=pr1,pr2=pr2)
    config = (
        PPOConfig()
        .environment(
            env=BattleEnv,            # 指定我們剛剛定義的環境 class
            env_config=beconfig# 傳入給 env 的一些自定設定
        )
        .env_runners(num_env_runners=1,sample_timeout_s=120)  # 可根據你的硬體調整
        .framework("torch")            # 或 "tf"
        .training(
        model={
            "custom_model": "my_mask_model",
        }
    )
    )
    benv = BattleEnv(config=beconfig)
    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
    config = config.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
            },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: 
            "shared_policy" if agent_id == "player" else "shared_policy")
    
    need_reload_1 = (model_path_1 is not None and model_path_1 != current_loaded_model_path_1)
    need_reload_2 = (model_path_2 is not None and model_path_2 != current_loaded_model_path_2 and not SameModel)

    if SameModel:
        # 雙方用同一個 model
        if need_reload_1:
            # 需要重載
            current_trainer_1 = config.build()
            print("載入模型:", model_path_1)
            # 這裡可以顯示轉圈(前端) -> 你可以用 SSE 或直接讓前端等 fetch
            current_trainer_1.restore(model_path_1)
            current_loaded_model_path_1 = model_path_1
        trainer1 = current_trainer_1
        trainer2 = current_trainer_1
        current_loaded_model_path_2 = model_path_1  # 同一個
    else:
        # 左方
        if need_reload_1:
            current_trainer_1 = config.build()
            print("載入模型(左):", model_path_1)
            current_trainer_1.restore(model_path_1)
            current_loaded_model_path_1 = model_path_1

        # 右方
        if need_reload_2:
            current_trainer_2 = config.build()
            print("載入模型(右):", model_path_2)
            current_trainer_2.restore(model_path_2)
            current_loaded_model_path_2 = model_path_2

        trainer1 = current_trainer_1
        trainer2 = current_trainer_2
        
        
    
    while not done:
        
        if SameModel:
            p_act = trainer1.compute_single_action(obs['player'], policy_id="shared_policy")
            e_act = trainer1.compute_single_action(obs['enemy'], policy_id="shared_policy")
           
        else:
            
            p_act = trainer1.compute_single_action(obs['player'], policy_id="shared_policy")
            # if p act in mask is 0, then choose random action
            e_act = trainer2.compute_single_action(obs['enemy'] ,policy_id="shared_policy")

        actions = {
            "player": p_act,
            "enemy": e_act
        }
        obs, rew, done_, tru, info = env.step(actions)
        
        done = done_["__all__"]

    res = info["__common__"]["result"]
    
    if res == 1:
        print("電腦(左)贏了!")
    elif res == -1:
        print("電腦(右)贏了!")
    else:
        print("平手或回合耗盡")
    

    return env.battle_log
    




# backend/main.py
def get_professions_data(profession_list, skill_mgr):
    """
    將職業資訊轉換為結構化的資料格式，不包含 skill_id。
    """
    professions_data = []
    for p in profession_list:
        profession = {
            "name": p.name,
            "hp": p.base_hp,
            "attack_coeff": p.baseAtk,
            "defense_coeff": p.baseDef,
            "passive": {
                "name": p.passive_name,
                "description": p.passive_desc
            },
            "skills": []
        }
        skill_ids = p.get_available_skill_ids({0: 0, 1: 0, 2: 0})
        for sid in skill_ids:
            skill = skill_mgr.skills.get(sid)
            if skill:
                skill_info = {
                    "name": skill.name,
                    "description": skill.desc,
                    "cooldown": skill.cool_down if skill.cool_down > 0 else None,
                    "type": skill.type
                }
                profession["skills"].append(skill_info)
        professions_data.append(profession)
    return professions_data




