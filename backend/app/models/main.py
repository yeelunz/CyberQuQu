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
    high_level_test_ai_vs_ai,       # (3) 高段環境測試
    compute_ai_elo,                 # (4) AI ELO
)

#---------------------------------------
# 建立 SkillManager & Professions
#---------------------------------------
def build_skill_manager():
    return sm

def build_professions():
    return [
        Paladin(),
        Mage(),
        Assassin(),
        Archer(),
        Berserker(),
        DragonGod(),
        BloodGod(),
        SteadfastWarrior(),
        Devour(),
        Ranger(),
        ElementalMage(),
        HuangShen(),
        GodOfStar()
    ]

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
def computer_vs_computer(skill_mgr, professions):
    env = BattleEnv(
        config=make_env_config(skill_mgr=skill_mgr, professions=professions,show_battlelog=True),
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
    input("按Enter返回主選單...")

#---------------------------------------
# (6) AI VS 電腦
#---------------------------------------
def ai_vs_computer(model_path, skill_mgr, professions):
    try:
        model = PPO.load(model_path)
    except:
        print(f"模型 {model_path} 載入失敗，請先進行訓練。")
        input("按Enter返回主選單...")
        return

    p_team = [{
        "profession": random.choice(professions),
        "hp": 0,
        "max_hp": 0,
        "status": {},
        "skip_turn": False,
        "is_defending": False,
        "times_healed": 0,
        "next_heal_double": False,
        "damage_multiplier": 1.0,
        "defend_multiplier": 1.0,
        "heal_multiplier": 1.0,
        "battle_log": [],
        "cooldowns": {}
    }]
    e_team = [{
        "profession": random.choice(professions),
        "hp": 0,
        "max_hp": 0,
        "status": {},
        "skip_turn": False,
        "is_defending": False,
        "times_healed": 0,
        "next_heal_double": False,
        "damage_multiplier": 1.0,
        "defend_multiplier": 1.0,
        "heal_multiplier": 1.0,
        "battle_log": [],
        "cooldowns": {}
    }]

    env = BattleEnv(
        team_size=1,
        enemy_team_size=1,
        max_rounds=30,
        player_team=p_team,
        enemy_team=e_team,
        skill_mgr=skill_mgr,
        show_battle_log=True
    )

    obs = env.reset()
    done = False
    while not done:
        player_obs = obs["player_obs"]
        p_mask = obs["player_action_mask"]
        avail_actions = np.where(p_mask == 1)[0]
        if len(avail_actions) == 0:
            action_player = 0
        else:
            pred, _ = model.predict(player_obs, deterministic=False)
            if pred not in avail_actions:
                action_player = random.choice(avail_actions)
            else:
                action_player = pred

        e_mask = obs["opponent_action_mask"]
        e_actions = np.where(e_mask == 1)[0]
        if len(e_actions) == 0:
            action_enemy = 0
        else:
            action_enemy = random.choice(e_actions)

        actions = {
            "player_action": action_player,
            "opponent_action": action_enemy
        }

        obs, rewards, done_, info = env.step(actions)
        done = done_

    final_rew = rewards["player_reward"]
    if final_rew > 0:
        print("AI(左)贏了!")
    elif final_rew < 0:
        print("電腦(右)贏了!")
    else:
        print("平手或回合耗盡")
    input("按Enter返回主選單...")

#---------------------------------------
# (7) AI VS AI
#---------------------------------------
def ai_vs_ai(model_path_1, model_path_2, skill_mgr, professions):
    print("AI1 職業選擇:")
    for i, p in enumerate(professions):
        print(f"{i}) {p.name}")
    try:
        p1_idx = int(input("請輸入AI1職業(0-12): "))
        if p1_idx < 0 or p1_idx > 12:
            p1_idx = 0
    except:
        p1_idx = 0

    print("AI2 職業選擇:")
    for i, p in enumerate(professions):
        print(f"{i}) {p.name}")
    try:
        p2_idx = int(input("請輸入AI2職業(0-12): "))
        if p2_idx < 0 or p2_idx > 12:
            p2_idx = 0
    except:
        p2_idx = 0

    p1 = professions[p1_idx]
    p2 = professions[p2_idx]

    beconfig = make_env_config(skill_mgr=skill_mgr, professions=professions, pr1=p1, pr2=p2, show_battlelog=True)
    benv = BattleEnv(config=beconfig)

    config = (
        PPOConfig()
        .environment(env=BattleEnv, env_config=beconfig)
        .env_runners(num_env_runners=1, sample_timeout_s=120)
        .framework("torch")
        .training(model={"custom_model": "my_mask_model"})
    )
    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    config = config.multi_agent(
        policies={"shared_policy": (None, benv.observation_space, benv.action_space, {})},
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: "shared_policy"
    )

    check_point_path = os.path.abspath("my_battle_ppo_checkpoints")
    trainer = config.build()
    trainer.restore(check_point_path)

    done = False
    obs, _ = benv.reset()
    while not done:
        pmask = obs["player"][0:3]
        emask = obs["enemy"][0:3]
        p_actions = np.where(pmask == 1)[0]
        e_actions = np.where(emask == 1)[0]
        
        p_act = trainer.compute_single_action(obs['player'], policy_id="shared_policy")
        e_act = trainer.compute_single_action(obs['enemy'], policy_id="shared_policy")

        actions = {
            "player": p_act,
            "enemy": e_act
        }
        obs, rew, done_, tru, info = benv.step(actions)
        done = done_["__all__"]

    info = info["__common__"]["result"]
    if info > 0:
        print("AI1(左)贏了!")
    elif info < 0:
        print("AI2(右)贏了!")
    else :
        print("平手或回合耗盡")

    input("按Enter返回主選單...")

def main():
    ray.init()
    skill_mgr = build_skill_manager()
    professions = build_professions()

    default_model_path_1 = "multiagent_ai1.zip"
    default_model_path_2 = "multiagent_ai2.zip"

    while True:
        print("\n=== 多智能體戰鬥系統 ===")
        print("1) 交叉疊代訓練 (多智能體)")
        print("2) 版本環境測試 (電腦隨機 vs 電腦隨機)")
        print("3) 高段環境測試 (AI vs AI, 交叉戰鬥100場)")
        print("4) AI ELO (AI vs 隨機電腦)")
        print("5) 電腦 VS 電腦")
        print("6) AI VS 電腦")
        print("7) AI VS AI")
        print("8) 各職業介紹")
        print("0) 離開")
        c = input("請輸入選項: ").strip()
        if c == "0":
            print("再見!")
            sys.exit(0)
        elif c == "1":
            try:
                iteration = int(input("請輸入要訓練的疊代次數(例如 5): "))
            except:
                iteration = 5
            multi_agent_cross_train(
                num_iterations=iteration,
                professions=professions,
                skill_mgr=skill_mgr,
                save_path_1=default_model_path_1,
                save_path_2=default_model_path_2
            )
        elif c == "2":
            version_test_random_vs_random(professions, skill_mgr, num_battles=50)
        elif c == "3":
            high_level_test_ai_vs_ai(
                model_path_1=default_model_path_1,
                model_path_2=default_model_path_2,
                professions=professions,
                skill_mgr=skill_mgr,
                num_battles=25
            )
        elif c == "4":
            compute_ai_elo(
                model_path_1=default_model_path_1,
                professions=professions,
                skill_mgr=skill_mgr,
                base_elo=1500,
                opponent_elo=1500,
                num_battles=200
            )
        elif c == "5":
            computer_vs_computer(skill_mgr, professions)
        elif c == "6":
            ai_vs_computer(default_model_path_1, skill_mgr, professions)
        elif c == "7":
            ai_vs_ai(default_model_path_1, default_model_path_2, skill_mgr, professions)
        elif c == "8":
            show_profession_info(professions, skill_mgr)
        else:
            print("無效選項，請重新輸入。")



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



if __name__ == "__main__":
    main()