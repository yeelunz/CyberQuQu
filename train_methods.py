# train_epoch.py
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from battle_env import BattleEnv

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from collections import defaultdict
from train_var import player_train_times, enemy_train_times
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# import RL model

import torch
import torch.nn as nn
import torch.nn.functional as F



def multi_agent_cross_train(num_iterations, professions, skill_mgr, 
                            save_path_1="multiagent_ai1.zip",
                            save_path_2="multiagent_ai2.zip"):
    """
    多智能體交叉訓練的簡易示例:
    - AI1 與 AI2 同時在 BattleEnv 中對打
    - 每一個 iteration，反覆 roll out 多個 episodes，收集資料後分別更新 model1, model2
    - 訓練完後存檔
    """
    print("=== 開始多智能體交叉訓練 ===")

    # 初始化環境配置
    beconfig = make_env_config(skill_mgr, professions, train_mode=True)
    
    ray.init()
    config = (
            PPOConfig()
            .environment(
                env=BattleEnv,
                env_config=beconfig
            )
            .env_runners(
                num_env_runners=1,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                sample_timeout_s=120
            )
            .framework("torch")
            .resources(
            num_gpus=1,  # 分配 1 個 GPU
            num_cpus_per_worker=2,  # 每個 worker 分配 1 個 CPU
            num_gpus_per_worker=1  # Worker 是否使用 GPU
            )
        )
    benv = BattleEnv(beconfig)
    config = config.multi_agent(
        policies={
            "player_policy": (None, benv.observation_space, benv.action_space, {}),
            "enemy_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: 
            "player_policy" if agent_id == "player" else "enemy_policy"

    )
    # TODO 遷移到新API
    config.api_stack(
    enable_rl_module_and_learner=False,
    enable_env_runner_and_connector_v2=False
    )
    
    algo = config.build()
    print("=== 模型初始化完成 ===")
    # print("type(algo):", type(algo))
    # set train time to 0
    for i in range(num_iterations):
        result = algo.train()
        
        print("result:", result)
        print("-" * 60)
        
    # 存檔
    checkpoint_dir = algo.save("./my_battle_ppo_checkpoints")
    print("Checkpoint saved at", checkpoint_dir)
    ray.shutdown()



#------------------------------------------------------------
# 以下示範 (2)~(4) 選單對應的測試或ELO
#------------------------------------------------------------
def version_test_random_vs_random(professions,skill_mgr,  num_battles=100):
    """
    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率（不計入平局）
    """
    print("\n開始進行每個職業相互對戰100場的測試...")

    # 初始化結果字典
    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0} for op in professions if op != p} for p in professions}
    # print(results)
    total_combinations = len(professions) * (len(professions) - 1)
    current_combination = 0
 
    for p in professions:
        for op in professions:
            if p == op:
                continue
            
            current_combination += 1
            print(f"\n對戰 {current_combination}/{total_combinations}: {p.name} VS {op.name}")
            
            for battle_num in range(1, num_battles + 1):

                # 初始化 BattleEnv，關閉 battle_log 以加快速度
                env = BattleEnv(
                    make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                )
                # 編號決定先後攻
                done = False
                obs, _ = env.reset()

                while not done:
                    # 取第 0~2 格，獲取合法動作
                    pmask = obs["player"][0:3]
                    emask = obs["enemy"][0:3]
                    p_actions = np.where(pmask == 1)[0]
                    e_actions = np.where(emask == 1)[0]
                    p_act = random.choice(p_actions) if len(p_actions) > 0 else 0
                    e_act = random.choice(e_actions) if len(e_actions) > 0 else 0


                    obs, rew, done, tru, info = env.step({
                        "player": p_act,
                        "enemy": e_act
                    })
             

                    done = done["__all__"]

                rew = rew["player"]

                # 判斷結果
                if rew > 0:
                    results[p.name][op.name]['win'] += 1
                elif rew < 0:
                    results[p.name][op.name]['loss'] += 1
                else:
                    results[p.name][op.name]['draw'] += 1

                # 顯示進度
                if battle_num % 10 == 0:
                    print(f"  完成 {battle_num}/{num_battles} 場")
                    
                

    # 計算勝率
    win_rate_table = {}
    for p in professions:
        if p.name not in win_rate_table:
            win_rate_table[p.name] = {}
        for op in professions:
            if p == op:
                continue
            wins = results[p.name][op.name]['win']
            losses = results[p.name][op.name]['loss']
            draws = results[p.name][op.name]['draw']
            total = wins + losses  # 排除平局
            win_rate = (wins / total) * 100 if total > 0 else 0
            win_rate_table[p.name][op.name] = {
                'win': wins,
                'loss': losses,
                'draw': draws,
                'win_rate': win_rate
            }

    # 顯示結果
    print("\n=== 每個職業相互對戰100場的勝率 ===")
    for player_prof, stats in win_rate_table.items():
        opponents = list(stats.keys())
        print(f"\n職業 {player_prof}")
        print(" | ".join(opponents) + " |")

        # 準備 '勝' 行
        win_values = [str(stats[op]['win']) for op in opponents]
        total_wins = sum([stats[op]['win'] for op in opponents])
        print("勝 | " + " | ".join(win_values) + f" | {total_wins}")

        # 準備 '負' 行
        loss_values = [str(stats[op]['loss']) for op in opponents]
        total_losses = sum([stats[op]['loss'] for op in opponents])
        print("負 | " + " | ".join(loss_values) + f" | {total_losses}")

        # 準備 '平' 行
        draw_values = [str(stats[op]['draw']) for op in opponents]
        total_draws = sum([stats[op]['draw'] for op in opponents])
        print("平 | " + " | ".join(draw_values) + f" | {total_draws}")

        # 準備 '勝率' 行（不計入平局）
        win_rate_values = [f"{stats[op]['win_rate']:.2f}%" for op in opponents]
        # 計算總勝率：所有勝利場次除以所有勝利和失敗場次
        total_wins_all = sum([stats[op]['win'] for op in opponents])
        total_losses_all = sum([stats[op]['loss'] for op in opponents])
        total_win_rate = (total_wins_all / (total_wins_all + total_losses_all)) * 100 if (total_wins_all + total_losses_all) > 0 else 0
        print("勝率 | " + " | ".join(win_rate_values) + f" | {total_win_rate:.2f}%")

    input("對戰完成。按Enter返回主選單...")

    return

def high_level_test_ai_vs_ai(model_path_1, model_path_2, professions, skill_mgr, num_battles=100):
    """
    (3) 高段環境測試: 雙方都是 AI，交叉戰鬥100場
 

    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率（不計入平局）
    """
    print("\n開始進行每個職業相互對戰100場的測試...")
    ray.init()

    # 初始化結果字典
    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0} for op in professions if op != p} for p in professions}
    # print(results)
    total_combinations = len(professions) * (len(professions) - 1)
    current_combination = 0
 
    for p in professions:
        for op in professions:
            if p == op:
                continue
            
            current_combination += 1
            print(f"\n對戰 {current_combination}/{total_combinations}: {p.name} VS {op.name}")
            
            for battle_num in range(1, num_battles + 1):

                # 初始化 BattleEnv，關閉 battle_log 以加快速度
                env = BattleEnv(
                    make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                )
                # 編號決定先後攻
                done = False
                obs, _ = env.reset()

                while not done:
                    # 取第 0~2 格，獲取合法動作
                    pmask = obs["player"][0:3]
                    emask = obs["enemy"][0:3]
                    p_actions = np.where(pmask == 1)[0]
                    e_actions = np.where(emask == 1)[0]
                    p_act = random.choice(p_actions) if len(p_actions) > 0 else 0
                    e_act = random.choice(e_actions) if len(e_actions) > 0 else 0


                    obs, rew, done, tru, info = env.step({
                        "player": p_act,
                        "enemy": e_act
                    })
             

                    done = done["__all__"]

                rew = rew["player"]

                # 判斷結果
                if rew > 0:
                    results[p.name][op.name]['win'] += 1
                elif rew < 0:
                    results[p.name][op.name]['loss'] += 1
                else:
                    results[p.name][op.name]['draw'] += 1

                # 顯示進度
                if battle_num % 10 == 0:
                    print(f"  完成 {battle_num}/{num_battles} 場")
                    
                

    # 計算勝率
    win_rate_table = {}
    for p in professions:
        if p.name not in win_rate_table:
            win_rate_table[p.name] = {}
        for op in professions:
            if p == op:
                continue
            wins = results[p.name][op.name]['win']
            losses = results[p.name][op.name]['loss']
            draws = results[p.name][op.name]['draw']
            total = wins + losses  # 排除平局
            win_rate = (wins / total) * 100 if total > 0 else 0
            win_rate_table[p.name][op.name] = {
                'win': wins,
                'loss': losses,
                'draw': draws,
                'win_rate': win_rate
            }

    # 顯示結果
    print("\n=== 每個職業相互對戰100場的勝率 ===")
    for player_prof, stats in win_rate_table.items():
        opponents = list(stats.keys())
        print(f"\n職業 {player_prof}")
        print(" | ".join(opponents) + " |")

        # 準備 '勝' 行
        win_values = [str(stats[op]['win']) for op in opponents]
        total_wins = sum([stats[op]['win'] for op in opponents])
        print("勝 | " + " | ".join(win_values) + f" | {total_wins}")

        # 準備 '負' 行
        loss_values = [str(stats[op]['loss']) for op in opponents]
        total_losses = sum([stats[op]['loss'] for op in opponents])
        print("負 | " + " | ".join(loss_values) + f" | {total_losses}")

        # 準備 '平' 行
        draw_values = [str(stats[op]['draw']) for op in opponents]
        total_draws = sum([stats[op]['draw'] for op in opponents])
        print("平 | " + " | ".join(draw_values) + f" | {total_draws}")

        # 準備 '勝率' 行（不計入平局）
        win_rate_values = [f"{stats[op]['win_rate']:.2f}%" for op in opponents]
        # 計算總勝率：所有勝利場次除以所有勝利和失敗場次
        total_wins_all = sum([stats[op]['win'] for op in opponents])
        total_losses_all = sum([stats[op]['loss'] for op in opponents])
        total_win_rate = (total_wins_all / (total_wins_all + total_losses_all)) * 100 if (total_wins_all + total_losses_all) > 0 else 0
        print("勝率 | " + " | ".join(win_rate_values) + f" | {total_win_rate:.2f}%")

    input("對戰完成。按Enter返回主選單...")

    return

def compute_ai_elo(model_path_1, professions, skill_mgr, base_elo=1000, num_battles=100):
    """
    (4) AI ELO: 與隨機電腦比較, 基準分 1000, 
    簡易 ELO 計算: ELO_new = ELO_old + K*(score - expected)
    這裡只示範一個簡易算式
    """
    try:
        model1 = PPO.load(model_path_1)
    except:
        print(f"模型 {model_path_1} 載入失敗。")
        input("按Enter返回主選單...")
        return

    ELO = base_elo
    K = 32
    # 簡化：隨機電腦 ELO = 1000
    opp_ELO = 1500

    print("=== AI ELO測試 vs Random ===")
    for i in range(num_battles):
        p_team = [{
            "profession": random.choice(professions),
            "hp": 0,"max_hp": 0,
            "status": {}, "skip_turn":False,
            "is_defending":False,
            "damage_multiplier":1.0,
            "defend_multiplier":1.0,
            "heal_multiplier":1.0,
            "battle_log":[],
            "cooldowns": {}
        }]
        e_team = [{
            "profession": random.choice(professions),
            "hp": 0,"max_hp": 0,
            "status": {}, "skip_turn":False,
            "is_defending":False,
            "damage_multiplier":1.0,
            "defend_multiplier":1.0,
            "heal_multiplier":1.0,
            "battle_log":[],
            "cooldowns": {}
        }]
        env = BattleEnv(1,1,30,p_team,e_team,skill_mgr,False)
        obs = env.reset()
        done = False
        while not done:
            p_obs = obs["player_obs"]
            p_mask = obs["player_action_mask"]
            p_act, _ = model1.predict(p_obs, deterministic=False)
            if p_mask[p_act]==0:
                valid_as = np.where(p_mask==1)[0]
                p_act = random.choice(valid_as) if len(valid_as)>0 else 0

            e_mask = obs["opponent_action_mask"]
            e_avail = np.where(e_mask==1)[0]
            e_act = random.choice(e_avail) if len(e_avail)>0 else 0

            obs, rewards, done, info = env.step({
                "player_action":p_act,
                "opponent_action": e_act
            })
        final_r = rewards["player_reward"]
        if final_r>0:
            score = 1  # win
        elif final_r<0:
            score = 0  # lose
        else:
            score = 0.5

        # 預期勝率
        expected = 1 / (1 + 10**((opp_ELO - ELO)/400))
        ELO = ELO + K*(score - expected)

    print(f"AI vs Random, 對戰 {num_battles} 場後, AI ELO = {ELO:.2f}")
    input("按Enter返回主選單...")


def make_env_config(skill_mgr,professions,show_battlelog = False,pr1 = None,pr2 = None,train_mode = False):
    if pr1 is None:
        pr1 = random.choice(professions)
    if pr2 is None:
        pr2 = random.choice(professions)
    config = {
    "team_size": 1,
    "enemy_team_size": 1,
    "max_rounds": 30,
    "player_team": [{
        "profession": pr1,
        "hp": 0,
        "max_hp": 0,
        "status": {},
        "skip_turn": False,
        "is_defending": False,
        "damage_multiplier": 1.0,
        "defend_multiplier": 1.0,
        "heal_multiplier": 1.0,
        "battle_log": [],
        "cooldowns": {}
    }],
    "enemy_team": [{
        "profession": pr2,
        "hp": 0,
        "max_hp": 0,
        "status": {},
        "skip_turn": False,
        "is_defending": False,
        "damage_multiplier": 1.0,
        "defend_multiplier": 1.0,
        "heal_multiplier": 1.0,
        "battle_log": [],
        "cooldowns": {}
    }],
    "skill_mgr": skill_mgr,
    "show_battle_log": show_battlelog,
    "round_count": 1,
    "done": False,
    "battle_log": [],
    "damage_coefficient": 1.0,
    "train_mode": train_mode,
    "all_professions": professions
    }
    return config