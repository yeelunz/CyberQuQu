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
    enable_env_runner_and_connector_v2=False)
    
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

                info = info["__common__"]["result"]

                # 判斷結果
                if info == 1:
                    results[p.name][op.name]['win'] += 1
                elif info == -1:
                    results[p.name][op.name]['loss'] += 1
                else:
                    results[p.name][op.name]['draw'] += 1

                # 顯示進度
                if battle_num % 10 == 0:
                    print(f"  完成 {battle_num}/{num_battles} 場")
                    
                

        win_rate_table = {} 
        enemy_win_rate_table = {}  # 新增enemy的表格

        for p in professions: 
            if p.name not in win_rate_table: 
                win_rate_table[p.name] = {} 
                enemy_win_rate_table[p.name] = {}  # 初始化enemy的數據
            for op in professions: 
                if p == op: 
                    continue 
                # Player方數據
                wins = results[p.name][op.name]['win'] 
                losses = results[p.name][op.name]['loss'] 
                draws = results[p.name][op.name]['draw'] 
                total = wins + losses  
                win_rate = (wins / total) * 100 if total > 0 else 0 
                
                # Enemy方數據 (對調勝負)
                enemy_wins = losses  # enemy的勝 = player的負
                enemy_losses = wins  # enemy的負 = player的勝
                enemy_win_rate = (enemy_wins / total) * 100 if total > 0 else 0
                
                win_rate_table[p.name][op.name] = { 
                    'win': wins, 
                    'loss': losses, 
                    'draw': draws, 
                    'win_rate': win_rate 
                }
                enemy_win_rate_table[p.name][op.name] = {
                    'win': enemy_wins,
                    'loss': enemy_losses,
                    'draw': draws,
                    'win_rate': enemy_win_rate
                }

    # 顯示結果 
    print("\n=== 每個職業相互對戰100場的勝率（括號內為enemy方數據）===") 
    for player_prof, stats in win_rate_table.items(): 
        opponents = list(stats.keys()) 
        print(f"\n職業 {player_prof}") 
        print(" | ".join(opponents) + " |") 
        
        # 準備 '勝' 行 
        win_values = []
        for op in opponents:
            player_wins = stats[op]['win']
            # 先找到對手視角的數據，然後取其losses作為enemy的wins
            enemy_wins = win_rate_table[op][player_prof]['loss']  # 關鍵改變：使用loss而不是win
            win_values.append(f"{player_wins}({enemy_wins})")
        
        total_wins = sum([stats[op]['win'] for op in opponents]) 
        total_enemy_wins = sum([win_rate_table[op][player_prof]['loss'] for op in opponents])
        print("勝 | " + " | ".join(win_values) + f" | {total_wins}({total_enemy_wins})") 
        
        # 準備 '負' 行 
        loss_values = []
        for op in opponents:
            player_losses = stats[op]['loss']
            # 同樣，對手的wins將作為enemy的losses
            enemy_losses = win_rate_table[op][player_prof]['win']  # 關鍵改變：使用win而不是loss
            loss_values.append(f"{player_losses}({enemy_losses})")
        
        total_losses = sum([stats[op]['loss'] for op in opponents]) 
        total_enemy_losses = sum([win_rate_table[op][player_prof]['win'] for op in opponents])
        print("負 | " + " | ".join(loss_values) + f" | {total_losses}({total_enemy_losses})") 
        
        # 準備 '平' 行 
        draw_values = [str(stats[op]['draw']) for op in opponents] 
        total_draws = sum([stats[op]['draw'] for op in opponents]) 
        print("平 | " + " | ".join(draw_values) + f" | {total_draws}") 
        
        # 準備 '勝率' 行
        win_rate_values = []
        for op in opponents:
            player_rate = stats[op]['win_rate']
            # 計算enemy的勝率時使用對調後的勝負數據
            enemy_wins = win_rate_table[op][player_prof]['loss']
            enemy_losses = win_rate_table[op][player_prof]['win']
            enemy_rate = (enemy_wins / (enemy_wins + enemy_losses)) * 100 if (enemy_wins + enemy_losses) > 0 else 0
            win_rate_values.append(f"{player_rate:.2f}%({enemy_rate:.2f}%)")
        
        total_win_rate = (total_wins / (total_wins + total_losses)) * 100 if (total_wins + total_losses) > 0 else 0 
        total_enemy_win_rate = (total_enemy_wins / (total_enemy_wins + total_enemy_losses)) * 100 
        print("勝率 | " + " | ".join(win_rate_values) + f" | {total_win_rate:.2f}%({total_enemy_win_rate:.2f}%)")


    input("對戰完成。按Enter返回主選單...")

    return

def high_level_test_ai_vs_ai(model_path_1, model_path_2, professions, skill_mgr, num_battles=100):
    """
    (3) 高段環境測試: 雙方都是 AI，交叉戰鬥100場
 

    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率（不計入平局）
    """
    # 初始化結果字典
    # test ray if init here

    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0} for op in professions if op != p} for p in professions}
    # print(results)
    total_combinations = len(professions) * (len(professions) - 1)
    current_combination = 0
    beconfig = make_env_config(skill_mgr=skill_mgr, professions=professions,show_battlelog=True)
    config = (
    PPOConfig()
    .environment(
        env=BattleEnv,            # 指定我們剛剛定義的環境 class
        env_config=beconfig# 傳入給 env 的一些自定設定
    )
    .env_runners(num_env_runners=1,sample_timeout_s=120)  # 可根據你的硬體調整
    .framework("torch")            # 或 "tf"
    )
    benv = BattleEnv(config=beconfig)
    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
    config = config.multi_agent(
    policies={
        "player_policy": (None, benv.observation_space, benv.action_space, {}),
        "enemy_policy": (None, benv.observation_space, benv.action_space, {}),
        },
    policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: 
        "player_policy" if agent_id == "player" else "enemy_policy"
        )
    check_point_path = "my_battle_ppo_checkpoints"
    check_point_path = os.path.abspath("my_battle_ppo_checkpoints")
    trainer = config.build()  # 用新的 API 构建训练器
    trainer.restore(check_point_path)
    
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
                    p_act = trainer.compute_single_action(obs['player'], policy_id="player_policy")
                    # if p act in mask is 0, then choose random action
                    e_act = trainer.compute_single_action(obs['enemy'] ,policy_id="enemy_policy")
                    actions = {"player": p_act, "enemy": e_act}
                    obs, rew, done, tru, info = env.step(actions)
                    done = done["__all__"]

                
                res = info["__common__"]["result"]

                # 判斷結果
                if res  == 1:
                    results[p.name][op.name]['win'] += 1
                elif res == -1:
                    results[p.name][op.name]['loss'] += 1
                else:
                    results[p.name][op.name]['draw'] += 1

                # 顯示進度
                if battle_num % 10 == 0:
                    print(f"  完成 {battle_num}/{num_battles} 場")

    # 計算勝率
   # 修改後的程式碼
    win_rate_table = {} 
    enemy_win_rate_table = {}  # 新增enemy的表格

    for p in professions: 
        if p.name not in win_rate_table: 
            win_rate_table[p.name] = {} 
            enemy_win_rate_table[p.name] = {}  # 初始化enemy的數據
        for op in professions: 
            if p == op: 
                continue 
            # Player方數據
            wins = results[p.name][op.name]['win'] 
            losses = results[p.name][op.name]['loss'] 
            draws = results[p.name][op.name]['draw'] 
            total = wins + losses  
            win_rate = (wins / total) * 100 if total > 0 else 0 
            
            # Enemy方數據 (對調勝負)
            enemy_wins = losses  # enemy的勝 = player的負
            enemy_losses = wins  # enemy的負 = player的勝
            enemy_win_rate = (enemy_wins / total) * 100 if total > 0 else 0
            
            win_rate_table[p.name][op.name] = { 
                'win': wins, 
                'loss': losses, 
                'draw': draws, 
                'win_rate': win_rate 
            }
            enemy_win_rate_table[p.name][op.name] = {
                'win': enemy_wins,
                'loss': enemy_losses,
                'draw': draws,
                'win_rate': enemy_win_rate
            }

    # 顯示結果 
    print("\n=== 每個職業相互對戰100場的勝率（括號內為enemy方數據）===") 
    for player_prof, stats in win_rate_table.items(): 
        opponents = list(stats.keys()) 
        print(f"\n職業 {player_prof}") 
        print(" | ".join(opponents) + " |") 
        
        # 準備 '勝' 行 
        win_values = []
        for op in opponents:
            player_wins = stats[op]['win']
            # 先找到對手視角的數據，然後取其losses作為enemy的wins
            enemy_wins = win_rate_table[op][player_prof]['loss']  # 關鍵改變：使用loss而不是win
            win_values.append(f"{player_wins}({enemy_wins})")
        
        total_wins = sum([stats[op]['win'] for op in opponents]) 
        total_enemy_wins = sum([win_rate_table[op][player_prof]['loss'] for op in opponents])
        print("勝 | " + " | ".join(win_values) + f" | {total_wins}({total_enemy_wins})") 
        
        # 準備 '負' 行 
        loss_values = []
        for op in opponents:
            player_losses = stats[op]['loss']
            # 同樣，對手的wins將作為enemy的losses
            enemy_losses = win_rate_table[op][player_prof]['win']  # 關鍵改變：使用win而不是loss
            loss_values.append(f"{player_losses}({enemy_losses})")
        
        total_losses = sum([stats[op]['loss'] for op in opponents]) 
        total_enemy_losses = sum([win_rate_table[op][player_prof]['win'] for op in opponents])
        print("負 | " + " | ".join(loss_values) + f" | {total_losses}({total_enemy_losses})") 
        
        # 準備 '平' 行 
        draw_values = [str(stats[op]['draw']) for op in opponents] 
        total_draws = sum([stats[op]['draw'] for op in opponents]) 
        print("平 | " + " | ".join(draw_values) + f" | {total_draws}") 
        
        # 準備 '勝率' 行
        win_rate_values = []
        for op in opponents:
            player_rate = stats[op]['win_rate']
            # 計算enemy的勝率時使用對調後的勝負數據
            enemy_wins = win_rate_table[op][player_prof]['loss']
            enemy_losses = win_rate_table[op][player_prof]['win']
            enemy_rate = (enemy_wins / (enemy_wins + enemy_losses)) * 100 if (enemy_wins + enemy_losses) > 0 else 0
            win_rate_values.append(f"{player_rate:.2f}%({enemy_rate:.2f}%)")
        
        total_win_rate = (total_wins / (total_wins + total_losses)) * 100 if (total_wins + total_losses) > 0 else 0 
        total_enemy_win_rate = (total_enemy_wins / (total_enemy_wins + total_enemy_losses)) * 100 
        print("勝率 | " + " | ".join(win_rate_values) + f" | {total_win_rate:.2f}%({total_enemy_win_rate:.2f}%)")

    input("對戰完成。按Enter返回主選單...")

    return


import os
import math
import random

def compute_ai_elo(model_path_1, professions, skill_mgr, base_elo=1000, opponent_elo=1500, num_battles=100, K=32):
    """
    計算 AI 的 ELO 分數，與固定 ELO1500 的隨機電腦對戰。

    參數:
        model_path_1: AI 模型的路徑
        professions: 職業列表，每個職業物件應有 .name 屬性
        skill_mgr: 技能管理器
        base_elo: AI 的初始 ELO 分數
        opponent_elo: 對手（電腦）的固定 ELO 分數
        num_battles: 每種攻擊順序的對戰場數 (總場數 = num_battles * 2)
        K: ELO 計算中的 K 值
    """
    print("=== AI ELO 測試 ===")

    # 初始化 ELO 結果的字典
    elo_results = {}
    total_first = 0
    total_second = 0
    randomELO = 1500
    
    # 選擇你要測試的 policy 1. Player 2. Enemy
    ch = input("請選擇要測試的策略：1. Player 2. Enemy")
    if ch == "1":
        choice = "player_policy"
    else:
        choice = "enemy_policy"
    

    # 設定環境配置
    beconfig = make_env_config(skill_mgr=skill_mgr, professions=professions, show_battlelog=False)
    config = (
        PPOConfig()
        .environment(
            env=BattleEnv,
            env_config=beconfig
        )
        .env_runners(num_env_runners=1, sample_timeout_s=120)
        .framework("torch")
    )
    benv = BattleEnv(config=beconfig)
    # 定義多代理策略
    config = config.multi_agent(
        policies={
            "player_policy": (None, benv.observation_space, benv.action_space, {}),
            "enemy_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: 
            "player_policy" if agent_id == "player" else "enemy_policy"
    )
    config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

    # 建立訓練器並載入檢查點
    check_point_path = os.path.abspath("my_battle_ppo_checkpoints")
    trainer = config.build()
    trainer.restore(check_point_path)
    
    for p in professions:
        print(f"\n=== 職業: {p.name} ===")
        # 初始化先攻和後攻的 ELO
        elo_first = base_elo
        elo_second = base_elo

        # 測試先攻（玩家政策）
        print("  測試先攻...")
        for battle_num in range(1, num_battles // 2 + 1):
            # 初始化 BattleEnv，AI 作為先攻方

            env = BattleEnv(make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()

            while not done:
                # AI 的動作
                if choice == "player_policy":
                    ai_act = trainer.compute_single_action(obs['player'], policy_id="player_policy")
                else:
                    ai_act = trainer.compute_single_action(obs['player'], policy_id="enemy_policy")
            
                # 電腦（隨機）動作
                # random action from 0 - 2
                enemy_act = random.choice([0, 1, 2])
                
                actions = {"player": ai_act, "enemy": enemy_act}
                obs, rew, done_dict, tru, info = env.step(actions)
                done = done_dict["__all__"]

            res = info["__common__"]["result"]
            # 確定比賽結果
            if res == 1:
                score = 1  # AI 勝利
            elif res == -1:
                score = 0  # AI 敗北
            else:
                score = 0.5  # 平局

            # 計算期望分數
            expected = 1 / (1 + 10 ** ((randomELO - elo_first) / 400))
            # 更新 ELO
            elo_first += K * (score - expected)

            # 顯示進度
            if battle_num % 10 == 0:
                print(f"    完成 {battle_num}/{num_battles // 2} 場 (先攻)")

        # 測試後攻（敵人政策）
        print("  測試後攻...")
        for battle_num in range(1, num_battles // 2 + 1):
            # 初始化 BattleEnv，AI 作為後攻方
            env_config = make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p)  # pr1=None 表示對手為隨機電腦
            env = BattleEnv(env_config)
            done = False
            obs, _ = env.reset()

            while not done:
                # 電腦（隨機）動作
                enemy_act = random.choice([0, 1, 2])
                # AI 的動作
                # AI 的動作
                if choice == "player_policy":
                    ai_act = trainer.compute_single_action(obs['enemy'], policy_id="player_policy")
                else:
                    ai_act = trainer.compute_single_action(obs['enemy'], policy_id="enemy_policy")
                    
                actions = {"player": enemy_act, "enemy": ai_act}
                obs, rew, done_dict, tru, info = env.step(actions)
                done = done_dict["__all__"]

            res = info["__common__"]["result"]
            # 確定比賽結果
            if res == 1:
                score = 1  # AI 勝利
            elif res == -1:
                score = 0  # AI 敗北
            else:
                score = 0.5  # 平局

            # 計算期望分數
            expected = 1 / (1 + 10 ** ((randomELO - elo_second) / 400))
            # 更新 ELO
            elo_second += K * (score - expected)

            # 顯示進度
            if battle_num % 10 == 0:
                print(f"    完成 {battle_num}/{num_battles // 2} 場 (後攻)")

        # 計算總和 ELO
        total_elo = (elo_first + elo_second) / 2

        # 儲存結果
        elo_results[p.name] = {
            "先攻方 ELO": elo_first,
            "後攻方 ELO": elo_second,
            "總和 ELO": total_elo
        }

        # 累加總和
        total_first += elo_first
        total_second += elo_second

    # 計算整體總和
    overall_total = len(professions)
    average_first = total_first / overall_total
    average_second = total_second / overall_total
    average_total = (average_first + average_second) / 2

    # 輸出結果表格
    print(f"\n=== ELO 結果 {choice}===")
    print(f"{'職業':<15} | {'先攻方 ELO':<15} | {'後攻方 ELO':<15} | {'總和 ELO':<10}")
    print("-" * 60)
    for prof, elos in elo_results.items():
        print(f"{prof:<15} | {elos['先攻方 ELO']:<15.2f} | {elos['後攻方 ELO']:<15.2f} | {elos['總和 ELO']:<10.2f}")
    print("-" * 60)
    print(f"{'總和':<15} | {average_first:<15.2f} | {average_second:<15.2f} | {average_total:<10.2f}")

    input("按 Enter 返回主選單...")




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