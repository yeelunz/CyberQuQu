# train_methods.py

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from battle_env import BattleEnv
from skills import SkillManager
from professions import (
    Paladin, Mage,Assassin,Archer, Berserker,DragonGod,BloodGod,
    SteadfastWarrior,SunWarrior,Ranger,ElementalMage,HuangShen,
    GodOfStar
)
from effect_mapping import EFFECT_MAPPING, EFFECT_VECTOR_LENGTH

def cross_evaluation(model, skill_mgr, professions, n_eval_episodes=5):
    """
    交叉勝率評估 (1v1)
    - model: 訓練中的PPO模型
    - 返回 wins, losses, draws 的字典
    """
    wins = {}
    losses = {}
    draws = {}

    for p1 in professions:
        for p2 in professions:
            if p1.profession_id == p2.profession_id:
                continue
            key = (p1.profession_id, p2.profession_id)
            wins[key] = 0
            losses[key] = 0
            draws[key] = 0
            for _ in range(n_eval_episodes):
                # 建立環境：p1 vs p2
                env = BattleEnv(
                    team_size=1,
                    enemy_team_size=1,
                    max_rounds=30,
                    player_team=[{
                        "profession": p1,
                        "hp": p1.base_hp,
                        "max_hp": p1.max_hp,
                        "status": {},
                        "skip_turn": False,
                        "is_defending": False,
                        "times_healed": 0,
                        "next_heal_double": False,
                        "damage_multiplier": 1.0,
                        "damage_multiplier_turns": 0,
                        "defend_multiplier": 1.0,
                        "defend_multiplier_turns": 0,
                        "heal_multiplier": 1.0,
                        "heal_multiplier_turns": 0,
                        "battle_log": [],
                        "cooldowns": {},
                    }],
                    enemy_team=[{
                        "profession": p2,
                        "hp": p2.base_hp,
                        "max_hp": p2.max_hp,
                        "status": {},
                        "skip_turn": False,
                        "is_defending": False,
                        "times_healed": 0,
                        "next_heal_double": False,
                        "damage_multiplier": 1.0,
                        "damage_multiplier_turns": 0,
                        "defend_multiplier": 1.0,
                        "defend_multiplier_turns": 0,
                        "heal_multiplier": 1.0,
                        "heal_multiplier_turns": 0,
                        "battle_log": [],
                        "cooldowns": {},
                    }],
                    skill_mgr=skill_mgr,
                    show_battle_log=False
                )
                obs = env.reset()
                done = False
                while not done:
                    # 模型選擇動作
                    action, _ = model.predict(obs, deterministic=True)
                    # 隨機敵方動作
                    enemy_skill_ids = p2.get_available_skill_ids()
                    enemy_action = np.random.choice(enemy_skill_ids) - 3 * p2.profession_id
                    enemy_action = enemy_action if enemy_action in [0, 1, 2] else 0
                    # 環境執行步驟
                    combined_action = [action[0], enemy_action]  # 包含我方和敵方的行動
                    obs, reward, done, _ = env.step(combined_action)
                    if done:
                        if reward > 0:
                            wins[key] += 1
                        elif reward < 0:
                            losses[key] += 1
                        else:
                            draws[key] += 1
                        break
    print('finish cross_evaluation')
    return wins, losses, draws

def compute_win_rate_table(wins, losses, draws, professions):
    """
    計算勝率表，包含每個職業對其他職業的勝場、負場、平手場次和勝率，
    以及每個職業的總勝場、總負場、總平手和總勝率。

    返回一個嵌套字典：
    {
        '職業名稱': {
            'opponents': {
                '對手職業名稱': {
                    'win': 勝場數,
                    'loss': 負場數,
                    'draw': 平手場次,
                    'win_rate': 勝率百分比
                },
                ...
            },
            'total_wins': 總勝場,
            'total_losses': 總負場,
            'total_draws': 總平手,
            'total_win_rate': 總勝率百分比
        },
        ...
    }
    """
    table = {}

    for p1 in professions:
        table[p1.name] = {
            'opponents': {},
            'total_wins': 0,
            'total_losses': 0,
            'total_draws': 0,
            'total_win_rate': 0.0
        }

        for p2 in professions:
            if p1.profession_id == p2.profession_id:
                continue

            key = (p1.profession_id, p2.profession_id)
            win = wins.get(key, 0)
            loss = losses.get(key, 0)
            draw = draws.get(key, 0)
            total = win + loss + draw
            win_rate = (win / total) * 100 if total > 0 else 0.0

            table[p1.name]['opponents'][p2.name] = {
                'win': win,
                'loss': loss,
                'draw': draw,
                'win_rate': win_rate
            }

            table[p1.name]['total_wins'] += win
            table[p1.name]['total_losses'] += loss
            table[p1.name]['total_draws'] += draw

        total_matches = table[p1.name]['total_wins'] + table[p1.name]['total_losses'] + table[p1.name]['total_draws']
        table[p1.name]['total_win_rate'] = (table[p1.name]['total_wins'] / total_matches) * 100 if total_matches > 0 else 0.0

    return table

def train_iteratively(num_iterations, max_episodes, skill_mgr, professions, save_model_path, desired_total_steps=1):
    """
    交叉對戰迭代訓練單一模型。
    - 每個迭代中，每個職業與其他所有職業對戰1次。
    - save_model_path: 模型保存路徑前綴。
    """
    # 初始化環境列表
    envs = []
    profession_pairs = []
    for player_prof in professions:
        for enemy_prof in professions:
            if player_prof.profession_id != enemy_prof.profession_id:
                for _ in range(1):  # 每對職業對戰1次
                    envs.append(lambda p=player_prof, e=enemy_prof: BattleEnv(
                        team_size=1,
                        enemy_team_size=1,
                        max_rounds=30,
                        player_team=[{
                            "profession": p,
                            "hp": p.base_hp,
                            "max_hp": p.max_hp,
                            "status": {},
                            "skip_turn": False,
                            "is_defending": False,
                            "times_healed": 0,
                            "next_heal_double": False,
                            "damage_multiplier": 1.0,
                            "damage_multiplier_turns": 0,
                            "defend_multiplier": 1.0,
                            "defend_multiplier_turns": 0,
                            "heal_multiplier": 1.0,
                            "heal_multiplier_turns": 0,
                            "battle_log": [],
                            "cooldowns": {},
                            "dragon_soul_stack": 0
                        }],
                        enemy_team=[{
                            "profession": e,
                            "hp": e.base_hp,
                            "max_hp": e.max_hp,
                            "status": {},
                            "skip_turn": False,
                            "is_defending": False,
                            "times_healed": 0,
                            "next_heal_double": False,
                            "damage_multiplier": 1.0,
                            "damage_multiplier_turns": 0,
                            "defend_multiplier": 1.0,
                            "defend_multiplier_turns": 0,
                            "heal_multiplier": 1.0,
                            "heal_multiplier_turns": 0,
                            "battle_log": [],
                            "cooldowns": {},
                            "dragon_soul_stack": 0
                        }],
                        skill_mgr=skill_mgr,
                        show_battle_log=False
                    ))
                profession_pairs.append((player_prof, enemy_prof))

    # 使用 DummyVecEnv 包裝所有環境
    vec_env = DummyVecEnv(envs[:1])  # 因為動作空間是固定的，這裡簡化為一個環境

    # 初始化單一模型
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # 用於記錄每次迭代的勝率
    win_rates = []
    print('num_iterations:', num_iterations)
    print('max_episodes:', desired_total_steps)
    print('number of profession_pairs:', len(profession_pairs))

    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1}/{num_iterations} ===")

        # 訓練模型
        print("訓練模型...")
        model.learn(total_timesteps=desired_total_steps, progress_bar=True)
        model.save(f"{save_model_path}_iter{i+1}.zip")
        
        # 評估勝率
        wins, losses, draws = cross_evaluation(
            model,
            skill_mgr,
            professions,
            n_eval_episodes=5  # 每對職業對戰5次
        )
        # 計算勝率表
        win_rate_table = compute_win_rate_table(wins, losses, draws, professions)
        win_rates.append({
            'iteration': i+1,
            'win_rate_table': win_rate_table
        })

        # 打印勝率表
        print(f"\n=== Iteration {i+1} 結果 ===")
        for player_prof, stats in win_rate_table.items():
            print(f"\n職業 {player_prof}")
            opponents = list(stats['opponents'].keys())
            print(" | ".join(opponents) + " |")

            # 準備 '勝' 行
            win_values = [str(stats['opponents'][op]['win']) for op in opponents]
            total_wins = stats['total_wins']
            print("勝 | " + " | ".join(win_values) + f" | {total_wins}")

            # 準備 '負' 行
            loss_values = [str(stats['opponents'][op]['loss']) for op in opponents]
            total_losses = stats['total_losses']
            print("負 | " + " | ".join(loss_values) + f" | {total_losses}")

            # 準備 '平' 行
            draw_values = [str(stats['opponents'][op]['draw']) for op in opponents]
            total_draws = stats['total_draws']
            print("平 | " + " | ".join(draw_values) + f" | {total_draws}")

            # 準備 '勝率' 行
            win_rate_values = [f"{stats['opponents'][op]['win_rate']:.2f}%" for op in opponents]
            total_win_rate = stats['total_win_rate']
            print("勝率 | " + " | ".join(win_rate_values) + f" | {total_win_rate:.2f}%")

    # 將勝率記錄保存為 CSV
    if win_rates:
        # 假設只保存最後一次的 win_rate_table
        final_win_rate_table = win_rates[-1]['win_rate_table']
        # 將嵌套字典轉換為適合保存的格式
        records = []
        for player_prof, stats in final_win_rate_table.items():
            for opponent_prof, result in stats['opponents'].items():
                records.append({
                    '職業': player_prof,
                    '對手職業': opponent_prof,
                    '勝': result['win'],
                    '負': result['loss'],
                    '平': result['draw'],
                    '勝率': f"{result['win_rate']:.2f}%"
                })
        df = pd.DataFrame(records)
        df.to_csv("training_win_rates.csv", index=False)
        print("\n=== 訓練完成，勝率記錄已保存至 training_win_rates.csv ===")
    else:
        print("\n=== 無勝率記錄可保存 ===")
