# train_methods.py
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from .battle_env import BattleEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from collections import defaultdict
from .train_var import player_train_times, enemy_train_times
import os
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.catalog import ModelCatalog
from datetime import datetime
import json
from .global_var import globalVar as gl

from .professions import build_professions
from .skills import build_skill_manager
import threading
import time

class MaskedFullyConnectedNetwork(TorchModelV2, nn.Module):
    """
    基於 RLlib 的預設 FullyConnectedNetwork，添加動作掩碼處理。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 使用內建 FullyConnectedNetwork 作為基礎模型
        self.base_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_base"
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        前向傳播，將掩碼應用到 logits 上。
        """
        # 基礎模型的輸出 logits
        base_out, state = self.base_model(input_dict, state, seq_lens)

        # 提取掩碼：假設 obs 的前 3 維是掩碼
        mask = input_dict["obs"][:, :3]  # (B, 3)

        # 掩碼處理：將無效的 logits 設為極大負值
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = base_out + inf_mask

        self._last_output = masked_logits
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        """
        繼承基礎模型的 value_function。
        """
        return self.base_model.value_function()


ModelCatalog.register_custom_model("my_mask_model", MaskedFullyConnectedNetwork)


stop_training_flag = threading.Event()


def multi_agent_cross_train(num_iterations,
                            model_name="my_multiagent_ai",
                            hyperparams=None):
    """
    多智能體交叉訓練
    """

    professions = build_professions()
    skill_mgr = build_skill_manager()

    if hyperparams is None:
        hyperparams = {}

    print("=== 開始多智能體交叉訓練 ===")
    print("模型名稱:", model_name)
    print("超參數:", hyperparams)
    

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
        .training(
            model={
                "custom_model": "my_mask_model",
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "ReLU",
                "vf_share_layers": False
            },
            use_gae=True,
            lr=hyperparams.get("learning_rate", 1e-4),
            train_batch_size=hyperparams.get("train_batch_size", 4000),
            # ... 如果有更多超參數需要動態帶入，繼續加 ...
        )
    )

    benv = BattleEnv(beconfig)
    config = config.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs:
            "shared_policy" if agent_id == "player" else "shared_policy"
    )

    # (保留) 遷移到新API
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    # 在這裡做「模型初始化中」的邏輯
    print("=== 正在執行 config.build() 中 ===")
    algo = config.build()
    print("=== 模型初始化完成 ===")

    # 先 yield 一個事件，告知「初始化完成」(前端會判斷 type=initialized)
    yield {
        "type": "initialized",
        "message": "環境初始化完成"
    }

    for i in range(num_iterations):
        if stop_training_flag.is_set():
            yield {
                "type": "stopped",
                "message": "訓練已被終止。"
            }
            break

        result = algo.train()
        print(f"=== Iteration {i + 1} ===")

        # 將需要的監控指標一起回傳
        yield {
            "type": "iteration",
            "iteration": i + 1,
            "timesteps_total": result.get("timesteps_total", 0), 
            "date": result.get("date", ""),
            "learner": result["info"]["learner"],
            "num_episodes": result["env_runners"]["num_episodes"],
            "episode_len_mean": result["env_runners"]["episode_len_mean"],
            "timers": result["timers"],
            "sampler_perf": result["env_runners"]["sampler_perf"],
            "cpu_util_percent": result["perf"]["cpu_util_percent"],
            "ram_util_percent": result["perf"]["ram_util_percent"]
        }

    # 訓練完成或被終止
    if not stop_training_flag.is_set():
        # 訓練完成 => 存檔
        save_root = os.path.join("data", "saved_models", model_name)
        os.makedirs(save_root, exist_ok=True)
        checkpoint_dir = algo.save(save_root)
        print("Checkpoint saved at", checkpoint_dir)

        meta_path = os.path.join(save_root, "training_meta.json")
        meta_info = {
            "model_name": model_name,
            "num_iterations": num_iterations,
            "hyperparams": hyperparams,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": gl["version"],
            "elo_result": None,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)

        print(f"模型訓練完成，相關資訊已儲存至 {save_root}")

        # 最後再 yield 一個事件，告知訓練流程已整體結束 (type=done)
        yield {
            "type": "done",
            "message": "訓練全部結束"
        }
    else:
        print("訓練過程被終止。")

    # 重置停止標誌
    stop_training_flag.clear()




    

import itertools
import math
from collections import defaultdict

#------------------------------------------------------------
# 以下示範 (2)~(4) 選單對應的測試或ELO
#------------------------------------------------------------
def calculate_ehi(combine_rate_table):
    
    """
    計算環境健康指標 (EHI)
    :param combine_rate_table: 職業之間的勝率表
    :return: EHI 分數
    """
    # 1. 收集每個職業的平均勝率
    avg_win_rates = {}
    for prof, opponents in combine_rate_table.items():
        win_rates = [data['win_rate'] for data in opponents.values()]
        avg_win = sum(win_rates) / len(win_rates) if win_rates else 0
        avg_win_rates[prof] = avg_win

    # 2. 計算香農熵（Shannon Entropy）
    total = sum(avg_win_rates.values())
    probabilities = [win / total for win in avg_win_rates.values()] if total > 0 else [0 for _ in avg_win_rates]
    shannon_entropy = -sum(p * math.log(p, 2) for p in probabilities if p > 0)

    # 正規化熵值（最大熵為 log2(N)）
    N = len(avg_win_rates)
    max_entropy = math.log2(N) if N > 0 else 1
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0

    # 3. 計算基尼係數（Gini Coefficient）
    sorted_win_rates = sorted(avg_win_rates.values())
    cumulative = 0
    cumulative_sum = 0
    for i, win in enumerate(sorted_win_rates, 1):
        cumulative += win
        cumulative_sum += cumulative
    if total > 0 and N > 1:
        gini = (2 * cumulative_sum) / (N * total) - (N + 1) / N
    else:
        gini = 0

    # 正規化基尼係數（0到1）
    # 基尼係數本身已在 [0,1] 範圍內
    normalized_gini = gini

    # 4. 計算剋制鏈（Counter Cycles）
    professions = list(combine_rate_table.keys())
    count_cycles = 0
    threshold = 60  # 勝率超過此值表示剋制
    for cycle in itertools.permutations(professions, 3):
        a, b, c = cycle
        if (combine_rate_table[a].get(b, {}).get('win_rate', 0) > threshold and
            combine_rate_table[b].get(c, {}).get('win_rate', 0) > threshold and
            combine_rate_table[c].get(a, {}).get('win_rate', 0) > threshold):
            count_cycles += 1

    # 最大可能的剋制環數
    C_max = math.comb(len(professions), 3) if len(professions) >= 3 else 1
    normalized_cycles = count_cycles / C_max if C_max > 0 else 0

    # 5. 計算 EHI（加權綜合）
    # 定義權重，可以根據需求調整
    weight_entropy = 0.4
    weight_gini = 0.4
    weight_cycles = 0.2

    # EHI = w1 * normalized_entropy + w2 * (1 - normalized_gini) + w3 * normalized_cycles
    # (1 - normalized_gini) 表示基尼係數越低，健康度越高
    ehi = (weight_entropy * normalized_entropy +
           weight_gini * (1 - normalized_gini) +
           weight_cycles * normalized_cycles)

    # 確保 EHI 在 [0,1] 範圍內
    ehi_normalized = min(max(ehi, 0), 1)

    return round(ehi_normalized, 3)

def calculate_esi(combine_rate_table, calculate_combined_metrics_func):
    """
    計算環境穩定性指標 (ESI)
    :param combine_rate_table: 職業之間的勝率表
    :param calculate_combined_metrics_func: 用於計算指標的函數
    :return: ESI 分數
    """
    import copy

    original_metrics = calculate_combined_metrics_func(combine_rate_table)
    
    metrics_changes = defaultdict(list)
    
    for prof in combine_rate_table.keys():
        for opp in combine_rate_table[prof].keys():
            for delta in [-1, 1]:  # 微調 -1% 和 +1%
                # 確保勝率在 0 到 100 之間
                new_win_rate = max(0, min(100, combine_rate_table[prof][opp]['win_rate'] + delta))
                if new_win_rate == combine_rate_table[prof][opp]['win_rate']:
                    continue  # 無變化
                
                # 創建新的勝率表
                modified_table = copy.deepcopy(combine_rate_table)
                modified_table[prof][opp]['win_rate'] = new_win_rate
                
                # 計算修改後的指標
                modified_metrics = calculate_combined_metrics_func(modified_table)
                
                # 計算指標的變化
                for key in original_metrics[prof]:
                    change = abs(modified_metrics[prof][key] - original_metrics[prof][key])
                    metrics_changes[key].append(change)
    
    # 計算每個指標的平均變化
    average_changes = {}
    for key, changes in metrics_changes.items():
        average_changes[key] = sum(changes) / len(changes) if changes else 0
    
    # 綜合指標（加權平均）
    # 假設所有指標權重相同
    total_keys = len(average_changes)
    esi = sum(average_changes.values()) / total_keys if total_keys > 0 else 0
    
    # 反轉指標，因為變化越小越穩定
    # 假設 ESI = 1 / (1 + esi) 使其範圍接近 0 到 1
    esi_score = 1 / (1 + esi) if (1 + esi) != 0 else 0
    
    return round(esi_score, 3)

def calculate_mpi(combine_rate_table, calculate_combined_metrics_func):
    """
    計算中游壓力指標 (MPI)
    :param combine_rate_table: 職業之間的勝率表
    :param calculate_combined_metrics_func: 用於計算指標的函數
    :return: MPI 分數
    """
    metrics = calculate_combined_metrics_func(combine_rate_table)
    
    # 根據 Adjusted NAS 將職業分層
    sorted_profs = sorted(metrics.keys(), key=lambda x: metrics[x]['Advanced NAS'], reverse=True)
    n = len(sorted_profs)
    if n == 0:
        return 0
    
    # top_third = sorted_profs[:n//2.5]
    # bottom_third = sorted_profs[-n//2.5:]
    # middle_third = sorted_profs[n//3:2*n//3]
    # 重新設計 top = 25% mid = 50% bot = 25%
    top_third = sorted_profs[:n//4]
    bottom_third = sorted_profs[-n//4:]
    middle_third = sorted_profs[n//4:3*n//4]
    
    # 計算中層職業被壓制的程度
    suppression_scores = []
    for middle_prof in middle_third:
        suppression = 0
        for top_prof in top_third:
            win_rate = combine_rate_table[top_prof][middle_prof]['win_rate']
            if win_rate > 70:
                suppression += (win_rate - 70)  # 超過50%的勝率表示壓制
        suppression_scores.append(suppression)
    
    # MPI 為中層職業平均壓制程度
    mpi = sum(suppression_scores) / len(suppression_scores) if suppression_scores else 0
    
    # 正規化 MPI（假設最大壓制程度為50）
    normalized_mpi = min(mpi / 50, 1)
    
    return round(normalized_mpi, 3)



def calculate_combined_metrics(combine_rate_table, max_iter=200, lr=0.01, tol=1e-6):
    """
    計算每個職業的 EIR, A-NAS (基於 BTL), MSI, PI20 和 WI 指標
    :param combine_rate_table: 職業之間的勝率表
    :param max_iter: BTL 模型的最大迭代次數
    :param lr: BTL 模型的學習率
    :param tol: 收斂容忍度
    :return: 每個職業的指標
    """
    metrics = {}
    
    professions = list(combine_rate_table.keys())
    theta = {prof: 1.0 for prof in professions}  # 初始 theta 值

    # BTL 模型的迭代估計
    for iteration in range(max_iter):
        grad = {prof: 0.0 for prof in professions}
        for i in professions:
            opponents = combine_rate_table[i]
            for j, data in opponents.items():
                if j == i:
                    continue  # 忽略自我對戰
                w_ij = data['win_rate'] / 100.0  # 轉換為 0~1 之間
                denom = theta[i] + theta[j]
                P_ij = theta[i] / denom if denom > 0 else 0.0
                grad[i] += (w_ij - P_ij)
        
        # 更新 theta 值
        max_grad = 0.0
        for prof in professions:
            update = math.exp(lr * grad[prof])
            theta[prof] *= update
            max_grad = max(max_grad, abs(grad[prof]))
        
        # 檢查是否收斂
        if max_grad < tol:
            # print(f"BTL 模型在迭代 {iteration + 1} 次後收斂。")
            break
    else:
        pass
        # print(f"BTL 模型在達到最大迭代次數 {max_iter} 後停止。")

    # 計算 theta 的平均值
    theta_values = list(theta.values())
    mean_theta = sum(theta_values) / len(theta_values) if theta_values else 1.0

    # 計算標準差
    variance_theta = sum((th - mean_theta) ** 2 for th in theta_values) / len(theta_values) if theta_values else 0.0
    std_theta = math.sqrt(variance_theta) if variance_theta > 0 else 1.0  # 避免除以零

    # 將 theta 映射到 A-NAS 指標，使得平均值對應到 0
    a_nas_dict = {}
    for prof in professions:
        a_nas = (theta[prof] - mean_theta) / std_theta
        a_nas_dict[prof] = round(a_nas, 3)

    # 接下來計算其他指標
    for player_prof, opponents in combine_rate_table.items():
        # 收集該職業對其他所有職業的勝率
        win_rates = [opponent_data['win_rate'] for opp, opponent_data in opponents.items() if opp != player_prof]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 50.0
        variance = sum((rate - avg_win_rate) ** 2 for rate in win_rates) / len(win_rates) if win_rates else 0.0
        std_dev = math.sqrt(variance) if win_rates else 0.0

        # 計算 EIR
        eir = (sum(rate for rate in win_rates) / len(win_rates)) if win_rates else 0.0

        # 計算 MSI
        msi = 1 - (std_dev / 100.0)

        # 計算 PI20
        pi20 = sum(1 for rate in win_rates if abs(rate - avg_win_rate) > 20) / len(win_rates) if win_rates else 0.0

        # 計算 WI
        total_influence = 0.0
        for other_prof, opponents_data in combine_rate_table.items():
            if other_prof == player_prof:
                continue
            opponent_win_rates = [data['win_rate'] for opp, data in opponents_data.items() if opp != player_prof]
            original_avg_win_rate = sum(opponents_data[opp]['win_rate'] for opp in opponents_data) / len(opponents_data) if opponents_data else 50.0
            new_avg_win_rate = sum(opponent_win_rates) / len(opponent_win_rates) if opponent_win_rates else original_avg_win_rate
            total_influence += (new_avg_win_rate - original_avg_win_rate)
        
        metrics[player_prof] = {
            "EIR": round(round(eir, 2)*0.5 + round(total_influence, 2)*0.5,2),
            "Advanced NAS": a_nas_dict[player_prof],
            "MSI": round(msi, 2),
            "PI20": round(pi20, 2),
        }
    
    return metrics

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
    # all professions add in skillusedFre
    skillusedFreq = {p.name: {0:0,1:0,2:0} for p in professions}
    
 
    for p in professions:
        for op in professions:
            if p == op:
                continue
            
            current_combination += 1
            print(f"\n對戰 {current_combination}/{total_combinations}: {p.name} VS {op.name}")
            
            for battle_num in range(1, num_battles + 1):
                if battle_num % 2 == 0:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                    )
                else:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=op, pr2=p)
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
                    
                    # 統計技能使用次數，如果p_action = 2，則在 skillusedFreq['p.profession'][2] += 1
                    # e_act = 2，則在 skillusedFreq['e.profession'][2] += 1

                    skillusedFreq[p.name][int(p_act)] += 1
                    skillusedFreq[op.name][int(e_act)] += 1
                    


                    obs, rew, done, tru, info = env.step({
                        "player": p_act,
                        "enemy": e_act
                    })
             

                    done = done["__all__"]

                info = info["__common__"]["result"]
                
                if battle_num % 2 == 1:
                    info = -info

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
        for p in professions: 
            if p.name not in win_rate_table: 
                win_rate_table[p.name] = {} 

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
                
                win_rate_table[p.name][op.name] = { 
                    'win': wins, 
                    'loss': losses, 
                    'draw': draws, 
                    'win_rate': win_rate 
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
        

        # 打印簡潔的勝率表
    print("\n=== 職業對戰勝率表 ===")

    # 獲取所有職業名稱
    professions = list(win_rate_table.keys())

    # 打印表頭
    header = "    | " + " | ".join(professions) + " |"
    print(header)
    print("-" * len(header))
    
    # copy win_rate_table
    combine_rate_table = win_rate_table.copy()
    
    # 
    for player_prof in professions:
        row = [player_prof]  # 當前職業名稱作為行首
        for opponent_prof in professions:
            if player_prof == opponent_prof:
                row.append("-")  # 同職業對戰顯示 "-"
            else:
                combine_rate_table[player_prof][opponent_prof]['win_rate'] = (win_rate_table[player_prof][opponent_prof]['win_rate'] + (100-win_rate_table[opponent_prof][player_prof]['win_rate'])) /2


    # 計算職業指標
    metrics = calculate_combined_metrics(combine_rate_table)
    # 打印指標表 | EIR | Advanced NAS | MSI | PI20
    print("\n=== 職業指標 ===")
    print("職業 | EIR | Advanced NAS | MSI | PI20")
    for prof, data in metrics.items():
        print(f"{prof} | {data['EIR']:.2f} | {data['Advanced NAS']:.2f} | {data['MSI']:.2f} | {data['PI20']:.2f}")
    
    # cal ehi and esi and mpi
    ehi = calculate_ehi(combine_rate_table)
    esi = calculate_esi(combine_rate_table, calculate_combined_metrics)
    mpi = calculate_mpi(combine_rate_table, calculate_combined_metrics)
    print(f"\nEHI: {ehi}, ESI: {esi}, MPI: {mpi}")
    
    res  = {
        "env_evaluation": {
            "ehi": ehi,
            "esi": esi,
            "mpi": mpi
        },
        "profession_evaluation": metrics,
        "combine_win_rate_table": combine_rate_table,
        "profession_skill_used_freq": skillusedFreq,
        "total_battles": int(num_battles*len(professions)*(len(professions)-1))
    }
    
    return res

def high_level_test_ai_vs_ai(model_path_1, model_path_2, professions, skill_mgr, num_battles=100):
    """
    (3) 高段環境測試: 雙方都是 AI，交叉戰鬥100場
 

    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率（不計入平局）
    """
    # 初始化結果字典
    # test ray if init here
    eps = 0
    steps = 0

    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0} for op in professions if op != p} for p in professions}
    skill_usage = {p.name: [0,0,0] for p in professions}
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
        "shared_policy" if agent_id == "player" else "shared_policy"
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
                eps += 1

                # 初始化 BattleEnv，關閉 battle_log 以加快速度
                # 前半 = p 先攻，後半 = op 先攻
                if battle_num % 2 == 1:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                    )
                else:
                    env = BattleEnv(
                        # make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=op, pr2=p)
                    )

                # 編號決定先後攻
                done = False
                obs, _ = env.reset()

                while not done:
                    steps += 1
                    
                    # 取第 0~2 格，獲取合法動作
                    p_act = trainer.compute_single_action(obs['player'], policy_id="shared_policy")
                    # if p act in mask is 0, then choose random action
                    e_act = trainer.compute_single_action(obs['enemy'] ,policy_id="shared_policy")
                    
                    # track skill usage
                    skill_usage[p.name][p_act] += 1
                    skill_usage[op.name][e_act] += 1
                    
                    
                    actions = {"player": p_act, "enemy": e_act}
                    obs, rew, done, tru, info = env.step(actions)
                    done = done["__all__"]

                
                res = info["__common__"]["result"]
                # res 代表先攻方的結果， 因此如果前面是op先攻，則結果要反轉
                if battle_num % 2 == 0:
                    res = -res

                
                # 判斷結果
                if res == 1:
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

    for p in professions: 
        if p.name not in win_rate_table: 
            win_rate_table[p.name] = {} 

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

            
            win_rate_table[p.name][op.name] = { 
                'win': wins, 
                'loss': losses, 
                'draw': draws, 
                'win_rate': win_rate 
            }

    # 顯示結果 
    print("\n=== 每個職業相互對戰100場的勝率（括號內為後攻方之數據）===") 
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
    
    print("\n=== 各職業使用技能0/技能1/技能2的次數統計 ===")
    for prof, skills in skill_usage.items():
        print(f"\n職業 {prof}")
        print(f"技能0: {skills[0]} 次")
        print(f"技能1: {skills[1]} 次")
        print(f"技能2: {skills[2]} 次")
    
    # prnit env info
    # 列印戰鬥平均回合數
    print(f"\n平均回合數: {steps / eps:.2f}")
    
    professions = list(win_rate_table.keys())
    combine_rate_table = win_rate_table.copy()
    # 打印每行數據
    for player_prof in professions:
        row = [player_prof]  # 當前職業名稱作為行首
        for opponent_prof in professions:
            if player_prof == opponent_prof:
                row.append("-")  # 同職業對戰顯示 "-"
            else:
                combine_rate_table[player_prof][opponent_prof]['win_rate'] = (win_rate_table[player_prof][opponent_prof]['win_rate'] + (100-win_rate_table[opponent_prof][player_prof]['win_rate'])) /2


    metrics = calculate_combined_metrics(combine_rate_table)
    # 打印指標表 | EIR | Advanced NAS | MSI | PI20
    print("\n=== 職業指標 ===")
    print("職業 | EIR | Advanced NAS | MSI | PI20")
    for prof, data in metrics.items():
        print(f"{prof} | {data['EIR']:.2f} | {data['Advanced NAS']:.2f} | {data['MSI']:.2f} | {data['PI20']:.2f}")
    
    # cal ehi and esi and mpi
    ehi = calculate_ehi(combine_rate_table)
    esi = calculate_esi(combine_rate_table, calculate_combined_metrics)
    mpi = calculate_mpi(combine_rate_table, calculate_combined_metrics)
    print(f"\nEHI: {ehi}, ESI: {esi}, MPI: {mpi}")
    input("對戰完成。按Enter返回主選單...")

    return


import os
import math
import random


def compute_ai_elo(model_path_1, professions, skill_mgr, base_elo=1500, opponent_elo=1500, num_battles=100, K=32):
    """
    計算 AI 的 ELO 分數，並回報進度。
    修改為生成器，定期 yield 進度資訊。
    """
    print("=== AI ELO 測試 ===")

    # 初始化 ELO 結果的字典
    elo_results = {}
    total_first = 0
    total_second = 0
    randomELO = 1500  # 用於跟AI對戰的「隨機對手」ELO

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
        .training(
            model={
                "custom_model": "my_mask_model",
            }
        )
    )

    benv = BattleEnv(config=beconfig)
    # 定義多代理策略
    config = config.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs:
            "shared_policy"
    )
    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)

    # 建立訓練器並載入檢查點
    check_point_path = os.path.abspath(model_path_1)

    # 注意：這裡只留一次 build
    trainer = config.build()

    if os.path.exists(check_point_path):
        trainer.restore(check_point_path)
    else:
        print(f"檢查點路徑 {check_point_path} 不存在。")

    total_professions = len(professions)
    total_steps = total_professions * num_battles * 2  # 先攻+後攻
    current_step = 0

    for p in professions:
        if stop_training_flag.is_set():
            yield {"type": "stopped", "message": "ELO 計算已被終止。"}
            return

        print(f"\n=== 職業: {p.name} ===")
        # 初始化先攻和後攻 ELO
        elo_first = base_elo
        elo_second = base_elo

        # 測試先攻
        for battle_num in range(1, num_battles // 2 + 1):
            if stop_training_flag.is_set():
                yield {"type": "stopped", "message": "ELO 計算已被終止。"}
                return

            env = BattleEnv(make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()

            while not done:
                ai_act = trainer.compute_single_action(obs['player'], policy_id="shared_policy")
                enemy_act = random.choice([0, 1, 2])  # 隨機對手
                actions = {"player": ai_act, "enemy": enemy_act}
                obs, rew, done_dict, tru, info = env.step(actions)
                done = done_dict["__all__"]

            res = info["__common__"]["result"]
            if res == 1:
                score = 1
            elif res == -1:
                score = 0
            else:
                score = 0.5

            expected = 1 / (1 + 10 ** ((randomELO - elo_first) / 400))
            elo_first += K * (score - expected)

            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            if battle_num % 50 == 0:
                progress_event =  {
                    "type": "progress",
                    "progress": progress_percentage *2 ,
                    "message": f"職業 {p.name} - 先攻方完成 {battle_num}/{num_battles // 2} 場"
                }
                print(f"Yielding progress: {progress_event}")  # 新增
                yield progress_event

        # 測試後攻
        for battle_num in range(1, num_battles // 2 + 1):
            if stop_training_flag.is_set():
                yield {"type": "stopped", "message": "ELO 計算已被終止。"}
                return

            env = BattleEnv(make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()

            while not done:
                enemy_act = random.choice([0, 1, 2])  # 隨機對手
                ai_act = trainer.compute_single_action(obs['enemy'], policy_id="shared_policy")
                actions = {"player": enemy_act, "enemy": ai_act}
                obs, rew, done_dict, tru, info = env.step(actions)
                done = done_dict["__all__"]

            # 先攻方 res=1 代表先攻(=player)贏；AI 是後攻 => AI 贏分數要反轉
            res = info["__common__"]["result"]
            if res == 1:
                score = 0
            elif res == -1:
                score = 1
            else:
                score = 0.5

            expected = 1 / (1 + 10 ** ((randomELO - elo_second) / 400))
            elo_second += K * (score - expected)

            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            if battle_num % 50 == 0:
                progress_event =  {
                    "type": "progress",
                    "progress": progress_percentage *2,
                    "message": f"職業 {p.name} - 後攻方完成 {battle_num}/{num_battles // 2} 場"
                }
                print(f"Yielding progress: {progress_event}")  # 新增
                yield progress_event

        total_elo = (elo_first + elo_second) / 2
        elo_results[p.name] = {
            "先攻方 ELO": round(elo_first),
            "後攻方 ELO": round(elo_second),
            "總和 ELO": round(total_elo)
        }
        print(f"Yielding final ELO for {p.name}: {elo_results[p.name]}")  # 新增
        yield {
            "type": "progress",
            "progress": progress_percentage*2,
            "message": f"職業 {p.name} 的 ELO 計算完成。"
        }

        total_first += elo_first
        total_second += elo_second

    overall_total = len(professions)
    average_first = total_first / overall_total
    average_second = total_second / overall_total
    average_total = (average_first + average_second) / 2

    print(f"\n=== ELO 結果===")
    print(f"{'職業':<15} | {'先攻方 ELO':<15} | {'後攻方 ELO':<15} | {'總和 ELO':<10}")
    print("-" * 60)
    for prof, elos in elo_results.items():
        print(f"{prof:<15} | {elos['先攻方 ELO']:<15} | {elos['後攻方 ELO']:<15} | {elos['總和 ELO']:<10}")
    print("-" * 60)
    print(f"{'平均':<15} | {round(average_first):<15} | {round(average_second):<15} | {round(average_total):<10}")

    print("\nELO 計算完成。")
    
    print("Final ELO Results:", elo_results)  # 已有的 print，確認最終結果
    
    yield {
        "type": "final_result",
        "elo_result": {
            "平均 ELO": round(average_total),
            "詳細": elo_results,
            "總和先攻方ELO": round(average_first),
            "總和後攻方ELO": round(average_second),
            "總和ELO": round(average_total),
        }
    }





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