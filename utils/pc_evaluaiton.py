from .battle_env import BattleEnv
from .metrics import calculate_combined_metrics, calculate_ehi, calculate_esi, calculate_mpi
from .data_stamp import Gdata
from .train_methods import make_env_config
import copy
import numpy as np
import random

def version_test_random_vs_random_sse(professions, skill_mgr, num_battles=100):
    """
    每個職業相互對戰 num_battles 場（使用隨機選擇技能），並計算勝率（不計入平局）。
    這裡改為生成器，使用 yield 回傳進度，用於 SSE。
    
    新增:
    - 紀錄每場對戰的回合數，最後計算平均回合數
    - 計算每個職業的平均勝率 (avg_win_rate)
    """
    print("\n[SSE] 開始進行每個職業相互對戰隨機對戰...")
    
    # 初始化結果字典
    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0} for op in professions if op != p} for p in professions}
    skillusedFreq = {p.name: {0:0,1:0,2:0,3:0} for p in professions}
    
    # 為了計算平均回合數
    total_rounds = 0
    total_matches = 0  # 職業對職業，單場對戰次數
    total_combinations = len(professions) * (len(professions) - 1)
    current_combination = 0
    
    for p in professions:
        for op in professions:
            if p == op:
                continue
            
            current_combination += 1
            # yield 進度 (以組合數量為基準)
            progress_percent = (current_combination / total_combinations) * 100
            yield {
                "type": "progress",
                "progress": progress_percent,
                "message": f"對戰組合 {p.name} VS {op.name} 進行中..."
            }
            
            # 開始對戰
            for battle_num in range(1, num_battles + 1):
                # 建立環境
                if battle_num % 2 == 0:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                    )
                else:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=op, pr2=p)
                    )

                done = False
                obs, _ = env.reset()
                
                rounds = 0  # 計算單場回合數
                while not done:
                    rounds += 1
                    pmask = obs["player"][0:4]
                    emask = obs["enemy"][0:4]
                    p_actions = np.where(pmask == 1)[0]
                    e_actions = np.where(emask == 1)[0]
                    p_act = random.choice(p_actions) if len(p_actions) > 0 else 0
                    e_act = random.choice(e_actions) if len(e_actions) > 0 else 0
                    
                    # 技能使用次數紀錄
                    # 注意：p.name 與 op.name 的使用順序要看哪邊是 "player"/"enemy"
                    # 這邊的版本_test_random_vs_random 以 p 為 "player"，op 為 "enemy"（或反之）
                    # 因此依此更新
                    if battle_num % 2 == 1:
                        # 先攻 p = enemy
                        skillusedFreq[op.name][e_act] += 1
                        skillusedFreq[p.name][p_act] += 1
                    else:
                        # 先攻 p = player
                        skillusedFreq[p.name][p_act] += 1
                        skillusedFreq[op.name][e_act] += 1

                    obs, rew, done_dict, tru, info = env.step({
                        "player": p_act,
                        "enemy": e_act
                    })
                    
                    done = done_dict["__all__"]

                total_rounds += rounds
                total_matches += 1
                
                # 獲取結果
                info_result = info["__common__"]["result"]
                
                # battle_num 奇偶數控制先後攻，所以要反轉結果:
                if battle_num % 2 == 1:
                    # 奇數場： pr1=op, pr2=p
                    # 先攻是 op => info_result=1 => op贏 (對 p 而言就是輸)
                    # 對 p 來說，要把結果反向
                    info_result = -info_result

                # 判斷勝負
                if info_result == 1:
                    results[p.name][op.name]['win'] += 1
                elif info_result == -1:
                    results[p.name][op.name]['loss'] += 1
                else:
                    results[p.name][op.name]['draw'] += 1
    
    # 全部對戰結束後，計算每個職業的勝率
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
            total = wins + losses  # 平局不計入
            w_rate = (wins / total) * 100 if total > 0 else 0
            win_rate_table[p.name][op.name] = {
                'win': wins,
                'loss': losses,
                'draw': draws,
                'win_rate': w_rate
            }
    
    # 產生 combine_rate_table (此處若仍需做後續指標計算，可與原程式相同)
    combine_rate_table = copy.deepcopy(win_rate_table)
    # (可加上你原先 calculate_combined_metrics, EHI, ESI, MPI 計算)
    metrics = calculate_combined_metrics(combine_rate_table)
    ehi = calculate_ehi(combine_rate_table)
    esi = calculate_esi(combine_rate_table, calculate_combined_metrics)
    mpi = calculate_mpi(combine_rate_table, calculate_combined_metrics)
    
    # ============ 新增：平均回合數 ================
    avg_rounds = total_rounds / total_matches if total_matches > 0 else 0
    
    # ============ 新增：職業平均勝率(顯示在 Profession Evaluation) =========
    # 每個職業對所有對手的 wins, losses 統計
    profession_avg_win_rate = {}
    for player_prof, vs_dict in results.items():
        total_wins = 0
        total_losses = 0
        for opponent_prof, record in vs_dict.items():
            total_wins += record['win']
            total_losses += record['loss']
        match_count = total_wins + total_losses
        avg_wr = (total_wins / match_count) if match_count > 0 else 0
        profession_avg_win_rate[player_prof] = avg_wr  # 0~1
    
    # 把職業平均勝率加進 metrics (Profession Evaluation) 裡
    # 假設 metrics[profName] = { "EIR": x, "Advanced NAS": y, "MSI": z, "PI20": w }
    for prof in metrics.keys():
        metrics[prof]["AVG_WR"] = profession_avg_win_rate.get(prof, 0) * 100

    # 結果封裝
    res = {
        "env_evaluation": {
            "ehi": ehi,
            "esi": esi,
            "mpi": mpi,
            "avg_rounds": avg_rounds,  # 新增
        },
        "profession_evaluation": metrics,   # 包含了 "AVG_WR"
        "combine_win_rate_table": combine_rate_table,
        "profession_skill_used_freq": skillusedFreq,
        "total_battles": int(num_battles*len(professions)*(len(professions)-1))
    }
    from .global_var import globalVar as gv
    r = Gdata(res,gv['version'],"cross_validation_pc")  # 這裡是你的自訂函數，用來儲存結果
    r.save()
    
    # 最後整包結束再 yield 一次
    yield {
        "type": "final_result",
        "progress": 100,
        "message": "對戰產生完成",
        "data": res
    }
