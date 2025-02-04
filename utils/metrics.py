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