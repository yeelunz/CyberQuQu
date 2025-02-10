import os
import json
import random
import time
from .battle_env import BattleEnv
from .metrics import calculate_combined_metrics, calculate_ehi, calculate_esi, calculate_mpi
from ray.rllib.algorithms.ppo import PPOConfig
from .train_methods import make_env_config
from .train_methods import stop_training_flag
from .data_stamp import Gdata
import copy

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

    t1 = time.time()
    fc_hiddens = [256, 256, 256]
    check_point_path = os.path.abspath(model_path_1)

    if os.path.exists(check_point_path):
        # load check_point_path/training_meta.json
        meta_path = os.path.join(check_point_path, "training_meta.json")
        # get fcnet_hiddens from json
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
                hyperparams = meta_info.get("hyperparams", {})
                fc_hiddens = hyperparams.get("fcnet_hiddens", [256, 256, 256])
                max_seq_len = hyperparams.get("max_seq_len", 10)
                mask_model = hyperparams.get("mask_model", "my_mask_model")
        except FileNotFoundError:
            print(f"找不到 {meta_path}，將使用預設的 fcnet_hiddens。")
    else:
        print(f"mata data 路徑 {check_point_path} 不存在。")

    # 設定環境配置
    beconfig = make_env_config(
        skill_mgr=skill_mgr, professions=professions, show_battlelog=False)
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
                "custom_model": mask_model,
                "fcnet_hiddens": fc_hiddens,
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": max_seq_len
            },
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
    t2 = time.time()

    config.api_stack(enable_rl_module_and_learner=False,
                     enable_env_runner_and_connector_v2=False)
    # 建立訓練器並載入檢查點
    check_point_path = os.path.abspath(model_path_1)
    # 注意：這裡只留一次 build
    trainer = config.build()

    if os.path.exists(check_point_path):
        trainer.restore(check_point_path)
    else:
        print(f"檢查點路徑 {check_point_path} 不存在。")

    t3 = time.time()

    print(f"建立訓練器耗時: {t2-t1:.2f}秒")
    print(f"載入檢查點耗時: {t3-t2:.2f}秒")

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

            env = BattleEnv(make_env_config(
                skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()
            policy = trainer.get_policy("shared_policy")
            state = policy.model.get_initial_state()

            while not done:
                ai_package = trainer.compute_single_action(
                    obs['player'], state=state, policy_id="shared_policy")
                ai_act = ai_package[0]
                state = ai_package[1]

                enemy_act = random.choice([0, 1, 2,3])  # 隨機對手
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
                progress_event = {
                    "type": "progress",
                    "progress": progress_percentage * 2,
                    "message": f"職業 {p.name} - 先攻方完成 {battle_num}/{num_battles // 2} 場"
                }
                print(f"Yielding progress: {progress_event}")  # 新增
                yield progress_event

        # 測試後攻
        for battle_num in range(1, num_battles // 2 + 1):
            if stop_training_flag.is_set():
                yield {"type": "stopped", "message": "ELO 計算已被終止。"}
                return

            env = BattleEnv(make_env_config(
                skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()
            policy = trainer.get_policy("shared_policy")
            state = policy.model.get_initial_state()

            while not done:
                enemy_act = random.choice([0, 1, 2,3])  # 隨機對手

                ai_package = trainer.compute_single_action(
                    obs['enemy'], state=state, policy_id="shared_policy")
                ai_act = ai_package[0]
                state = ai_package[1]

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
                progress_event = {
                    "type": "progress",
                    "progress": progress_percentage * 2,
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
        print(f"{prof:<15} | {elos['先攻方 ELO']:<15} | {
              elos['後攻方 ELO']:<15} | {elos['總和 ELO']:<10}")
    print("-" * 60)
    print(f"{'平均':<15} | {round(average_first):<15} | {
          round(average_second):<15} | {round(average_total):<10}")

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

def version_test_random_vs_random_sse_ai(professions, skill_mgr, num_battles=100, model_path_1="my_battle_ppo_checkpoints"):
    """
    每個職業相互對戰 num_battles 場（使用隨機選擇技能），並計算勝率（不計入平局）。
    這裡改為生成器，使用 yield 回傳進度，用於 SSE。

    新增:
    - 紀錄每場對戰的回合數，最後計算平均回合數
    - 計算每個職業的平均勝率 (avg_win_rate)
    """
    print("\n[SSE] 開始進行每個職業相互對戰隨機對戰...")

    # 初始化結果字典
    results = {p.name: {op.name: {'win': 0, 'loss': 0, 'draw': 0}
                        for op in professions if op != p} for p in professions}
    skillusedFreq = {p.name: {0: 0, 1: 0, 2: 0,3:0} for p in professions}

    fc_hiddens = [256, 256, 256]
    check_point_path = os.path.abspath(model_path_1)

    if os.path.exists(check_point_path):
        # load check_point_path/training_meta.json
        meta_path = os.path.join(check_point_path, "training_meta.json")
        # get fcnet_hiddens from json
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
                hyperparams = meta_info.get("hyperparams", {})
                fc_hiddens = hyperparams.get("fcnet_hiddens", [256, 256, 256])
                max_seq_len = hyperparams.get("max_seq_len", 10)
                mask_model = hyperparams.get("mask_model", "my_mask_model")
        except FileNotFoundError:
            print(f"找不到 {meta_path}，將使用預設的 fcnet_hiddens。")

    # 初始化 AI ELO
    beconfig = make_env_config(
        skill_mgr=skill_mgr, professions=professions, show_battlelog=True)
    config = (
        PPOConfig()
        .environment(
            env=BattleEnv,            # 指定我們剛剛定義的環境 class
            env_config=beconfig  # 傳入給 env 的一些自定設定
        )
        .env_runners(num_env_runners=1, sample_timeout_s=120)  # 可根據你的硬體調整
        .framework("torch")            # 或 "tf"
        .training(
            model={
                "custom_model": mask_model,
                "fcnet_hiddens": fc_hiddens,
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": max_seq_len
            },
        )
    )
    benv = BattleEnv(config=beconfig)
    config.api_stack(enable_rl_module_and_learner=False,
                     enable_env_runner_and_connector_v2=False)
    config = config.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs:
        "shared_policy" if agent_id == "player" else "shared_policy"
    )

    check_point_path = os.path.abspath(model_path_1)
    trainer = config.build()  # 用新的 API 构建训练器
    trainer.restore(check_point_path)

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
                        make_env_config(skill_mgr, professions,
                                        show_battlelog=False, pr1=p, pr2=op)
                    )
                else:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions,
                                        show_battlelog=False, pr1=op, pr2=p)
                    )

                done = False
                obs, _ = env.reset()
                policy = trainer.get_policy("shared_policy")
                state_1 = policy.model.get_initial_state()
                state_2 = policy.model.get_initial_state()

                rounds = 0  # 計算單場回合數
                while not done:
                    rounds += 1
                    if battle_num % 2 == 0:
                        p_act_pack = trainer.compute_single_action(
                            obs['player'], state=state_1, policy_id="shared_policy")
                        p_act = p_act_pack[0]
                        state_1 = p_act_pack[1]
                        # if p act in mask is 0, then choose random action
                        e_act_pack = trainer.compute_single_action(
                            obs['enemy'], state=state_2, policy_id="shared_policy")
                        e_act = e_act_pack[0]
                        state_2 = e_act_pack[1]
                    else:
                        e_act_pack = trainer.compute_single_action(
                            obs['enemy'], state=state_1, policy_id="shared_policy")
                        e_act = e_act_pack[0]
                        state_1 = e_act_pack[1]
                        # if p act in mask is 0, then choose random action
                        p_act_pack = trainer.compute_single_action(
                            obs['player'], state=state_2, policy_id="shared_policy")
                        p_act = p_act_pack[0]
                        state_2 = p_act_pack[1]

                    # 技能使用次數紀錄
                    # 注意：p.name 與 op.name 的使用順序要看哪邊是 "player"/"enemy"
                    # 這邊的版本_test_random_vs_random 以 p 為 "player"，op 為 "enemy"（或反之）
                    # 因此依此更新
                    if battle_num % 2 == 0:
                        # 先攻 p = player
                        skillusedFreq[p.name][p_act] += 1
                        skillusedFreq[op.name][e_act] += 1
                        
                    else:
                        # 先攻 p = enemy
                        skillusedFreq[op.name][e_act] += 1
                        skillusedFreq[p.name][p_act] += 1

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
    metrics = calculate_combined_metrics(combine_rate_table)
    ehi = calculate_ehi(combine_rate_table)
    esi = calculate_esi(combine_rate_table, calculate_combined_metrics)
    mpi = calculate_mpi(combine_rate_table, calculate_combined_metrics)

    avg_rounds = total_rounds / total_matches if total_matches > 0 else 0

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
    name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    r = Gdata(res, gv['version'], "cross_validation_ai",name=name,
              model=model_path_1)  # 這裡是你的自訂函s數，用來儲存結果
    r.save()
    # 最後整包結束再 yield 一次
    yield {
        "type": "final_result",
        "progress": 100,
        "message": "對戰產生完成",
        "data": res
    }

