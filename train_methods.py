# train_epoch.py
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from battle_env import BattleEnv

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

    # 初始化 2 個 PPO 模型（注意：若你想共享權重，也可以只用一個 model）
    # 這裡示範最簡單的 DummyVecEnv 包起來，但因為我們是自訂 step => 會相對複雜
    # 下面只是一種示範，你也可以手動寫 rollouts
    def make_env():
        p_team = [{
            "profession": random.choice(professions),
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
        }]
        e_team = [{
            "profession": random.choice(professions),
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
        }]
        return BattleEnv(
            team_size=1,
            enemy_team_size=1,
            max_rounds=30,
            player_team=p_team,
            enemy_team=e_team,
            skill_mgr=skill_mgr,
            show_battle_log=False  # 訓練時可關閉
        )

    # Dummy 環境 (SB3 需要一個可 call 的 function array)
    venv_1 = DummyVecEnv([make_env])
    venv_2 = DummyVecEnv([make_env])

    # 初始化兩個 PPO
    model1 = PPO("MlpPolicy", venv_1, verbose=0)
    model2 = PPO("MlpPolicy", venv_2, verbose=0)

    # 這裡只是示範 loop，實際你應該用更完整的rollout收集
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        # 每一個 iteration 我們做若干 episodes
        n_episodes = 5  # 你可以調整
        for ep in range(n_episodes):
            env = make_env()
            obs = env.reset()
            done = False
            while not done:
                # 分別取得 obs (player_obs, opponent_obs)
                player_obs = obs["player_obs"]
                oppo_obs = obs["opponent_obs"]
                p_mask = obs["player_action_mask"]
                e_mask = obs["opponent_action_mask"]

                # model1 -> player_action
                p_action, _ = model1.predict(player_obs, deterministic=False)
                # 如果該動作 mask=0 => 換一個
                if p_mask[p_action] == 0:
                    valid_as = np.where(p_mask==1)[0]
                    p_action = np.random.choice(valid_as) if len(valid_as)>0 else 0

                # model2 -> opponent_action
                e_action, _ = model2.predict(oppo_obs, deterministic=False)
                if e_mask[e_action] == 0:
                    valid_as = np.where(e_mask==1)[0]
                    e_action = np.random.choice(valid_as) if len(valid_as)>0 else 0

                actions = {
                    "player_action": p_action,
                    "opponent_action": e_action
                }
                next_obs, rewards, done, info = env.step(actions)

                # 取出 player_reward / opponent_reward
                r1 = rewards["player_reward"]
                r2 = rewards["opponent_reward"]

                # ... 這裡若你要把資料送進 replay buffer，就手動實作(因 SB3預設是一個 agent)
                # 這個範例就簡化不做 buffer, 直接結束 ...

                obs = next_obs

        # 結束 ep 後，針對 model1, model2 各自做 learn() (此範例為教學示意)
        # 一般來說，你需要先收集 Rollouts 到 buffer，再來做 model.learn()
        # 這裡僅示範每個 iteration 結束就隨意 train N timesteps
        model1.learn(total_timesteps=100)
        model2.learn(total_timesteps=100)

        # 可測試一下互打情況...
        print("  => 訓練中暫不做任何評估")

    # 訓練完後存檔
    model1.save(save_path_1)
    model2.save(save_path_2)
    print(f"訓練完成, 已儲存到 {save_path_1}, {save_path_2}")




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

    total_combinations = len(professions) * (len(professions) - 1)
    current_combination = 0

    for p in professions:
        for op in professions:
            if p == op:
                continue
            current_combination += 1
            print(f"\n對戰 {current_combination}/{total_combinations}: {p.name} VS {op.name}")

            for battle_num in range(1, num_battles + 1):
                # 建立兩隊隊伍，分別是p和op
                team_p  = [{
                "profession": random.choice(professions),
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
                }]
                
                team_e = [{
                "profession": random.choice(professions),
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
                }]

                # 初始化BattleEnv，關閉battle_log以加快速度
                env = BattleEnv(
                    team_size=1,
                    enemy_team_size=1,
                    max_rounds=30,
                    player_team=team_p,
                    enemy_team=team_e,
                    skill_mgr=skill_mgr,
                    show_battle_log=False  # 不顯示戰鬥日誌
                )
                obs = env.reset()
                done = False

                while not done:
                    pmask = obs["player_action_mask"]
                    emask = obs["opponent_action_mask"]
                    p_actions = np.where(pmask==1)[0]
                    e_actions = np.where(emask==1)[0]
                    p_act = random.choice(p_actions) if len(p_actions)>0 else 0
                    e_act = random.choice(e_actions) if len(e_actions)>0 else 0
                    obs, rew, done, info = env.step({
                        "player_action": p_act,
                        "opponent_action": e_act
                    })
                rew = rew["player_reward"]
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
    """
    try:
        model1 = PPO.load(model_path_1)
        model2 = PPO.load(model_path_2)
    except:
        print("模型載入失敗，請先確保已訓練並存檔。")
        input("按Enter返回主選單...")
        return

    print("=== 高段環境測試 (AI vs AI) ===")
    wins = 0
    losses = 0
    draws = 0
    for i in range(num_battles):
        p_team = [{
            "profession": random.choice(professions),
            "hp": 0, "max_hp": 0,
            "status": {}, "skip_turn": False,
            "is_defending": False,
            "damage_multiplier":1.0,
            "defend_multiplier":1.0,
            "heal_multiplier":1.0,
            "battle_log":[],
            "cooldowns": {}
        }]
        e_team = [{
            "profession": random.choice(professions),
            "hp": 0, "max_hp": 0,
            "status": {}, "skip_turn": False,
            "is_defending": False,
            "damage_multiplier":1.0,
            "defend_multiplier":1.0,
            "heal_multiplier":1.0,
            "battle_log":[],
            "cooldowns": {}
        }]
        env = BattleEnv(1,1,30,p_team,e_team,skill_mgr,show_battle_log=False)
        obs = env.reset()
        done = False
        while not done:
            p_obs = obs["player_obs"]
            p_mask = obs["player_action_mask"]
            e_obs = obs["opponent_obs"]
            e_mask = obs["opponent_action_mask"]

            # AI1
            p_act, _ = model1.predict(p_obs, deterministic=False)
            if p_mask[p_act]==0:
                valid_as = np.where(p_mask==1)[0]
                p_act = random.choice(valid_as) if len(valid_as)>0 else 0
            # AI2
            e_act, _ = model2.predict(e_obs, deterministic=False)
            if e_mask[e_act]==0:
                valid_as = np.where(e_mask==1)[0]
                e_act = random.choice(valid_as) if len(valid_as)>0 else 0

            obs, rewards, done, info = env.step({
                "player_action": p_act,
                "opponent_action": e_act
            })
        final_r = rewards["player_reward"]
        if final_r>0:
            wins+=1
        elif final_r<0:
            losses+=1
        else:
            draws+=1
    print(f"AI vs AI ({num_battles}場): AI1贏{wins}、AI2贏{losses}、平{draws}")
    input("按Enter返回主選單...")

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
    opp_ELO = 1000

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
