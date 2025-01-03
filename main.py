# main.py

import sys
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

from battle_env import BattleEnv
from skills import SkillManager, sm
from professions import (
    Paladin, Mage, Assassin, Archer, Berserker, DragonGod, BloodGod,
    SteadfastWarrior, SunWarrior, Ranger, ElementalMage, HuangShen,
    GodOfStar
)
from train_methods import train_iteratively, cross_evaluation, compute_win_rate_table
from status_effects import StatusEffect, DamageMultiplier
from effect_mapping import EFFECT_MAPPING, EFFECT_VECTOR_LENGTH

from tabulate import tabulate


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
        SunWarrior(),
        Ranger(),
        ElementalMage(),
        HuangShen(),
        GodOfStar()
    ]


def show_profession_info(profession_list, skill_mgr):
    print("\n========== 職業介紹 ==========")
    for p in profession_list:
        print(f"{p.name} (HP={p.base_hp})")
        print(f"攻擊係數{p.baseAtk} 防禦係數{p.baseDef}")
        print(f"  被動: {p.passive_desc}")
        # 列出技能
        skill_ids = p.get_available_skill_ids()
        for sid in skill_ids:
            desc = skill_mgr.get_skill_desc(sid)
            # turns sid to skill 
            skill = skill_mgr.skills[sid]
            print(f"    技能 {skill.name} => {desc}")
        print("----------------------------------")
    input("按Enter返回主選單...")


def build_team_by_user(size, profession_list):
    team = []
    for i in range(size):
        while True:
            print("\n可選職業: (0)隨機")
            for idx, pf in enumerate(profession_list, start=1):
                print(f" ({idx}) {pf.name}")
            sel = input(f"隊伍成員{i} 職業選擇: ")
            try:
                s = int(sel)
                if s == 0:
                    pr = random.choice(profession_list)
                    break
                elif 1 <= s <= len(profession_list):
                    pr = profession_list[s-1]
                    break
            except:
                pass
            print("輸入錯誤, 請重新選擇.")
        team.append({
            "profession": pr,
            "hp": pr.base_hp,
            "max_hp": pr.max_hp,
            "status": {},
            "skip_turn": False,
            "is_defending": False,
            "times_healed": 0,
            "next_heal_double": False,
            "damage_multiplier": 1.0,
            "defend_multiplier": 1.0,
            "heal_multiplier": 1.0,
            "battle_log": [],
            "cooldowns": {},
        })
    return team


def build_random_team(size, profession_list):
    team = []
    for i in range(size):
        pr = random.choice(profession_list)
        team.append({
            "profession": pr,
            "hp": pr.base_hp,
            "max_hp": pr.max_hp,
            "status": {},
            "skip_turn": False,
            "is_defending": False,
            "times_healed": 0,
            "next_heal_double": False,
            "damage_multiplier": 1.0,
            "defend_multiplier": 1.0,
            "heal_multiplier": 1.0,
            "battle_log": [],
            "cooldowns": {},

        })
    return team


def ai_vs_ai(model_path_player, model_path_enemy, skill_mgr, professions):
    """
    AI vs AI 對戰模式
    """
    try:
        model_player = PPO.load(model_path_player)
    except FileNotFoundError:
        print(f"Player 模型檔案 {model_path_player} 未找到。請先訓練模型。")
        input("按Enter返回主選單...")
        return

    try:
        model_enemy = PPO.load(model_path_enemy)
    except FileNotFoundError:
        print(f"Enemy 模型檔案 {model_path_enemy} 未找到。請先訓練模型。")
        input("按Enter返回主選單...")
        return

    while True:
        print("\nAI vs AI 對戰模式:")
        print("1) 1v1")
        print("0) 返回主選單")
        c = input("請輸入選項: ")
        if c == "0":
            return
        if c != "1":
            print("無效選項，請重新選擇。")
            continue
        ts = 1  # 固定為1

        print("選擇我方隊伍:")
        p_team = build_team_by_user(ts, professions)
        print("選擇敵方隊伍:")
        e_team = build_team_by_user(ts, professions)

        env = BattleEnv(
            team_size=ts,
            enemy_team_size=ts,
            max_rounds=30,
            player_team=p_team,
            enemy_team=e_team,
            skill_mgr=skill_mgr,
            show_battle_log=True
        )

        obs = env.reset()
        done = False
        while not done:
            # 我方AI選擇動作
            action_player, _ = model_player.predict(obs.reshape(1, -1), deterministic=True)
            # 敵方AI選擇動作
            # 將觀察重新排列以符合敵方模型的觀察空間
            # [P1_HP, P1_prof_id, E1_HP, E1_prof_id, round_count]
            # 轉換為 [E1_HP, E1_prof_id, P1_HP, P1_prof_id, round_count]
            obs_enemy = obs[[2, 3, 0, 1, 4]]
            action_enemy, _ = model_enemy.predict(obs_enemy.reshape(1, -1), deterministic=True)
            # 合併動作
            combined_action = np.concatenate([action_player.flatten(), action_enemy.flatten()])
            # 環境執行步驟
            obs, reward, done, _ = env.step(combined_action)
            if done:
                if reward > 0:
                    print("我方贏了!")
                elif reward < 0:
                    print("敵方贏了!")
                else:
                    print("平手或回合耗盡")
                break

    return


def ai_vs_player(model_path_player, model_path_enemy, skill_mgr, professions):
    """
    AI vs 玩家 對戰模式
    - AI控制敵方
    - 玩家控制我方
    """
    try:
        model_enemy = PPO.load(model_path_enemy)
    except FileNotFoundError:
        print(f"Enemy 模型檔案 {model_path_enemy} 未找到。請先訓練模型。")
        input("按Enter返回主選單...")
        return

    # 玩家選擇隊伍
    print("選擇你的隊伍:")
    p_team = build_team_by_user(1, professions)  # 固定為1
    print("選擇敵方隊伍:")
    e_team = build_team_by_user(1, professions)  # 固定為1

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
        # 玩家選擇技能
        user = env.player_team[0]
        if user["hp"] <= 0:
            break

        avails = user["profession"].get_available_skill_ids()
        print("\n你的技能:")
        for sid in avails:
            d = skill_mgr.get_skill_desc(sid)
            print(f" {sid}) {d}")

        chosen = None
        while True:
            c = input("請選擇技能ID: ")
            try:
                cint = int(c)
                if cint in avails:
                    chosen = cint
                    break
            except:
                pass
            print("無效技能ID，請重新選擇.")

        # 準備動作
        action_player = np.array([chosen], dtype=np.int32)

        # AI選擇技能
        if user["hp"] > 0:
            action_enemy, _ = model_enemy.predict(obs.reshape(1, -1), deterministic=True)
        else:
            action_enemy = np.array([0], dtype=np.int32)  # 隨便一個數值，因為已經死亡

        # 合併動作
        combined_action = np.concatenate([action_player.flatten(), action_enemy.flatten()])
        # 執行步驟
        obs, rew, done, _ = env.step(combined_action)

    # 判斷結果
    if rew > 0:
        print("你贏了!")
    elif rew < 0:
        print("你輸了QQ")
    else:
        print("平手或回合耗盡")


def player_vs_random_computer(skill_mgr, professions):
    """
    玩家 VS 電腦（電腦隨機選擇技能）對戰模式
    """
    # 玩家選擇隊伍
    print("選擇你的隊伍:")
    p_team = build_team_by_user(1, professions)  # 固定為1

    # 電腦隨機選擇隊伍
    e_team = build_random_team(1, professions)

    env = BattleEnv(
        team_size=1,
        enemy_team_size=1,
        max_rounds=30,
        player_team=p_team,
        enemy_team=e_team,
        skill_mgr=skill_mgr,
        show_battle_log=True  # 顯示戰鬥日誌
    )

    obs = env.reset()
    done = False

    while not done:
        # 玩家選擇技能
        user = env.player_team[0]
        if user["hp"] <= 0:
            break

        avails = user["profession"].get_available_skill_ids()
        print("\n你的技能:")
        for sid in avails:
            d = skill_mgr.get_skill_desc(sid)
            print(f" {sid}) {d}")

        chosen = None

        while True:
            c = input("請選擇技能ID: ")
            try:
                cint = int(c)
                if cint in avails:
                    chosen = cint
                    break
            except:
                pass
            print("無效技能ID，請重新選擇.")

        # 準備動作
        action_player = np.array([chosen], dtype=np.int32)

        # 電腦隨機選擇技能
        enemy = env.enemy_team[0]
        if enemy["hp"] > 0:
            enemy_avails = enemy["profession"].get_available_skill_ids()
            chosen_enemy = random.choice(enemy_avails)
            action_enemy = np.array([chosen_enemy], dtype=np.int32)
            print(f"電腦選擇了技能ID {chosen_enemy} ({skill_mgr.get_skill_name(chosen_enemy)})")
        else:
            action_enemy = np.array([0], dtype=np.int32)  # 隨便一個數值，因為已經死亡

        # 合併動作
        combined_action = np.concatenate([action_player.flatten(), action_enemy.flatten()])
        # 執行步驟
        obs, rew, done, _ = env.step(combined_action)

    # 判斷結果
    player_hp = sum([m["hp"] for m in env.player_team])
    enemy_hp = sum([e["hp"] for e in env.enemy_team])
    if player_hp > enemy_hp:
        print("你贏了!")
    elif player_hp < enemy_hp:
        print("你輸了QQ")
    else:
        print("平手或回合耗盡")


def computer_vs_computer(skill_mgr, professions):
    """
    電腦 VS 電腦 對戰模式
    - 兩邊的電腦都自動選擇技能，技能ID範圍為0到2
    """
    ts = 1  # 隊伍大小固定為1

    # 隨機建立我方和敵方隊伍
    p_team = build_random_team(ts, professions)
    e_team = build_random_team(ts, professions)

    env = BattleEnv(
        team_size=ts,
        enemy_team_size=ts,
        max_rounds=30,
        player_team=p_team,
        enemy_team=e_team,
        skill_mgr=skill_mgr,
        show_battle_log=True
    )

    obs = env.reset()
    done = False
    round_counter = 1

    while not done:
        print(f"\n=== 第 {round_counter} 回合 ===")
        round_counter += 1

        # 我方電腦選擇技能 (0-2)
        player = env.player_team[0]
        if player["hp"] > 0:
            available_skills_p = player["profession"].get_available_skill_ids()
            if available_skills_p:
                action_player = np.array([random.choice(available_skills_p)], dtype=np.int32)
                skill_name_p = skill_mgr.get_skill_name(action_player[0]) if action_player[0] in skill_mgr.skills else "未知技能"
                print(f"我方電腦選擇了技能ID {action_player[0]} ({skill_name_p})")
            else:
                action_player = np.array([0], dtype=np.int32)  # 隨便一個數值，因為沒有可用技能
                print(f"我方電腦沒有可用技能，跳過行動。")
        else:
            action_player = np.array([0], dtype=np.int32)  # 任意數值，因為已經死亡

        # 敵方電腦選擇技能 (0-2)
        enemy = env.enemy_team[0]
        if enemy["hp"] > 0:
            available_skills_e = enemy["profession"].get_available_skill_ids()
            if available_skills_e:
                action_enemy = np.array([random.choice(available_skills_e)], dtype=np.int32)
                skill_name_e = skill_mgr.get_skill_name(action_enemy[0]) if action_enemy[0] in skill_mgr.skills else "未知技能"
                print(f"敵方電腦選擇了技能ID {action_enemy[0]} ({skill_name_e})")
            else:
                action_enemy = np.array([0], dtype=np.int32)  # 隨便一個數值，因為沒有可用技能
                print(f"敵方電腦沒有可用技能，跳過行動。")
        else:
            action_enemy = np.array([0], dtype=np.int32)  # 任意數值，因為已經死亡

        # 合併動作
        combined_action = np.concatenate([action_player.flatten(), action_enemy.flatten()])
        # 執行步驟
        obs, rew, done, _ = env.step(combined_action)

    # 判斷結果
    if rew > 0:
        print("\n=== 對戰結果 ===")
        print("我方電腦贏了!")
    elif rew < 0:
        print("\n=== 對戰結果 ===")
        print("敵方電腦贏了!")
    else:
        print("\n=== 對戰結果 ===")
        print("平手或回合耗盡")


def professions_fight_each_other(skill_mgr, professions, num_battles=100):
    """
    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率
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
                team_p = build_random_team(1, [p])
                team_e = build_random_team(1, [op])

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
                    # 我方隨機選擇技能
                    player = env.player_team[0]
                    if player["hp"] > 0:
                        available_skills_p = player["profession"].get_available_skill_ids()
                        if available_skills_p:
                            chosen_p = random.choice(available_skills_p)
                        else:
                            chosen_p = 0  # 任意技能ID，因為沒有可用技能
                    else:
                        chosen_p = 0  # 任意技能ID，因為已經死亡

                    # 敵方隨機選擇技能
                    enemy = env.enemy_team[0]
                    if enemy["hp"] > 0:
                        available_skills_e = enemy["profession"].get_available_skill_ids()
                        if available_skills_e:
                            chosen_e = random.choice(available_skills_e)
                        else:
                            chosen_e = 0  # 任意技能ID，因為沒有可用技能
                    else:
                        chosen_e = 0  # 任意技能ID，因為已經死亡

                    # 合併動作
                    combined_action = np.concatenate([np.array([chosen_p]), np.array([chosen_e])])
                    # 執行步驟
                    obs, rew, done, _ = env.step(combined_action)

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
            total = results[p.name][op.name]['win'] + results[p.name][op.name]['loss'] + results[p.name][op.name]['draw']
            win_rate = (results[p.name][op.name]['win'] / total) * 100 if total > 0 else 0
            win_rate_table[p.name][op.name] = f"{win_rate:.2f}%"

    # 顯示結果表
    print("\n=== 每個職業相互對戰100場的勝率 ===")
    headers = ["職業"] + [op.name for op in professions]
    table = []
    for p in professions:
        row = [p.name]
        for op in professions:
            if p == op:
                row.append("-")
            else:
                row.append(win_rate_table[p.name][op.name])
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))

    input("對戰完成。按Enter返回主選單...")

    return


def main():
    skill_mgr = build_skill_manager()
    professions = build_professions()

    # 模型檔案路徑(統一寫在這)
    default_model_path_player = "ppo20_iter_iter.zip"


    while True:
        print("\n=== 回合制戰鬥系統 ===")
        print("1) 顯示職業介紹")
        print("2) 迭代訓練 (vs AI)")
        print("3) 交叉勝率評估 (1v1)")
        print("4) AI vs AI 對戰")
        print("5) AI vs 玩家 對戰")
        print("6) 玩家 VS 電腦 (電腦隨機選擇技能)")
        print("7) 電腦 VS 電腦 對戰")  # 已有選項
        print("8) 每個職業相互對戰100場並計算勝率")  # 新增選項
        print("0) 離開")

        choice = input("請輸入選項: ").strip()
        if choice == "0":
            print("再見!")
            sys.exit(0)
        elif choice == "1":
            show_profession_info(professions, skill_mgr)
        elif choice == "2":
            print("請輸入迭代次數(預設3): ", end="")
            try:
                n = int(input())
            except:
                n = 3
            print("請輸入player模型存檔前綴(預設ppo_player): ", end="")
            sp_player = input().strip() or "ppo_player"

            max_episodes = 10  # 調整為合理的步數
            train_iteratively(n, max_episodes, skill_mgr, professions, sp_player, desired_total_steps=2)
            print("訓練完成.")
        elif choice == "3":
            print(f"使用player模型路徑: {default_model_path_player}")

            # 載入模型
            try:
                model_player = PPO.load(default_model_path_player)
            except FileNotFoundError:
                print(f"模型檔案 {default_model_path_player} 未找到。請先訓練模型。")
                input("按Enter返回主選單...")
                continue
            # 進行交叉勝率評估
            wins, losses, draws = cross_evaluation(
                model_player,  # 傳遞模型對象
                skill_mgr,
                professions,
                n_eval_episodes=25  # 每對職業對戰25次
            )
            # 計算勝率表
            win_rate_table = compute_win_rate_table(wins, losses, draws, professions)
            print("=== 交叉勝率評估結果 ===")
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
            input("按Enter返回主選單...")
        elif choice == "4":
            ai_vs_ai(default_model_path_player, default_model_path_player, skill_mgr, professions)
        elif choice == "5":
            ai_vs_player(default_model_path_player, default_model_path_player, skill_mgr, professions)
        elif choice == "6":
            player_vs_random_computer(skill_mgr, professions)
        elif choice == "7":
            computer_vs_computer(skill_mgr, professions)  # 呼叫已有的對戰模式函數
        elif choice == "8":
            professions_fight_each_other(skill_mgr, professions, num_battles=100)  # 呼叫新增的對戰模式函數
        else:
            print("無效選項，請重新輸入。")


def professions_fight_each_other(skill_mgr, professions, num_battles=100):
    """
    每個職業相互對戰100場（使用隨機選擇技能），並計算勝率
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
                team_p = build_random_team(1, [p])
                team_e = build_random_team(1, [op])

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
                    # 我方隨機選擇技能
                    player = env.player_team[0]
                    if player["hp"] > 0:
                        available_skills_p = player["profession"].get_available_skill_ids()
                        if available_skills_p:
                            chosen_p = random.choice(available_skills_p)
                        else:
                            chosen_p = 0  # 任意技能ID，因為沒有可用技能
                    else:
                        chosen_p = 0  # 任意技能ID，因為已經死亡

                    # 敵方隨機選擇技能
                    enemy = env.enemy_team[0]
                    if enemy["hp"] > 0:
                        available_skills_e = enemy["profession"].get_available_skill_ids()
                        if available_skills_e:
                            chosen_e = random.choice(available_skills_e)
                        else:
                            chosen_e = 0  # 任意技能ID，因為沒有可用技能
                    else:
                        chosen_e = 0  # 任意技能ID，因為已經死亡

                    # 合併動作
                    combined_action = np.concatenate([np.array([chosen_p]), np.array([chosen_e])])
                    # 執行步驟
                    obs, rew, done, _ = env.step(combined_action)

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
            total = results[p.name][op.name]['win'] + results[p.name][op.name]['loss'] + results[p.name][op.name]['draw']
            win_rate = (results[p.name][op.name]['win'] / total) * 100 if total > 0 else 0
            win_rate_table[p.name][op.name] = f"{win_rate:.2f}%"

    # 顯示結果表
    print("\n=== 每個職業相互對戰100場的勝率 ===")
    headers = ["職業"] + [op.name for op in professions]
    table = []
    for p in professions:
        row = [p.name]
        for op in professions:
            if p == op:
                row.append("-")
            else:
                row.append(win_rate_table[p.name][op.name])
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))

    input("對戰完成。按Enter返回主選單...")

    return


if __name__ == "__main__":
    main()
