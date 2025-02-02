
import os, json, random, copy
from flask import Response, stream_with_context, jsonify, request


from .train_methods import make_env_config, build_professions, build_skill_manager
from .battle_env import BattleEnv
from ray.rllib.algorithms.ppo import PPOConfig


def mvmg_run_single_battle(left_trainer, right_trainer, pr1, pr2,professions,skill_mgr):
    """
    進行單場模型間對戰，
    使用 model_path 作為兩邊的模型（因為兩邊的模型會一樣，只載入一次檢查點）。
    回傳結果：
      1 代表 model A 勝，
     -1 代表 model B 勝，
      0 代表平手或回合耗盡。
    """
    # 建立環境配置（可根據你的 make_env_config 實作調整）
    beconfig = make_env_config(
        skill_mgr=skill_mgr,
        professions=professions,
        show_battlelog=False,
        pr1=pr1,
        pr2=pr2
    )
    
    # 只需以 get_trainer 載入一次模型檢查點

    
    env = BattleEnv(config=beconfig)
    done = False
    obs, _ = env.reset()
    
    # 由同一個 trainer 分別取得對戰雙方的 policy 與初始 state
    l_policy = left_trainer.get_policy("shared_policy")
    r_policy = right_trainer.get_policy("shared_policy")
    state_l = l_policy.get_initial_state()
    state_r = r_policy.get_initial_state()
    
    while not done:
        act_A_pack = left_trainer.compute_single_action(obs['player'], state=state_l, policy_id="shared_policy")
        act_B_pack = right_trainer.compute_single_action(obs['enemy'], state=state_r, policy_id="shared_policy")
        action_A = act_A_pack[0]
        action_B = act_B_pack[0]
        state_l = act_A_pack[1]
        state_r = act_B_pack[1]
        actions = {"player": action_A, "enemy": action_B}
        obs, rew, done_dict, _, info = env.step(actions)
        done = done_dict["__all__"]
        
    result = info["__common__"]["result"]
    return result

def version_test_model_vs_model_generate_sse(professions, skill_mgr, num_battles, model_path_A, model_path_B):
    """
    模型間互相對戰產生：包含同職業內戰（intra）與交叉對戰（cross）
    透過 SSE 回傳目前進度與最終統計結果。

    注意：因為兩個模型實際上相同，所以只用 model_path_A 載入一次檢查點即可，
    model_path_B 參數將被忽略。
    """
    # 統計資料初始化
    intra_results = {p.name: {'win_modelA': 0, 'win_modelB': 0, 'draw': 0} for p in professions}
    cross_results = {}
    total_combinations = len(professions) * len(professions)  # 包含相同職業對戰
    current_combination = 0
    
    l_model = get_trainer(model_path_A)
    if model_path_A  ==  model_path_B:
        r_model = l_model
    else :
        r_model = get_trainer(model_path_B)
    
    pr = build_professions()
    m = build_skill_manager()

    for p in professions:
        for op in professions:
            current_combination += 1
            progress_percent = (current_combination / total_combinations) * 100
            yield {
                "type": "progress",
                "progress": progress_percent,
                "message": f"進行 {p.name} vs {op.name} 對戰..."
            }
            for battle_num in range(1, num_battles + 1):
                # 只使用 model_path_A 進行單場對戰（兩邊模型相同）
                if battle_num % 2 == 0:
                    result = mvmg_run_single_battle(l_model, r_model, p, op,pr,m)
                else:
                    result = mvmg_run_single_battle(r_model, l_model, op, p,pr,m)
                # 依照奇偶場次調整先後（若需要調整先後邏輯，請依需求修改）
                if battle_num % 2 == 1:
                    result = -result
                if p.name == op.name:
                    # 同職業內戰
                    if result == 1:
                        intra_results[p.name]['win_modelA'] += 1
                    elif result == -1:
                        intra_results[p.name]['win_modelB'] += 1
                    else:
                        intra_results[p.name]['draw'] += 1
                else:
                    # 交叉對戰
                    key = f"{p.name} vs {op.name}"
                    if key not in cross_results:
                        cross_results[key] = {'win_modelA': 0, 'win_modelB': 0, 'draw': 0}
                    if result == 1:
                        cross_results[key]['win_modelA'] += 1
                    elif result == -1:
                        cross_results[key]['win_modelB'] += 1
                    else:
                        cross_results[key]['draw'] += 1

    final_data = {
        "model_A": os.path.basename(model_path_A),
        "model_B": os.path.basename(model_path_B),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intra_results": intra_results,
        "cross_results": cross_results,
        "total_battles": num_battles * total_combinations
    }
    
    # save the final data to a file data/model_vs/yy_mm_dd_hh_mm_ss.json
    with open(f'data/model_vs/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.json', 'w') as f:
        json.dump(final_data, f)
        
    
    yield {
        "type": "final_result",
        "progress": 100,
        "message": "模型對戰產生完成",
        "data": final_data
    }
from datetime import datetime

# 以下為輔助函式，請依你原專案實作調整
def get_trainer(model_path, fc_hiddens_default=[256, 256, 256]):
    """
    根據模型路徑讀取 training_meta.json，並建立 PPOConfig 與 trainer，
    最後回傳 trainer。
    """
    fc_hiddens = fc_hiddens_default
    max_seq_len = 10
    meta_path = os.path.abspath(model_path)
    meta_path = os.path.join(model_path, "training_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
                hyperparams = meta_info.get("hyperparams", {})
                fc_hiddens = hyperparams.get("fcnet_hiddens", fc_hiddens_default)
                max_seq_len = hyperparams.get("max_seq_len", 10)
        except Exception as e:
            print(f"讀取 meta 時錯誤: {e}")
    # 建立環境與訓練器配置（請根據你原本的邏輯修改）
    beconfig = make_env_config(
        skill_mgr=build_skill_manager(),
        professions=build_professions(),
        show_battlelog=False
    )
    config = (
        PPOConfig()
        .environment(env=BattleEnv, env_config=beconfig)
        .env_runners(num_env_runners=1, sample_timeout_s=120)
        .framework("torch")
        .training(
            model={
                "custom_model": "my_mask_model",
                "fcnet_hiddens": fc_hiddens,
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": max_seq_len
            },
        )
    )
    benv = BattleEnv(config=beconfig)
    config = config.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs:
            "shared_policy" if agent_id == "player" else "shared_policy"
    )
    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    trainer = config.build()
    trainer.restore(model_path)
    return trainer

# 後端路由範例
from flask import Blueprint
main_routes = Blueprint("main_routes", __name__)

