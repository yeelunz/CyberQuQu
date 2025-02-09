# train_methods.py

import numpy as np
import random
from ray.rllib.algorithms.ppo import PPOConfig
from .battle_env import BattleEnv
from .metrics import calculate_combined_metrics, calculate_ehi, calculate_esi, calculate_mpi
import os
from datetime import datetime
import json
from .global_var import globalVar as gl
from .data_stamp import Gdata
from .professions import build_professions
from .skills import build_skill_manager
import threading
import time
from .model_struct import MaskedLSTMNetwork, MaskedLSTMNetworkWithEmb, MaskedLSTMNetworkWithMergedEmb,MaskedLSTMNetworkWithEmbV2,MaskedLSTMNetworkWithMergedEmbV2,MaskedLSTMNetworkV2,TransformerMaskedNetworkV2,MaskedLSTMNetworkWithAttention,EnhancedMaskedLSTMNetwork,HRLMaskedLSTMNetwork,TransformerMaskedNetwork
from ray.rllib.models import ModelCatalog


# get







stop_training_flag = threading.Event()
ModelCatalog.register_custom_model("my_mask_model", MaskedLSTMNetwork)
ModelCatalog.register_custom_model("my_mask_model_v2", MaskedLSTMNetworkV2)
ModelCatalog.register_custom_model("my_mask_model_transformer_v2", TransformerMaskedNetworkV2)
ModelCatalog.register_custom_model("my_mask_model_enhanced", EnhancedMaskedLSTMNetwork)
ModelCatalog.register_custom_model("my_mask_model_hrl", HRLMaskedLSTMNetwork)
ModelCatalog.register_custom_model("my_mask_model_transformer", TransformerMaskedNetwork)
ModelCatalog.register_custom_model("my_mask_model_with_attention", MaskedLSTMNetworkWithAttention)
ModelCatalog.register_custom_model("my_mask_model_with_emb", MaskedLSTMNetworkWithEmb)
ModelCatalog.register_custom_model("my_mask_model_with_emb_v2", MaskedLSTMNetworkWithEmbV2)
ModelCatalog.register_custom_model("my_mask_model_with_emb_combined", MaskedLSTMNetworkWithMergedEmb)
ModelCatalog.register_custom_model("my_mask_model_with_emb_combined_v2", MaskedLSTMNetworkWithMergedEmbV2)

def multi_agent_cross_train(num_iterations,
                            model_name="my_multiagent_ai",
                            hyperparams=None):
    """
    多智能體交叉訓練
    """
    # 如果get不到就是240
    SAMPLE_TIME_OUT_S = gl.get("sample_time_out_s", 240)
    NUM_GPUS_PER_ENV_RUNNER = gl.get("num_gpus_per_env_runner", 0)

    professions = build_professions()
    skill_mgr = build_skill_manager()
    beconfig = make_env_config(skill_mgr, professions, train_mode=True)

    if hyperparams is None:
        hyperparams = {}

    # 以下依照前端輸入處理超參數的邏輯：
    # 1. Learning Rate 與 LR Schedule 互斥：若 learning_rate 有值，則 schedule 固定為 None
    lr = hyperparams.get("learning_rate")
    lr_schedule = hyperparams.get("lr_schedule")
    if lr is not None:
        lr_schedule = None
    else:
        lr = 5e-5  # 預設 learning rate

    # 2. Entropy Coefficient 與 Entropy Schedule 互斥
    entropy_coeff = hyperparams.get("entropy_coeff")
    entropy_coeff_schedule = hyperparams.get("entropy_coeff_schedule")
    if entropy_coeff is not None:
        entropy_coeff_schedule = None
    else:
        entropy_coeff = 0.0  # 預設 entropy coefficient

    # 3. Grad Clip 與 Grad Clip By
    grad_clip = hyperparams.get("grad_clip", None)
    if grad_clip is None:
        grad_clip_by = 'global_norm'  # 當 grad_clip 為 None 時，固定回傳預設值
    else:
        grad_clip_by = hyperparams.get("grad_clip_by", 'global_norm')

    # 修改 config.training() 部分，帶入所有超參數：
    
    config = (
        PPOConfig()
        .environment(
            env=BattleEnv,
            env_config=beconfig
        )
        .env_runners(
            num_env_runners=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=NUM_GPUS_PER_ENV_RUNNER,
            sample_timeout_s=SAMPLE_TIME_OUT_S
        )
        .framework("torch")
        .training(
            model={
                "custom_model": hyperparams.get("mask_model", "my_mask_model"),
                "fcnet_hiddens": hyperparams.get("fcnet_hiddens", [256, 256]),
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": hyperparams.get("max_seq_len", 10),  # 與模型一致
            },
            use_gae=True,
            gamma=hyperparams.get("gamma", 0.99),
            lr=lr,
            lr_schedule=lr_schedule,
            train_batch_size=hyperparams.get("train_batch_size", 4000),
            minibatch_size=hyperparams.get("minibatch_size", 128),
            entropy_coeff=entropy_coeff,
            entropy_coeff_schedule=entropy_coeff_schedule,
            grad_clip=grad_clip,
            grad_clip_by=grad_clip_by,
            lambda_=hyperparams.get("lambda", 1.0),
            clip_param=hyperparams.get("clip_param", 0.3),
            vf_clip_param=hyperparams.get("vf_clip_param", 10.0),
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
        print('result\n',result)

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
            "cost": round(num_iterations/5 * hyperparams.get("train_batch_size", 4000)/4000 * cal_size(hyperparams.get("fcnet_hiddens", [256, 256]))* gl['cost'][hyperparams.get("mask_model", "my_mask_model")],2),
            "hidden_info": False
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

    # 這邊是訓練完了 如果是 mask_model_with_emb_combined 就要把 embedding 存起來到embedding.json
    if hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb" \
        or hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb_combined"\
        or hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb_v2"\
        or hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb_combined_v2":
        print("Saving embeddings...")
        save_root = os.path.join("data", "saved_models", model_name)
        meta_path = os.path.join(save_root, "embeddings.json")
        model = algo.get_policy("shared_policy").model
        embeddings = model.get_all_embeddings()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

    # 重置停止標誌
    stop_training_flag.clear()



def make_env_config(skill_mgr, professions, show_battlelog=False, pr1=None, pr2=None, train_mode=False):
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



def cal_size(fc_net_hidden):
    """
    根據 fc_net_hidden 設定計算整個模型（前面的全連接層 + 最後的 LSTM cell）的參數數量，
    並以 fc_net_hidden = [256,256] 的模型作為基準 (比例=1)。
    
    假設：
      - 輸入維度固定為 256。
      - fc_net_hidden 列表長度 >= 1。
        * 若長度為 1：表示模型僅有一層 LSTM cell，
          輸入來自 256 維的外部輸入，隱藏單元數為 fc_net_hidden[0]。
        * 若長度 >= 2：表示前面所有層（除了最後一層）為全連接層，
          第一層接收 256 維輸入，其餘 fc 層依序連接，
          最後一層是 LSTM cell，其輸入維度來自倒數第二層，全連接層的輸出維度，
          隱藏單元數為 fc_net_hidden[-1].
    
    回傳：
      與基準模型參數數量的比例（float）。
    """
    # 固定輸入維度
    input_dim = 256

    if not fc_net_hidden or len(fc_net_hidden) < 1:
        raise ValueError("fc_net_hidden 必須至少包含一個元素")
    
    total_params = 0
    n = len(fc_net_hidden)
    
    if n == 1:
        # 模型只有 LSTM cell 層，直接接外部輸入
        lstm_in_dim = input_dim
        lstm_hidden = fc_net_hidden[0]
        lstm_params = 4 * (lstm_in_dim * lstm_hidden + lstm_hidden * lstm_hidden + lstm_hidden)
        total_params += lstm_params
    else:
        # 先計算前面的 fc 層參數（除最後一層）
        # 第一個 fc 層：從輸入到 fc_net_hidden[0]
        fc0_params = input_dim * fc_net_hidden[0] + fc_net_hidden[0]
        total_params += fc0_params
        # 後續 fc 層（從 fc_net_hidden[i-1] 到 fc_net_hidden[i]），i = 1 ~ n-2
        for i in range(1, n - 1):
            fc_params = fc_net_hidden[i - 1] * fc_net_hidden[i] + fc_net_hidden[i]
            total_params += fc_params
        # 最後一層為 LSTM cell，其輸入維度來自 fc_net_hidden[-2]
        lstm_in_dim = fc_net_hidden[-2]
        lstm_hidden = fc_net_hidden[-1]
        lstm_params = 4 * (lstm_in_dim * lstm_hidden + lstm_hidden * lstm_hidden + lstm_hidden)
        total_params += lstm_params


    ratio = total_params / 591104
    return ratio