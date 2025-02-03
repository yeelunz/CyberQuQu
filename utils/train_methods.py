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
from .data_stamp import Gdata
from .professions import build_professions
from .skills import build_skill_manager
import threading
import time
import torch
import torch.nn as nn
import numpy as np

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class MaskedLSTMNetwork(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    自訂模型：結合 MLP + LSTM 與動作 Mask 處理
    假設觀測空間 (obs) 前 3 維為 mask，其餘為真實特徵。
    注意：要求動作空間的輸出數 (num_outputs) 與 mask 維度一致。
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # 先呼叫 nn.Module 的初始化
        nn.Module.__init__(self)
        # 再呼叫 TorchModelV2 的初始化（RecurrentNetwork 為 mixin）
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # 假設 obs 空間為 1D 張量，前 3 維為 mask
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 3
        self.real_obs_dim = self.obs_dim - self.mask_dim

        # 若動作輸出數與 mask 維度不一致，則拋出錯誤
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        # 取得 MLP 隱藏層結構 (可由超參數設定，預設 [256, 256, 256])
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256, 256])

        # -------------------------
        # 建立 MLP 層：輸入為 real_obs_dim，依序經過各隱藏層
        # -------------------------
        mlp_layers = []
        in_size = self.real_obs_dim
        for hidden_size in fcnet_hiddens:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            in_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_out_size = in_size  # MLP 的最後輸出維度

        # -------------------------
        # 建立 LSTM 層
        # 設定：input_size 與 hidden_size 均為 mlp_out_size，且採用 batch_first=True
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=mlp_out_size,
            hidden_size=mlp_out_size,
            batch_first=True,
        )

        # -------------------------
        # 建立最終輸出層：映射 LSTM 輸出至 action logits
        # -------------------------
        self.logits_layer = nn.Linear(mlp_out_size, num_outputs)

        # -------------------------
        # 建立 Value function 分支
        # -------------------------
        self.value_branch = nn.Linear(mlp_out_size, 1)

        # -------------------------
        # 儲存 forward 時產生的中間特徵，供 value_function() 使用
        # -------------------------
        self._features = None

        # 固定 LSTM 的時間步長（此處僅作參考，實際切片由 config 決定）
        self.max_seq_len = model_config.get("max_seq_len", 10)

    @override(TorchModelV2)
    def get_initial_state(self):
        """
        回傳單一樣本的初始 RNN state，形狀 [hidden_dim]
        RLlib 會根據實際 batch size 自行擴展。
        """
        # LSTM 的隱藏狀態維度
        hidden_size = self.logits_layer.in_features
        h0 = torch.zeros(hidden_size, dtype=torch.float)
        c0 = torch.zeros(hidden_size, dtype=torch.float)
        return [h0, c0]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        前向傳播（RNN 部分）：處理單步輸入，並維護 LSTM 狀態。
        輸入：
        - inputs: [B, obs_dim]
        - state: list [hidden_state, cell_state]，每個 shape 為 [B_state, hidden_dim]
        - seq_lens: 各序列的實際長度（本範例未用到，可忽略）
        回傳：
        - masked_logits: [B, num_outputs]
        - new_state: list [new_hidden_state, new_cell_state]，每個 shape 為 [B, hidden_dim]
        """

        B_input = inputs.size(0)
        h_in, c_in = state
        B_state = h_in.size(0)


        # 1. 分離 mask 與真實觀測
        mask = inputs[:, :self.mask_dim]
        real_obs = inputs[:, self.mask_dim:]

        # 2. 經過 MLP 產生特徵向量
        mlp_out = self.mlp(real_obs)
        # Debug: mlp_out shape
        # print("mlp_out shape:", mlp_out.shape)

        # 3. 為 LSTM 增加時間軸：變為 [B_input, 1, mlp_out_size]
        lstm_input = mlp_out.unsqueeze(1)
        # print("lstm_input shape:", lstm_input.shape)

        # 4. 處理 state：原始 state shape 為 [B_state, hidden_dim]
        if B_state != B_input:
            # print(f"State batch size ({B_state}) != inputs batch size ({B_input}). Adjusting state...")
            if B_state == 1:
                # 如果 state 只有一個樣本，則用 expand 將其展開（因為該維度是 singleton）
                h_in = h_in.expand(B_input, -1)
                c_in = c_in.expand(B_input, -1)
                # print("After expand, h_in shape:", h_in.shape)
            elif B_input % B_state == 0:
                # 如果 inputs 的 batch size 是 state 的整數倍，則使用 repeat
                repeats = B_input // B_state
                h_in = h_in.repeat(repeats, 1)
                c_in = c_in.repeat(repeats, 1)
                # print("After repeat, h_in shape:", h_in.shape)
            else:
                raise RuntimeError(
                    f"State batch size ({B_state}) cannot be matched to inputs batch size ({B_input})."
                )
        else:
            # print("State batch size matches inputs batch size.")
            pass

        # Debug: 檢查調整後的 state shape
        # print("Final h_in shape (before unsqueeze):", h_in.shape)

        # LSTM 要求的 shape 為 [num_layers, B_input, hidden_dim]，所以 unsqueeze 第一個維度
        h_in = h_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)
        # print("h_in shape after unsqueeze:", h_in.shape)
        # print("c_in shape after unsqueeze:", c_in.shape)

        # 5. LSTM 前向傳播
        lstm_out, [h_out, c_out] = self.lstm(lstm_input, [h_in, c_in])
        # lstm_out: [B_input, 1, hidden_dim] -> 壓縮時間軸
        lstm_out = lstm_out.squeeze(1)
        # print("lstm_out shape after squeeze:", lstm_out.shape)

        # 6. 計算動作 logits
        logits = self.logits_layer(lstm_out)
        # print("logits shape:", logits.shape)

        # 7. 套用動作 mask（無效動作 logits 加極大負值）
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask
        # print("masked_logits shape:", masked_logits.shape)

        # 8. 儲存特徵供 value_function() 使用
        self._features = lstm_out

        # 9. 回傳新的 state（去除 LSTM 層的 layer 維度）
        new_state = [h_out.squeeze(0), c_out.squeeze(0)]
        # print("new_state h_out shape:", new_state[0].shape)
        # print("new_state c_out shape:", new_state[1].shape)

        return masked_logits, new_state

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        RLlib 在非序列情境下會呼叫 forward()，
        這裡直接轉接到 forward_rnn()，確保狀態可以正確傳遞。
        """
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        """
        根據 forward_rnn() 中儲存的特徵，計算 state-value V(s)
        """
        assert self._features is not None, "必須先執行 forward()/forward_rnn() 才能取得 value_function!"
        # value_branch 輸出 shape [B, 1]，reshape 為 [B]
        return torch.reshape(self.value_branch(self._features), [-1])


class MaskedLSTMNetworkWithEmb(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    示範如何解析自訂的 obs，並對不同部位 (職業、技能、特效...) 建立獨立的 embedding。
    你可依實際觀測索引做調整。
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # =============== 基本設定 ===============
        self.mask_dim = 3
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})"
            )

        # =============== Embedding 設定 ===============
        self.num_professions = 13
        self.num_global_skills = 39
        self.num_effects = 20
        
        profession_emb_dim = 8
        skill_emb_dim = 8
        effect_emb_dim = 4

        # 我方/敵方 embedding
        self.profession_embedding_p = nn.Embedding(self.num_professions, profession_emb_dim)
        self.profession_embedding_e = nn.Embedding(self.num_professions, profession_emb_dim)
        self.skill_embedding_p = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.skill_embedding_e = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.effect_embedding_p = nn.Embedding(self.num_effects, effect_emb_dim)
        self.effect_embedding_e = nn.Embedding(self.num_effects, effect_emb_dim)

        # =============== 網路層架構 ===============
        # 計算 MLP 輸入維度 (需手動計算)
        self._mlp_input_dim = (
            profession_emb_dim * 2 +  # 我方+敵方職業
            skill_emb_dim * 2 +       # 我方+敵方技能
            14 * (effect_emb_dim + 3) * 2 +  # 我方+敵方特效 (每個特效含3個數值)
            3 + 1 + 3 + 1 +           # 我方冷卻、HP%、乘子、累積傷害
            3 + 1 + 3 + 1 +           # 敵方冷卻、HP%、乘子、累積傷害
            2                         # 公共觀測
        )

        # MLP 層
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        layers = []
        in_size = self._mlp_input_dim
        for hidden_size in fcnet_hiddens:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.mlp = nn.Sequential(*layers)

        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=in_size,
            batch_first=True
        )

        # 輸出層
        self.logits_layer = nn.Linear(in_size, num_outputs)
        self.value_branch = nn.Linear(in_size, 1)

        self._features = None
        self.max_seq_len = model_config.get("max_seq_len", 10)
        self.effect_id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,19]

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.lstm.hidden_size
        return [
            torch.zeros(hidden_size, dtype=torch.float),
            torch.zeros(hidden_size, dtype=torch.float)
        ]


    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.logits_layer.in_features
        h0 = torch.zeros(hidden_size, dtype=torch.float)
        c0 = torch.zeros(hidden_size, dtype=torch.float)
        return [h0, c0]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """RLlib 在非 RNN 的地方會呼叫 forward()，這裡直接轉接到 forward_rnn()。"""
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        前向傳播（RNN 部分）：處理單步輸入，並維護 LSTM 狀態。
        輸入：
        - inputs: [B, obs_dim]  # B 是批次大小，obs_dim 是觀測空間的維度
        - state: list [hidden_state, cell_state]，每個 shape 為 [B_state, hidden_dim]
        - seq_lens: 各序列的實際長度（本範例未用到，可忽略）
        回傳：
        - masked_logits: [B, num_outputs]  # 動作 logits，已套用 mask
        - new_state: list [new_hidden_state, new_cell_state]，每個 shape 為 [B, hidden_dim]
        """
        # 1. 取得批次大小
        B = inputs.shape[0]  # 批次大小
        h_in, c_in = state   # 解包 LSTM 的 hidden state 和 cell state

        # 2. 處理狀態批次不匹配問題
        if h_in.dim() == 1:
            # 如果狀態是 1D，增加批次維度
            h_in = h_in.unsqueeze(0)
            c_in = c_in.unsqueeze(0)

        B_state = h_in.size(0)  # 狀態的批次大小
        if B_state != B:
            if B_state == 1:
                # 如果狀態批次是 1，擴展到輸入批次大小
                h_in = h_in.expand(B, -1)
                c_in = c_in.expand(B, -1)
            elif B % B_state == 0:
                # 如果輸入批次是狀態批次的整數倍，複製狀態
                repeats = B // B_state
                h_in = h_in.repeat(repeats, 1)
                c_in = c_in.repeat(repeats, 1)
            else:
                raise RuntimeError(
                    f"State batch ({B_state}) must be 1 or divide input batch ({B})"
                )

        # 3. 調整狀態形狀為 LSTM 所需格式 [num_layers, B, hidden_dim]
        h_in = h_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)

        # 4. 解析觀測 (inputs) 的各個部分
        # 4.1 取動作 mask (前 3 維)
        mask = inputs[:, :self.mask_dim]

        # 4.2 解析「我方職業」(profession_p)
        profession_p_one_hot = inputs[:, 3:16]  # one-hot: obs[3..15] (共 13 維)
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)  # 轉成 id
        prof_p_emb = self.profession_embedding_p(prof_p_id)  # [B, profession_emb_dim]

        # 4.3 解析「敵方職業」(profession_e)
        profession_e_one_hot = inputs[:, 78:91]  # one-hot: obs[78..90]
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding_e(prof_e_id)  # [B, profession_emb_dim]

        # 4.4 解析「我方上次使用技能」(skill_p)
        local_skill_p_one_hot = inputs[:, 155:158]  # one-hot: obs[155..157]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)
        global_skill_p_id = prof_p_id * 3 + local_skill_p_id  # 轉成 global skill id
        skill_p_emb = self.skill_embedding_p(global_skill_p_id)  # [B, skill_emb_dim]

        # 4.5 解析「敵方上次使用技能」(skill_e)
        local_skill_e_one_hot = inputs[:, 152:155]  # one-hot: obs[152..154]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 3 + local_skill_e_id
        skill_e_emb = self.skill_embedding_e(global_skill_e_id)  # [B, skill_emb_dim]

        # 4.6 解析「我方特效」(effect_p)
        effect_p_features = []
        for i in range(14):  # 14 種特效
            eff_id = self.effect_id_list[i]  # 特效 ID
            eff_emb = self.effect_embedding_p(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)  # [B, effect_emb_dim]

            # 取得特效的數值 (exist, stack, remain)
            exist = inputs[:, 33 + i * 3]
            stack = inputs[:, 34 + i * 3]
            remain = inputs[:, 35 + i * 3]

            # 拼接 embedding 和數值
            eff_i = torch.cat([
                eff_emb,
                exist.unsqueeze(1),
                stack.unsqueeze(1),
                remain.unsqueeze(1)
            ], dim=1)  # [B, effect_emb_dim + 3]
            effect_p_features.append(eff_i)

        # 將 14 個特效拼接成一個向量
        effect_p_vec = torch.cat(effect_p_features, dim=1)  # [B, 14 * (effect_emb_dim + 3)]

        # 4.7 解析「敵方特效」(effect_e)
        effect_e_features = []
        for i in range(14):
            eff_id = self.effect_id_list[i]
            eff_emb = self.effect_embedding_e(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            exist = inputs[:, 108 + i * 3]
            stack = inputs[:, 109 + i * 3]
            remain = inputs[:, 110 + i * 3]
            eff_i = torch.cat([eff_emb, exist.unsqueeze(1), stack.unsqueeze(1), remain.unsqueeze(1)], dim=1)
            effect_e_features.append(eff_i)
        effect_e_vec = torch.cat(effect_e_features, dim=1)  # [B, 14 * (effect_emb_dim + 3)]

        # 4.8 解析其他連續特徵
        cont_p_cooldown = inputs[:, 0:3]  # 我方冷卻
        cont_p_hp_ratio = inputs[:, 16:17]  # 我方 HP%
        cont_p_multipliers = inputs[:, 18:21]  # 我方乘子
        cont_p_accDmg = inputs[:, 21:22]  # 我方累積傷害

        cont_e_cooldown = inputs[:, 75:78]  # 敵方冷卻
        cont_e_hp_ratio = inputs[:, 91:92]  # 敵方 HP%
        cont_e_multipliers = inputs[:, 93:96]  # 敵方乘子
        cont_e_accDmg = inputs[:, 96:97]  # 敵方累積傷害

        cont_pub_obs = inputs[:, 150:152]  # 公共觀測

        # 拼接所有連續特徵
        cont_features = torch.cat([
            cont_p_cooldown,
            cont_p_hp_ratio,
            cont_p_multipliers,
            cont_p_accDmg,
            cont_e_cooldown,
            cont_e_hp_ratio,
            cont_e_multipliers,
            cont_e_accDmg,
            cont_pub_obs
        ], dim=1)  # [B, 連續特徵總維度]

        # 5. 拼接所有特徵
        combined = torch.cat([
            prof_p_emb,
            prof_e_emb,
            skill_p_emb,
            skill_e_emb,
            effect_p_vec,
            effect_e_vec,
            cont_features,
        ], dim=1)  # [B, 總特徵維度]

        # 6. 通過 MLP
        mlp_out = self.mlp(combined)  # [B, hidden_dim]

        # 7. 通過 LSTM
        lstm_in = mlp_out.unsqueeze(1)  # [B, 1, hidden_dim]
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        lstm_out = lstm_out.squeeze(1)  # [B, hidden_dim]

        # 8. 計算動作 logits
        logits = self.logits_layer(lstm_out)  # [B, num_outputs]
        inf_mask = (1.0 - mask) * -1e10  # 無效動作 mask
        masked_logits = logits + inf_mask  # 套用 mask

        # 9. 儲存特徵供 value_function 使用
        self._features = lstm_out

        # 10. 回傳結果
        new_state = [h_out.squeeze(0), c_out.squeeze(0)]  # 去除 LSTM 層的維度
        return masked_logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None
        return torch.reshape(self.value_branch(self._features), [-1])

    # =============== 4) 為了可視化: 取出所有 embedding 權重，並存成 dict ===============
    def get_all_embeddings(self):
        """
        取得本模型裡面所有 embedding 的 weight，並以 dict 形式回傳，方便你存成 JSON。
        """
        return {
            "profession_p": self.profession_embedding_p.weight.detach().cpu().numpy().tolist(),
            "profession_e": self.profession_embedding_e.weight.detach().cpu().numpy().tolist(),
            "skill_p": self.skill_embedding_p.weight.detach().cpu().numpy().tolist(),
            "skill_e": self.skill_embedding_e.weight.detach().cpu().numpy().tolist(),
            "effect_p": self.effect_embedding_p.weight.detach().cpu().numpy().tolist(),
            "effect_e": self.effect_embedding_e.weight.detach().cpu().numpy().tolist(),
        }



class MaskedLSTMNetworkWithMergedEmb(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    此版本將玩家與敵方對應的 embedding 合併為同一組。
    包含：
      - 職業 embedding：共 13 個向量
      - 技能 embedding：共 39 個向量 (global skill id)
      - 特效 embedding：共 20 個向量
    其它部分保持與原本設計類似，最終拼接玩家/敵方的特徵後進行 MLP -> LSTM -> 輸出 logits。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # =============== 基本設定 ===============
        self.mask_dim = 3
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})")

        # =============== Embedding 設定 ===============
        self.num_professions = 13
        self.num_global_skills = 39
        self.num_effects = 20

        profession_emb_dim = 8
        skill_emb_dim = 8
        effect_emb_dim = 4

        # 合併玩家與敵方：共用同一組 embedding table
        self.profession_embedding = nn.Embedding(self.num_professions, profession_emb_dim)
        self.skill_embedding = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.effect_embedding = nn.Embedding(self.num_effects, effect_emb_dim)

        # =============== 網路層架構 ===============
        # 計算 MLP 輸入維度 (需手動計算)
        # 說明：
        #  - 玩家職業與敵方職業：各 profession_emb_dim (2*profession_emb_dim)
        #  - 玩家上次使用技能與敵方上次使用技能：各 skill_emb_dim (2*skill_emb_dim)
        #  - 玩家特效：14 個特效，每個特效拼接 (effect_emb_dim + 3)，共 14*(effect_emb_dim+3)
        #  - 敵方特效：同上，另加 14*(effect_emb_dim+3)
        #  - 其他連續特徵：玩家部分：冷卻 (3) + HP% (1) + 乘子 (3) + 累積傷害 (1)
        #                    敵方部分：冷卻 (3) + HP% (1) + 乘子 (3) + 累積傷害 (1)
        #  - 公共觀測：2
        self._mlp_input_dim = (
            profession_emb_dim * 2 + 
            skill_emb_dim * 2 +       
            14 * (effect_emb_dim + 3) * 2 +  
            (3 + 1 + 3 + 1) +  # 玩家連續特徵
            (3 + 1 + 3 + 1) +  # 敵方連續特徵
            2                # 公共觀測
        )

        # MLP 層
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        layers = []
        in_size = self._mlp_input_dim
        for hidden_size in fcnet_hiddens:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.mlp = nn.Sequential(*layers)

        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=in_size,
            batch_first=True
        )

        # 輸出層 (動作 logits 與 value branch)
        self.logits_layer = nn.Linear(in_size, num_outputs)
        self.value_branch = nn.Linear(in_size, 1)

        self._features = None
        self.max_seq_len = model_config.get("max_seq_len", 10)
        # 特效 ID 列表 (依照觀測順序：共 14 種，分別為 1..13 與 19)
        self.effect_id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,19]

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.logits_layer.in_features
        h0 = torch.zeros(hidden_size, dtype=torch.float)
        c0 = torch.zeros(hidden_size, dtype=torch.float)
        return [h0, c0]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """RLlib 在非 RNN 的情況下呼叫 forward()，這裡直接轉接到 forward_rnn()。"""
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        前向傳播：解析各部分的觀測，查詢 embedding，拼接後送入 MLP -> LSTM -> 輸出 logits。
        """
        B = inputs.shape[0]  # 批次大小
        h_in, c_in = state

        # 處理狀態批次不匹配問題
        if h_in.dim() == 1:
            h_in = h_in.unsqueeze(0)
            c_in = c_in.unsqueeze(0)
        B_state = h_in.size(0)
        if B_state != B:
            if B_state == 1:
                h_in = h_in.expand(B, -1)
                c_in = c_in.expand(B, -1)
            elif B % B_state == 0:
                repeats = B // B_state
                h_in = h_in.repeat(repeats, 1)
                c_in = c_in.repeat(repeats, 1)
            else:
                raise RuntimeError(f"State batch ({B_state}) must be 1 or divide input batch ({B})")
        h_in = h_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)

        # 4. 解析各部分觀測
        # 4.1 取動作 mask (前 3 維)
        mask = inputs[:, :self.mask_dim]

        # 4.2 玩家職業 (obs[3:16]) -> one-hot 轉 id -> embedding
        profession_p_one_hot = inputs[:, 3:16]
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)
        prof_p_emb = self.profession_embedding(prof_p_id)

        # 4.3 敵方職業 (obs[78:91])
        profession_e_one_hot = inputs[:, 78:91]
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding(prof_e_id)

        # 4.4 玩家上次使用技能 (obs[155:158]) -> 先取 local skill id，再轉 global skill id -> embedding
        local_skill_p_one_hot = inputs[:, 155:158]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)
        global_skill_p_id = prof_p_id * 3 + local_skill_p_id
        skill_p_emb = self.skill_embedding(global_skill_p_id)

        # 4.5 敵方上次使用技能 (obs[152:155])
        local_skill_e_one_hot = inputs[:, 152:155]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 3 + local_skill_e_id
        skill_e_emb = self.skill_embedding(global_skill_e_id)

        # 4.6 玩家特效 (obs[33:74]) -> 每 3 維一組，共 14 組
        effect_p_features = []
        for i in range(14):
            eff_id = self.effect_id_list[i]
            # 查詢 embedding (用同一個 effect_embedding)
            eff_emb = self.effect_embedding(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            exist = inputs[:, 33 + i * 3]
            stack = inputs[:, 34 + i * 3]
            remain = inputs[:, 35 + i * 3]
            eff_i = torch.cat([eff_emb,
                               exist.unsqueeze(1),
                               stack.unsqueeze(1),
                               remain.unsqueeze(1)], dim=1)
            effect_p_features.append(eff_i)
        effect_p_vec = torch.cat(effect_p_features, dim=1)

        # 4.7 敵方特效 (obs[108:149])
        effect_e_features = []
        for i in range(14):
            eff_id = self.effect_id_list[i]
            eff_emb = self.effect_embedding(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            exist = inputs[:, 108 + i * 3]
            stack = inputs[:, 109 + i * 3]
            remain = inputs[:, 110 + i * 3]
            eff_i = torch.cat([eff_emb,
                               exist.unsqueeze(1),
                               stack.unsqueeze(1),
                               remain.unsqueeze(1)], dim=1)
            effect_e_features.append(eff_i)
        effect_e_vec = torch.cat(effect_e_features, dim=1)

        # 4.8 解析其他連續特徵
        cont_p_cooldown = inputs[:, 0:3]      # 玩家冷卻
        cont_p_hp_ratio = inputs[:, 16:17]      # 玩家 HP%
        cont_p_multipliers = inputs[:, 18:21]     # 玩家乘子
        cont_p_accDmg = inputs[:, 21:22]        # 玩家累積傷害

        cont_e_cooldown = inputs[:, 75:78]      # 敵方冷卻
        cont_e_hp_ratio = inputs[:, 91:92]        # 敵方 HP%
        cont_e_multipliers = inputs[:, 93:96]       # 敵方乘子
        cont_e_accDmg = inputs[:, 96:97]        # 敵方累積傷害

        cont_pub_obs = inputs[:, 150:152]       # 公共觀測

        cont_features = torch.cat([
            cont_p_cooldown,
            cont_p_hp_ratio,
            cont_p_multipliers,
            cont_p_accDmg,
            cont_e_cooldown,
            cont_e_hp_ratio,
            cont_e_multipliers,
            cont_e_accDmg,
            cont_pub_obs
        ], dim=1)

        # 5. 拼接所有特徵
        combined = torch.cat([
            prof_p_emb,
            prof_e_emb,
            skill_p_emb,
            skill_e_emb,
            effect_p_vec,
            effect_e_vec,
            cont_features,
        ], dim=1)

        # 6. 通過 MLP
        mlp_out = self.mlp(combined)

        # 7. 通過 LSTM
        lstm_in = mlp_out.unsqueeze(1)
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        lstm_out = lstm_out.squeeze(1)

        # 8. 動作 logits (並套用動作 mask)
        logits = self.logits_layer(lstm_out)
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        self._features = lstm_out
        new_state = [h_out.squeeze(0), c_out.squeeze(0)]
        return masked_logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None
        return torch.reshape(self.value_branch(self._features), [-1])

    # 為方便視覺化，回傳所有 embedding 的權重 (合併版本)
    def get_all_embeddings(self):
        return {
            "profession": self.profession_embedding.weight.detach().cpu().numpy().tolist(),
            "skill": self.skill_embedding.weight.detach().cpu().numpy().tolist(),
            "effect": self.effect_embedding.weight.detach().cpu().numpy().tolist(),
        }



ModelCatalog.register_custom_model("my_mask_model", MaskedLSTMNetwork)
ModelCatalog.register_custom_model("my_mask_model_with_emb", MaskedLSTMNetworkWithEmb)
ModelCatalog.register_custom_model("my_mask_model_with_emb_combined", MaskedLSTMNetworkWithMergedEmb)


stop_training_flag = threading.Event()


def multi_agent_cross_train(num_iterations,
                            model_name="my_multiagent_ai",
                            hyperparams=None):
    """
    多智能體交叉訓練
    """

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
            num_gpus_per_env_runner=1,
            sample_timeout_s=120
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
            # ... 可根據需要增加更多動態帶入的超參數 ...
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
    
    # 這邊是訓練完了 如果是 mask_model_with_emb_combined 就要把 embedding 存起來到embedding.json
    if hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb" or hyperparams.get("mask_model", "my_mask_model") == "my_mask_model_with_emb_combined":
        print("Saving embeddings...")
        save_root = os.path.join("data", "saved_models", model_name)
        meta_path = os.path.join(save_root, "embeddings.json")
        model = algo.get_policy("shared_policy").model
        embeddings = model.get_all_embeddings()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

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



import os
import math
import random
import time

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
    fc_hiddens = [256, 256 ,256]
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
                max_seq_len  = hyperparams.get("max_seq_len", 10)
                mask_model = hyperparams.get("mask_model", "my_mask_model")
        except FileNotFoundError:
            print(f"找不到 {meta_path}，將使用預設的 fcnet_hiddens。")
    else:
        print(f"mata data 路徑 {check_point_path} 不存在。")
        
    

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
            "custom_model": mask_model,
            "fcnet_hiddens": fc_hiddens,
            "fcnet_activation": "ReLU",
            "vf_share_layers": False,
            "max_seq_len" : max_seq_len
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
    
    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
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

            env = BattleEnv(make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=p))
            done = False
            obs, _ = env.reset()
            policy = trainer.get_policy("shared_policy")
            state = policy.model.get_initial_state()

            while not done:
                ai_package = trainer.compute_single_action(obs['player'], state=state, policy_id="shared_policy")
                ai_act = ai_package[0]
                state = ai_package[1]

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
            policy = trainer.get_policy("shared_policy")
            state = policy.model.get_initial_state()

            while not done:
                enemy_act = random.choice([0, 1, 2])  # 隨機對手
 
                ai_package = trainer.compute_single_action(obs['enemy'], state=state, policy_id="shared_policy")
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


import copy

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
    skillusedFreq = {p.name: {0:0,1:0,2:0} for p in professions}
    
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
                    pmask = obs["player"][0:3]
                    emask = obs["enemy"][0:3]
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



def version_test_random_vs_random_sse_ai(professions, skill_mgr, num_battles=100,model_path_1="my_battle_ppo_checkpoints"):
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
    skillusedFreq = {p.name: {0:0,1:0,2:0} for p in professions}
    
    fc_hiddens = [256, 256 ,256]
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
                max_seq_len  = hyperparams.get("max_seq_len", 10)
                mask_model = hyperparams.get("mask_model", "my_mask_model")
        except FileNotFoundError:
            print(f"找不到 {meta_path}，將使用預設的 fcnet_hiddens。")
    
    
    # 初始化 AI ELO
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
            "custom_model": mask_model,
            "fcnet_hiddens": fc_hiddens,
            "fcnet_activation": "ReLU",
            "vf_share_layers": False,
            "max_seq_len" : max_seq_len
        },
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
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=p, pr2=op)
                    )
                else:
                    env = BattleEnv(
                        make_env_config(skill_mgr, professions, show_battlelog=False, pr1=op, pr2=p)
                    )

                done = False
                obs, _ = env.reset()
                policy = trainer.get_policy("shared_policy")
                state_1 = policy.model.get_initial_state()
                state_2 = policy.model.get_initial_state()
                
                rounds = 0  # 計算單場回合數
                while not done:
                    rounds += 1
 
                    
                    p_act_pack = trainer.compute_single_action(obs['player'], state=state_1,policy_id="shared_policy")
                    p_act = p_act_pack[0]
                    state_1 = p_act_pack[1]
                    # if p act in mask is 0, then choose random action
                    e_act_pack = trainer.compute_single_action(obs['enemy'] ,state=state_2,policy_id="shared_policy")
                    e_act = e_act_pack[0]
                    state_2 = e_act_pack[1]
                    
                    
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
    r = Gdata(res,gv['version'],"cross_validation_ai",model = model_path_1)  # 這裡是你的自訂函數，用來儲存結果
    r.save()
    # 最後整包結束再 yield 一次
    yield {
        "type": "final_result",
        "progress": 100,
        "message": "對戰產生完成",
        "data": res
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