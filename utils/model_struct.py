from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
import torch
from ray.rllib.models.catalog import ModelCatalog
import numpy as np
# F
import torch.nn.functional as F


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

        # 假設 obs 空間為 1D 張量，前 4 維為 mask
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4
        self.real_obs_dim = self.obs_dim - self.mask_dim

        # 若動作輸出數與 mask 維度不一致，則拋出錯誤
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        # 取得 MLP 隱藏層結構 (可由超參數設定，預設 [256, 256, 256])
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])

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
        self.mask_dim = 4
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})"
            )

        # =============== Embedding 設定 ===============
        self.num_professions = 13
        self.num_global_skills = 52
        self.num_effects = 13
        
        profession_emb_dim = 8
        skill_emb_dim = 12
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
            13 * (effect_emb_dim + 3) * 2 +  # 我方+敵方特效 (每個特效含3個數值)
            4 + 1 + 3 + 1 +           # 我方冷卻、HP%、乘子、累積傷害
            4 + 1 + 3 + 1 +           # 敵方冷卻、HP%、乘子、累積傷害
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
        self.effect_id_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]

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
        # 4.1 取動作 mask (前 4 維)
        mask = inputs[:, :self.mask_dim]

        # 4.2 解析「我方職業」(profession_p)
        profession_p_one_hot = inputs[:, 4:17]  # one-hot: obs[4..17] (共 16 維)
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)  # 轉成 id
        prof_p_emb = self.profession_embedding_p(prof_p_id)  # [B, profession_emb_dim]

        # 4.3 解析「敵方職業」(profession_e)
        profession_e_one_hot = inputs[:, 80:93]  # 
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding_e(prof_e_id)  # [B, profession_emb_dim]

        # 4.4 解析「我方上次使用技能」(skill_p)
        local_skill_p_one_hot = inputs[:, 158:162]  # one-hot: obs[155..157]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)
        global_skill_p_id = prof_p_id * 4 + local_skill_p_id  # 轉成 global skill id
        skill_p_emb = self.skill_embedding_p(global_skill_p_id)  # [B, skill_emb_dim]

        # 4.5 解析「敵方上次使用技能」(skill_e)
        local_skill_e_one_hot = inputs[:, 154:158]  # one-hot: obs[152..154]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 4 + local_skill_e_id
        skill_e_emb = self.skill_embedding_e(global_skill_e_id)  # [B, skill_emb_dim]

        # 4.6 解析「我方特效」(effect_p)
        effect_p_features = []
        for i in range(13):  # 14 種特效
            eff_id = self.effect_id_list[i]  # 特效 ID
            eff_emb = self.effect_embedding_p(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)  # [B, effect_emb_dim]

            # 取得特效的數值 (exist, stack, remain)
            exist = inputs[:, 37 + i * 3]
            stack = inputs[:, 38 + i * 3]
            remain = inputs[:, 39 + i * 3]

            # 拼接 embedding 和數值
            eff_i = torch.cat([
                eff_emb,
                exist.unsqueeze(1),
                stack.unsqueeze(1),
                remain.unsqueeze(1)
            ], dim=1)  # [B, effect_emb_dim + 3]
            effect_p_features.append(eff_i)

        # 將 13 個特效拼接成一個向量
        effect_p_vec = torch.cat(effect_p_features, dim=1)  # [B, 14 * (effect_emb_dim + 3)]

        # 4.7 解析「敵方特效」(effect_e)
        effect_e_features = []
        for i in range(13):
            eff_id = self.effect_id_list[i]
            eff_emb = self.effect_embedding_e(torch.tensor([eff_id], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            exist = inputs[:, 113 + i * 3]
            stack = inputs[:, 114 + i * 3]
            remain = inputs[:, 115 + i * 3]
            eff_i = torch.cat([eff_emb, exist.unsqueeze(1), stack.unsqueeze(1), remain.unsqueeze(1)], dim=1)
            effect_e_features.append(eff_i)
        effect_e_vec = torch.cat(effect_e_features, dim=1)  # [B, 14 * (effect_emb_dim + 3)]

        # 4.8 解析其他連續特徵
        cont_p_cooldown = inputs[:, 162:166]  # 我方冷卻
        cont_p_hp_ratio = inputs[:, 17:18]  # 我方 HP%
        cont_p_multipliers = inputs[:, 19:22]  # 我方乘子
        cont_p_accDmg = inputs[:, 22:23]  # 我方累積傷害

        cont_e_cooldown = inputs[:, 166:170]  # 敵方冷卻
        cont_e_hp_ratio = inputs[:, 93:94]  # 敵方 HP%
        cont_e_multipliers = inputs[:,95:98]  # 敵方乘子
        cont_e_accDmg = inputs[:, 98:99]  # 敵方累積傷害
        cont_pub_obs = inputs[:, 152:154]  # 公共觀測

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


class MaskedLSTMNetworkWithEmbV2(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    此模型示範如何解析自訂的觀測資料，並針對職業、技能、特效進行嵌入，
    且在職業與技能部分融合額外資訊：
      - 職業部分：融合 profession id 與額外資訊（最大HP、baseAtk、baseDef）
      - 技能部分：融合 skill id 與 skill type（來自 one-hot 轉換後的 id）
    融合方式採用串接後線性映射，而非算術加法。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # =============== 基本設定 ===============
        self.mask_dim = 4
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})")

        # =============== Embedding 設定 ===============
        self.num_professions = 13
        self.num_global_skills = 52
        self.num_effects = 13

        # 嵌入向量維度設定
        profession_emb_dim = 8
        skill_emb_dim = 12
        effect_emb_dim = 4

        # --------------------
        # 職業 embedding（依據 id），並融合額外資訊：maxHP、baseAtk、baseDef (各1維)
        self.profession_embedding_p = nn.Embedding(self.num_professions, profession_emb_dim)
        self.profession_embedding_e = nn.Embedding(self.num_professions, profession_emb_dim)
        # 融合層：串接後映射回原職業嵌入向量維度
        self.profession_fusion_p = nn.Linear(profession_emb_dim + 3, profession_emb_dim)
        self.profession_fusion_e = nn.Linear(profession_emb_dim + 3, profession_emb_dim)

        # --------------------
        # 技能 embedding（依據 global skill id），並融合技能類型資訊（假設轉換後有 3 種可能）
        self.skill_embedding_p = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.skill_embedding_e = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.skill_type_embedding_p = nn.Embedding(3, skill_emb_dim)
        self.skill_type_embedding_e = nn.Embedding(3, skill_emb_dim)
        # 融合層：串接技能 id 與技能類型嵌入後映射成同樣的維度
        self.skill_fusion_p = nn.Linear(skill_emb_dim * 2, skill_emb_dim)
        self.skill_fusion_e = nn.Linear(skill_emb_dim * 2, skill_emb_dim)

        # --------------------
        # 特效 embedding（每個特效包含嵌入向量及 3 個數值：exist, stack, remain）
        self.effect_embedding_p = nn.Embedding(self.num_effects, effect_emb_dim)
        self.effect_embedding_e = nn.Embedding(self.num_effects, effect_emb_dim)

        # =============== 網路層架構 ===============
        # 計算 MLP 輸入維度：
        # - 職業部分：我方與敵方各一個向量 (profession_emb_dim * 2)
        # - 技能部分：我方與敵方各一個向量 (skill_emb_dim * 2)
        # - 特效部分：我方與敵方各 13 個特效，每個特效 (effect_emb_dim + 3)
        # - 連續特徵：我方 4+1+3+1，敵方 4+1+3+1，再加上公共觀測 2
        self._mlp_input_dim = (
            profession_emb_dim * 2 +    # 我方 + 敵方職業
            skill_emb_dim * 2 +         # 我方 + 敵方技能
            13 * (effect_emb_dim + 3) * 2 +   # 我方 + 敵方特效
            4 + 1 + 3 + 1 +             # 我方連續特徵：冷卻, HP%, 乘子, 累積傷害
            4 + 1 + 3 + 1 +             # 敵方連續特徵：冷卻, HP%, 乘子, 累積傷害
            2                           # 公共觀測
        )

        # MLP 層：預設兩層，每層 256 單位 (可由 model_config 調整)
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

        # 輸出層：動作 logits 與 value branch
        self.logits_layer = nn.Linear(in_size, num_outputs)
        self.value_branch = nn.Linear(in_size, 1)

        self._features = None
        self.max_seq_len = model_config.get("max_seq_len", 10)
        self.effect_id_list = list(range(self.num_effects))

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.logits_layer.in_features
        h0 = torch.zeros(hidden_size, dtype=torch.float)
        c0 = torch.zeros(hidden_size, dtype=torch.float)
        return [h0, c0]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """RLlib 在非 RNN 部分呼叫 forward()，此處轉接至 forward_rnn()。"""
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        前向傳播（RNN 部分）：處理單步輸入並維護 LSTM 狀態。
        """
        # 1. 取得批次大小
        B = inputs.shape[0]
        h_in, c_in = state

        # 2. 處理狀態批次不匹配
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
        # 調整 LSTM 輸入狀態形狀為 [num_layers, B, hidden_dim]
        h_in = h_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)

        # 3. 解析觀測資料
        # 3.1 取動作 mask (前 4 維)
        mask = inputs[:, :self.mask_dim]

        # 3.2 解析「我方職業」
        # 假設 obs[4:17] 為 one-hot 編碼，共 13 維
        profession_p_one_hot = inputs[:, 4:17]
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)
        prof_p_emb = self.profession_embedding_p(prof_p_id)
        # 額外資訊：最大HP (index 18)、baseAtk (index 23)、baseDef (index 24)
        extra_prof_p = torch.cat([
            inputs[:, 18:19],
            inputs[:, 23:24],
            inputs[:, 24:25]
        ], dim=1)
        # 融合資訊
        prof_p_combined = self.profession_fusion_p(torch.cat([prof_p_emb, extra_prof_p], dim=1))

        # 3.3 解析「敵方職業」
        profession_e_one_hot = inputs[:, 80:93]
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding_e(prof_e_id)
        # 敵方額外資訊分別位於 indices 94, 95, 96
        extra_prof_e = torch.cat([
            inputs[:, 94:95],
            inputs[:, 95:96],
            inputs[:, 96:97]
        ], dim=1)
        prof_e_combined = self.profession_fusion_e(torch.cat([prof_e_emb, extra_prof_e], dim=1))

        # 3.4 解析「我方技能」：融合 skill id 與 skill type
        # 取得玩家選擇的技能槽：假設 obs[158:162] 為 one-hot (四個槽位)
        local_skill_p_one_hot = inputs[:, 158:162]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)  # 值介於 0～3
        global_skill_p_id = prof_p_id * 4 + local_skill_p_id  # 轉換成 global skill id
        skill_id_emb_p = self.skill_embedding_p(global_skill_p_id)

        # 取得玩家各技能槽的 skill type：假設分別位於 obs[25:28], [28:31], [31:34], [34:37]
        skill0_type = torch.argmax(inputs[:, 25:28], dim=1)
        skill1_type = torch.argmax(inputs[:, 28:31], dim=1)
        skill2_type = torch.argmax(inputs[:, 31:34], dim=1)
        skill3_type = torch.argmax(inputs[:, 34:37], dim=1)
        all_skill_types = torch.stack([skill0_type, skill1_type, skill2_type, skill3_type], dim=1)  # [B, 4]
        # 根據玩家選擇的技能槽 (local_skill_p_id) 取得對應 skill type
        selected_skill_type = all_skill_types.gather(1, local_skill_p_id.unsqueeze(1)).squeeze(1)
        skill_type_emb_p = self.skill_type_embedding_p(selected_skill_type)
        # 融合 skill id 與 skill type 的資訊
        skill_p_combined = self.skill_fusion_p(torch.cat([skill_id_emb_p, skill_type_emb_p], dim=1))

        # 3.5 解析「敵方技能」：同樣融合 skill id 與 skill type
        local_skill_e_one_hot = inputs[:, 154:158]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 4 + local_skill_e_id
        skill_id_emb_e = self.skill_embedding_e(global_skill_e_id)

        # 假設敵方 skill type 分別位於 obs[101:104], [104:107], [107:110], [110:113]
        enemy_skill0_type = torch.argmax(inputs[:, 101:104], dim=1)
        enemy_skill1_type = torch.argmax(inputs[:, 104:107], dim=1)
        enemy_skill2_type = torch.argmax(inputs[:, 107:110], dim=1)
        enemy_skill3_type = torch.argmax(inputs[:, 110:113], dim=1)
        all_enemy_skill_types = torch.stack([enemy_skill0_type, enemy_skill1_type, enemy_skill2_type, enemy_skill3_type], dim=1)
        selected_enemy_skill_type = all_enemy_skill_types.gather(1, local_skill_e_id.unsqueeze(1)).squeeze(1)
        skill_type_emb_e = self.skill_type_embedding_e(selected_enemy_skill_type)
        skill_e_combined = self.skill_fusion_e(torch.cat([skill_id_emb_e, skill_type_emb_e], dim=1))

        # 3.6 解析「我方特效」
        effect_p_features = []
        for i in range(self.num_effects):
            # 固定特效 id 為 i
            eff_emb = self.effect_embedding_p(torch.tensor([i], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            # 取出 exist, stack, remain，假設分別位於 obs[37 + i*3], [38 + i*3], [39 + i*3]
            exist = inputs[:, 37 + i * 3]
            stack = inputs[:, 38 + i * 3]
            remain = inputs[:, 39 + i * 3]
            eff_feature = torch.cat([eff_emb,
                                     exist.unsqueeze(1),
                                     stack.unsqueeze(1),
                                     remain.unsqueeze(1)], dim=1)
            effect_p_features.append(eff_feature)
        effect_p_vec = torch.cat(effect_p_features, dim=1)

        # 3.7 解析「敵方特效」
        effect_e_features = []
        for i in range(self.num_effects):
            eff_emb = self.effect_embedding_e(torch.tensor([i], device=inputs.device))
            eff_emb = eff_emb.expand(B, -1)
            exist = inputs[:, 113 + i * 3]
            stack = inputs[:, 114 + i * 3]
            remain = inputs[:, 115 + i * 3]
            eff_feature = torch.cat([eff_emb,
                                     exist.unsqueeze(1),
                                     stack.unsqueeze(1),
                                     remain.unsqueeze(1)], dim=1)
            effect_e_features.append(eff_feature)
        effect_e_vec = torch.cat(effect_e_features, dim=1)

        # 3.8 解析其他連續特徵
        # 我方：冷卻 (obs[162:166]), HP% (obs[17:18]), 乘子 (obs[19:22]), 累積傷害 (obs[22:23])
        cont_p_cooldown    = inputs[:, 162:166]
        cont_p_hp_ratio    = inputs[:, 17:18]
        cont_p_multipliers = inputs[:, 19:22]
        cont_p_accDmg      = inputs[:, 22:23]
        # 敵方：冷卻 (obs[166:170]), HP% (obs[93:94]), 乘子 (obs[95:98]), 累積傷害 (obs[98:99])
        cont_e_cooldown    = inputs[:, 166:170]
        cont_e_hp_ratio    = inputs[:, 93:94]
        cont_e_multipliers = inputs[:, 95:98]
        cont_e_accDmg      = inputs[:, 98:99]
        # 公共觀測：假設在 obs[152:154]
        cont_pub_obs       = inputs[:, 152:154]

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

        # 4. 拼接所有特徵，作為 MLP 輸入
        combined = torch.cat([
            prof_p_combined,
            prof_e_combined,
            skill_p_combined,
            skill_e_combined,
            effect_p_vec,
            effect_e_vec,
            cont_features
        ], dim=1)

        # 5. 透過 MLP
        mlp_out = self.mlp(combined)

        # 6. 透過 LSTM (增加時間維度)
        lstm_in = mlp_out.unsqueeze(1)  # [B, 1, hidden_dim]
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        lstm_out = lstm_out.squeeze(1)

        # 7. 產生動作 logits 並套用 mask
        logits = self.logits_layer(lstm_out)
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        # 儲存特徵以供 value_function 使用
        self._features = lstm_out

        # 回傳 logits 與新的 RNN state
        new_state = [h_out.squeeze(0), c_out.squeeze(0)]
        return masked_logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None
        return torch.reshape(self.value_branch(self._features), [-1])

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
    合併玩家與敵方 embedding 的版本，結構與新版程式碼對齊。
    - 職業/技能/特效共用同一組 embedding
    - 調整特徵索引與新版觀測空間一致
    - 統一狀態處理與 LSTM 流程
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # =============== 基本設定 ===============
        self.mask_dim = 4
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})")

        # =============== Embedding 設定 (合併版) ===============
        self.num_professions = 13
        self.num_global_skills = 52  # 注意與新版對齊 (13職業*4技能)
        self.num_effects = 13        # 與新版特效數一致

        profession_emb_dim = 8
        skill_emb_dim = 12           # 與新版技能嵌入維度一致
        effect_emb_dim = 4

        # 合併玩家與敵方的 embedding 表
        self.profession_embedding = nn.Embedding(self.num_professions, profession_emb_dim)
        self.skill_embedding = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.effect_embedding = nn.Embedding(self.num_effects, effect_emb_dim)

        # =============== 網路層架構 ===============
        # MLP 輸入維度計算 (需與特徵拼接結果一致)
        self._mlp_input_dim = (
            profession_emb_dim * 2 +    # 玩家+敵方職業
            skill_emb_dim * 2 +         # 玩家+敵方技能
            13 * (effect_emb_dim + 3) * 2 +  # 雙方特效各13個 (新版結構)
            (4 + 1 + 3 + 1) * 2 +       # 雙方連續特徵 (冷卻4維)
            2                           # 公共觀測
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
        self.effect_id_list = list(range(13))  # 特效ID 0~12 (與新版對齊)

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.logits_layer.in_features
        return [torch.zeros(hidden_size), torch.zeros(hidden_size)]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """轉接到 forward_rnn"""
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        B = inputs.shape[0]
        h_in, c_in = state

        # 狀態批次處理 (與新版邏輯一致)
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
                raise RuntimeError(f"State batch ({B_state}) must divide input batch ({B})")
        h_in, c_in = h_in.unsqueeze(0), c_in.unsqueeze(0)

        # 1. 動作 mask
        mask = inputs[:, :self.mask_dim]

        # 2. 解析職業 (索引與新版對齊)
        profession_p_one_hot = inputs[:, 4:17]  # 玩家職業 one-hot [4..16]
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)
        prof_p_emb = self.profession_embedding(prof_p_id)

        profession_e_one_hot = inputs[:, 80:93]  # 敵方職業 [80..92]
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding(prof_e_id)

        # 3. 解析技能 (使用新版索引與計算方式)
        # 玩家技能 (obs[158:162] 對應 local_skill_p)
        local_skill_p_one_hot = inputs[:, 158:162]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)
        global_skill_p_id = prof_p_id * 4 + local_skill_p_id  # 4技能/職業
        skill_p_emb = self.skill_embedding(global_skill_p_id)

        # 敵方技能 (obs[154:158])
        local_skill_e_one_hot = inputs[:, 154:158]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 4 + local_skill_e_id
        skill_e_emb = self.skill_embedding(global_skill_e_id)

        # 4. 解析特效 (索引與新版對齊)
        def process_effects(start_idx, batch_size):
            features = []
            for i in range(13):
                eff_id = self.effect_id_list[i]
                emb = self.effect_embedding(torch.tensor([eff_id], device=inputs.device)).expand(batch_size, -1)
                exist = inputs[:, start_idx + i*3]
                stack = inputs[:, start_idx + i*3 + 1]
                remain = inputs[:, start_idx + i*3 + 2]
                features.append(torch.cat([emb, 
                                         exist.unsqueeze(1),
                                         stack.unsqueeze(1),
                                         remain.unsqueeze(1)], dim=1))
            return torch.cat(features, dim=1)

        # 玩家特效起始索引 37 (新版)
        effect_p_vec = process_effects(37, B)
        # 敵方特效起始索引 113
        effect_e_vec = process_effects(113, B)

        # 5. 連續特徵 (索引與新版對齊)
        cont_p_cooldown = inputs[:, 162:166]    # 玩家冷卻4維
        cont_p_hp_ratio = inputs[:, 17:18]      # HP%
        cont_p_multipliers = inputs[:, 19:22]   # 乘子3維
        cont_p_accDmg = inputs[:, 22:23]        # 累積傷害

        cont_e_cooldown = inputs[:, 166:170]    # 敵方冷卻
        cont_e_hp_ratio = inputs[:, 93:94]
        cont_e_multipliers = inputs[:, 95:98]
        cont_e_accDmg = inputs[:, 98:99]

        cont_pub_obs = inputs[:, 152:154]       # 公共觀測

        cont_features = torch.cat([
            cont_p_cooldown, cont_p_hp_ratio, cont_p_multipliers, cont_p_accDmg,
            cont_e_cooldown, cont_e_hp_ratio, cont_e_multipliers, cont_e_accDmg,
            cont_pub_obs
        ], dim=1)

        # 特徵拼接
        combined = torch.cat([
            prof_p_emb, prof_e_emb,
            skill_p_emb, skill_e_emb,
            effect_p_vec, effect_e_vec,
            cont_features
        ], dim=1)

        # 網路前傳
        mlp_out = self.mlp(combined)
        lstm_in = mlp_out.unsqueeze(1)
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        lstm_out = lstm_out.squeeze(1)

        # 輸出處理
        logits = self.logits_layer(lstm_out)
        masked_logits = logits + (1.0 - mask) * -1e10

        self._features = lstm_out
        return masked_logits, [h_out.squeeze(0), c_out.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self):
        return self.value_branch(self._features).squeeze(1)

    def get_all_embeddings(self):
        """合併版嵌入權重"""
        return {
            "profession": self.profession_embedding.weight.detach().cpu().tolist(),
            "skill": self.skill_embedding.weight.detach().cpu().tolist(),
            "effect": self.effect_embedding.weight.detach().cpu().tolist(),
        }
        

class MaskedLSTMNetworkWithMergedEmbV2(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    合併玩家與敵方 embedding 的版本，結構與新版程式碼對齊。
    - 職業/技能/特效共用同一組 embedding 表
    - 調整特徵索引與新版觀測空間一致
    - 技能部分融合 skill id 與 skill type 資訊（透過串接後線性融合）
    - 統一狀態處理與 LSTM 流程
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # =============== 基本設定 ===============
        self.mask_dim = 4
        self.obs_dim = obs_space.shape[0]
        if num_outputs != self.mask_dim:
            raise ValueError(f"num_outputs({num_outputs}) != mask_dim({self.mask_dim})")

        # =============== Embedding 設定 (合併版) ===============
        self.num_professions = 13
        self.num_global_skills = 52  # 注意：13 職業 * 4 技能
        self.num_effects = 13        # 與新版特效數一致

        profession_emb_dim = 8
        skill_emb_dim = 12           # 與新版技能嵌入維度一致
        effect_emb_dim = 4

        # 合併玩家與敵方的 embedding 表
        self.profession_embedding = nn.Embedding(self.num_professions, profession_emb_dim)
        self.skill_embedding = nn.Embedding(self.num_global_skills, skill_emb_dim)
        self.effect_embedding = nn.Embedding(self.num_effects, effect_emb_dim)

        # ---- 新增：技能類型相關 embedding 與融合層 ----
        # 假設 skill type 轉成 id 後共有 3 種可能
        self.skill_type_embedding = nn.Embedding(3, skill_emb_dim)
        # 融合層：將 skill id 與 skill type 的 embedding 串接後映射回 skill_emb_dim
        self.skill_fusion = nn.Linear(skill_emb_dim * 2, skill_emb_dim)

        # =============== 網路層架構 ===============
        # MLP 輸入維度計算 (需與特徵拼接結果一致)
        # - 職業：玩家 + 敵方 (profession_emb_dim * 2)
        # - 技能：玩家 + 敵方 (skill_emb_dim * 2) ※ 這裡使用融合後的技能向量
        # - 特效：雙方各 13 個特效，每個特效 (effect_emb_dim + 3)
        # - 連續特徵：雙方 (4+1+3+1) 維，再加上公共觀測 2
        self._mlp_input_dim = (
            profession_emb_dim * 2 +              # 玩家 + 敵方職業
            skill_emb_dim * 2 +                   # 玩家 + 敵方技能 (融合後)
            13 * (effect_emb_dim + 3) * 2 +         # 雙方特效
            (4 + 1 + 3 + 1) * 2 +                 # 雙方連續特徵
            2                                     # 公共觀測
        )

        # MLP 層 (預設兩層，每層 256 單位，可由 model_config 調整)
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
        self.effect_id_list = list(range(self.num_effects))  # 特效 ID 0~12

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.logits_layer.in_features
        return [torch.zeros(hidden_size), torch.zeros(hidden_size)]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """轉接到 forward_rnn"""
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        B = inputs.shape[0]
        h_in, c_in = state

        # 處理狀態批次 (與新版邏輯一致)
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
                raise RuntimeError(f"State batch ({B_state}) must divide input batch ({B})")
        h_in, c_in = h_in.unsqueeze(0), c_in.unsqueeze(0)

        # 1. 取得動作 mask (前 4 維)
        mask = inputs[:, :self.mask_dim]

        # 2. 解析職業 (索引與新版對齊)
        # 玩家職業：obs[4:17] (one-hot)
        profession_p_one_hot = inputs[:, 4:17]
        prof_p_id = torch.argmax(profession_p_one_hot, dim=1)
        prof_p_emb = self.profession_embedding(prof_p_id)

        # 敵方職業：obs[80:93] (one-hot)
        profession_e_one_hot = inputs[:, 80:93]
        prof_e_id = torch.argmax(profession_e_one_hot, dim=1)
        prof_e_emb = self.profession_embedding(prof_e_id)

        # 3. 解析技能 (融合 skill id 與 skill type)
        # 玩家技能：
        #   - 選擇槽位 one-hot: obs[158:162]
        local_skill_p_one_hot = inputs[:, 158:162]
        local_skill_p_id = torch.argmax(local_skill_p_one_hot, dim=1)  # 值介於 0～3
        global_skill_p_id = prof_p_id * 4 + local_skill_p_id   # 每個職業 4 個技能
        # 取得 skill id 的 embedding
        skill_id_emb_p = self.skill_embedding(global_skill_p_id)
        # 取得玩家各技能槽的 skill type (one-hot 轉 id)
        # 分別位於 obs[25:28], [28:31], [31:34], [34:37]
        skill0_type = torch.argmax(inputs[:, 25:28], dim=1)
        skill1_type = torch.argmax(inputs[:, 28:31], dim=1)
        skill2_type = torch.argmax(inputs[:, 31:34], dim=1)
        skill3_type = torch.argmax(inputs[:, 34:37], dim=1)
        all_skill_types = torch.stack([skill0_type, skill1_type, skill2_type, skill3_type], dim=1)  # [B, 4]
        # 根據玩家選擇的技能槽取得對應的 skill type
        selected_skill_type = all_skill_types.gather(1, local_skill_p_id.unsqueeze(1)).squeeze(1)
        skill_type_emb_p = self.skill_type_embedding(selected_skill_type)
        # 融合：串接後線性映射
        skill_p_emb = self.skill_fusion(torch.cat([skill_id_emb_p, skill_type_emb_p], dim=1))

        # 敵方技能：
        #   - 選擇槽位 one-hot: obs[154:158]
        local_skill_e_one_hot = inputs[:, 154:158]
        local_skill_e_id = torch.argmax(local_skill_e_one_hot, dim=1)
        global_skill_e_id = prof_e_id * 4 + local_skill_e_id
        skill_id_emb_e = self.skill_embedding(global_skill_e_id)
        # 敵方 skill type 分別位於 obs[101:104], [104:107], [107:110], [110:113]
        enemy_skill0_type = torch.argmax(inputs[:, 101:104], dim=1)
        enemy_skill1_type = torch.argmax(inputs[:, 104:107], dim=1)
        enemy_skill2_type = torch.argmax(inputs[:, 107:110], dim=1)
        enemy_skill3_type = torch.argmax(inputs[:, 110:113], dim=1)
        all_enemy_skill_types = torch.stack([enemy_skill0_type, enemy_skill1_type, enemy_skill2_type, enemy_skill3_type], dim=1)
        selected_enemy_skill_type = all_enemy_skill_types.gather(1, local_skill_e_id.unsqueeze(1)).squeeze(1)
        skill_type_emb_e = self.skill_type_embedding(selected_enemy_skill_type)
        skill_e_emb = self.skill_fusion(torch.cat([skill_id_emb_e, skill_type_emb_e], dim=1))

        # 4. 解析特效 (索引與新版對齊)
        def process_effects(start_idx, batch_size):
            features = []
            for i in range(self.num_effects):
                eff_id = self.effect_id_list[i]
                emb = self.effect_embedding(torch.tensor([eff_id], device=inputs.device)).expand(batch_size, -1)
                exist = inputs[:, start_idx + i*3]
                stack = inputs[:, start_idx + i*3 + 1]
                remain = inputs[:, start_idx + i*3 + 2]
                features.append(torch.cat([emb,
                                           exist.unsqueeze(1),
                                           stack.unsqueeze(1),
                                           remain.unsqueeze(1)], dim=1))
            return torch.cat(features, dim=1)

        # 玩家特效起始索引 37 (新版)
        effect_p_vec = process_effects(37, B)
        # 敵方特效起始索引 113
        effect_e_vec = process_effects(113, B)

        # 5. 解析連續特徵 (索引與新版對齊)
        cont_p_cooldown    = inputs[:, 162:166]   # 玩家冷卻 (4維)
        cont_p_hp_ratio    = inputs[:, 17:18]     # HP%
        cont_p_multipliers = inputs[:, 19:22]     # 乘子 (3維)
        cont_p_accDmg      = inputs[:, 22:23]     # 累積傷害

        cont_e_cooldown    = inputs[:, 166:170]   # 敵方冷卻
        cont_e_hp_ratio    = inputs[:, 93:94]
        cont_e_multipliers = inputs[:, 95:98]
        cont_e_accDmg      = inputs[:, 98:99]

        cont_pub_obs       = inputs[:, 152:154]   # 公共觀測 (2維)

        cont_features = torch.cat([
            cont_p_cooldown, cont_p_hp_ratio, cont_p_multipliers, cont_p_accDmg,
            cont_e_cooldown, cont_e_hp_ratio, cont_e_multipliers, cont_e_accDmg,
            cont_pub_obs
        ], dim=1)

        # 6. 拼接所有特徵
        combined = torch.cat([
            prof_p_emb, prof_e_emb,
            skill_p_emb, skill_e_emb,
            effect_p_vec, effect_e_vec,
            cont_features
        ], dim=1)

        # 7. 網路前向傳播：MLP -> LSTM -> 輸出層
        mlp_out = self.mlp(combined)
        lstm_in = mlp_out.unsqueeze(1)  # [B, 1, hidden_dim]
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        lstm_out = lstm_out.squeeze(1)  # [B, hidden_dim]

        logits = self.logits_layer(lstm_out)
        masked_logits = logits + (1.0 - mask) * -1e10

        self._features = lstm_out
        return masked_logits, [h_out.squeeze(0), c_out.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self):
        return self.value_branch(self._features).squeeze(1)

    def get_all_embeddings(self):
        """
        取得本模型裡面所有 embedding 的 weight，並以 dict 形式回傳，方便你存成 JSON。
        注意：此方法僅回傳原始 embedding 表，不含融合層參數。
        """
        return {
            "profession": self.profession_embedding.weight.detach().cpu().numpy().tolist(),
            "skill": self.skill_embedding.weight.detach().cpu().numpy().tolist(),
            "effect": self.effect_embedding.weight.detach().cpu().numpy().tolist(),
        }
        
        
class MaskedLSTMNetworkWithAttention(RecurrentNetwork, TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4
        self.real_obs_dim = self.obs_dim - self.mask_dim
        
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )
        
        # 1. 輸入標準化
        self.input_norm = nn.LayerNorm(self.real_obs_dim)
        
        # 2. 更強大的 MLP 架構（包含 LayerNorm、ReLU 與 Dropout）
        fcnet_hiddens = model_config.get("fcnet_hiddens", [512, 512])
        dropout_rate = model_config.get("dropout_rate", 0.1)
        
        mlp_layers = []
        in_size = self.real_obs_dim
        for hidden_size in fcnet_hiddens:
            mlp_layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_out_size = in_size
        
        # 3. 使用雙向 LSTM（輸出維度會翻倍）
        self.lstm = nn.LSTM(
            input_size=mlp_out_size,
            hidden_size=mlp_out_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # 4. 注意力機制
        # embed_dim 設為 mlp_out_size * 2（因雙向 LSTM），必須能被 num_heads 整除
        embed_dim = mlp_out_size * 2
        num_heads = 4
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension (mlp_out_size*2) must be divisible by num_heads (4).")
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        # 5. 強化輸出層
        attention_out_size = embed_dim
        self.logits_layer = nn.Sequential(
            nn.Linear(attention_out_size, attention_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_out_size // 2, num_outputs)
        )
        
        # 6. 強化 Value 分支
        self.value_branch = nn.Sequential(
            nn.Linear(attention_out_size, attention_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_out_size // 2, 1)
        )
        
        self._features = None
        self.max_seq_len = model_config.get("max_seq_len", 10)
        
        # 7. 參數初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        import numpy as np
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif "bias" in name:
                    nn.init.constant_(param, 0)
    
    def get_initial_state(self):
        # 回傳單一樣本的初始狀態，每個 state 為 1D 張量（形狀 [hidden_size]）
        hidden_size = self.lstm.hidden_size
        return [torch.zeros(hidden_size, dtype=torch.float),
                torch.zeros(hidden_size, dtype=torch.float)]
    
    def forward_rnn(self, inputs, state, seq_lens):
        B_input = inputs.size(0)
        h_in, c_in = state  # h_in, c_in 的形狀預期為 [B_state, hidden_size]，通常 B_state==1
        
        B_state = h_in.size(0)
        if B_state != B_input:
            if B_state == 1:
                h_in = h_in.expand(B_input, -1)
                c_in = c_in.expand(B_input, -1)
            elif B_input % B_state == 0:
                repeats = B_input // B_state
                h_in = h_in.repeat(repeats, 1)
                c_in = c_in.repeat(repeats, 1)
            else:
                raise RuntimeError(
                    f"State batch size ({B_state}) cannot be matched to inputs batch size ({B_input})."
                )
        # 現在 h_in, c_in 為 [B, hidden_size]
        # LSTM 要求 state 為 3D： [num_layers, B, hidden_size]
        num_layers = self.lstm.num_layers * (2 if self.lstm.bidirectional else 1)
        h_in = h_in.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_in = c_in.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        
        # 分離 mask 與真實觀測（假設 inputs shape 為 [B, obs_dim]）
        mask = inputs[:, :self.mask_dim]         # [B, mask_dim]
        real_obs = inputs[:, self.mask_dim:]       # [B, real_obs_dim]
        
        # 1. 輸入標準化並經由 MLP 提取特徵
        normalized_obs = self.input_norm(real_obs)
        mlp_out = self.mlp(normalized_obs)           # [B, mlp_out_size]
        
        # 2. 為 LSTM 增加時間維度（單步：T=1）
        lstm_input = mlp_out.unsqueeze(1)            # [B, 1, mlp_out_size]
        
        # 3. LSTM 前向傳播
        lstm_out, (h_out, c_out) = self.lstm(lstm_input, (h_in, c_in))
        # lstm_out: [B, 1, embed_dim]，其中 embed_dim = mlp_out_size * 2（因雙向）
        lstm_out = lstm_out.squeeze(1)                # [B, embed_dim]
        
        # 4. 注意力機制
        # MultiheadAttention 要求輸入 shape 為 [L, B, E]（這裡 L=1）
        attn_input = lstm_out.unsqueeze(0)            # [1, B, embed_dim]
        attention_out, _ = self.attention(attn_input, attn_input, attn_input)
        attention_out = attention_out.squeeze(0)      # [B, embed_dim]
        
        # 5. 計算 logits，並利用 mask（將無效動作 logits 加上極大負值）
        logits = self.logits_layer(attention_out)
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask
        
        self._features = attention_out
        
        # 6. 回傳新的 state：
        # 為與基本架構保持一致，這裡僅回傳最後一層的 state，形狀 [B, hidden_size]
        new_h = h_out[-1]  # 取最後一層
        new_c = c_out[-1]
        return masked_logits, [new_h, new_c]
    
    def forward(self, input_dict, state, seq_lens):
        return self.forward_rnn(input_dict["obs"], state, seq_lens)
    
    def value_function(self):
        assert self._features is not None, "必須先執行 forward()/forward_rnn() 才能取得 value_function!"
        return torch.reshape(self.value_branch(self._features), [-1])
    
    
class MaskedLSTMNetworkV2(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    自訂模型：結合 MLP + LSTM 與動作 Mask 處理
    假設觀測空間 (obs) 前 4 維為 mask，其餘為真實特徵。
    注意：要求動作空間的輸出數 (num_outputs) 與 mask 維度一致。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # 先呼叫 nn.Module 的初始化
        nn.Module.__init__(self)
        # 再呼叫 TorchModelV2 的初始化（RecurrentNetwork 為 mixin）
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        # 定義觀測空間相關維度：前 4 維為 mask，其餘為真實特徵
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4  # 假設前4維為 mask
        self.real_obs_dim = self.obs_dim - self.mask_dim

        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        # -------------------------
        # 參數設定（可透過 model_config 調整）
        # -------------------------
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        self.use_layer_norm = model_config.get("use_layer_norm", True)  # 是否使用 layer normalization
        self.fc_dropout = model_config.get("fc_dropout", 0.1)           # MLP 中的 dropout 機率

        # -------------------------
        # 建立 MLP 層：輸入為 real_obs_dim，依序經過各隱藏層
        # 每層線性層後依序接上 LayerNorm（可選）、ReLU 與 Dropout
        # -------------------------
        mlp_layers = []
        in_size = self.real_obs_dim
        for hidden_size in fcnet_hiddens:
            linear_layer = nn.Linear(in_size, hidden_size)
            # 使用 Kaiming 初始化權重
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
            mlp_layers.append(linear_layer)
            if self.use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(self.fc_dropout))
            in_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_out_size = in_size

        # -------------------------
        # 增加 mask 處理分支：
        # 將原始 mask 映射到與 MLP 輸出相同的維度，
        # 並可選擇加入 layer norm，之後與 MLP 輸出融合。
        # -------------------------
        self.mask_processor = nn.Linear(self.mask_dim, mlp_out_size)
        nn.init.kaiming_normal_(self.mask_processor.weight, nonlinearity='relu')
        if self.use_layer_norm:
            self.mask_layer_norm = nn.LayerNorm(mlp_out_size)
        else:
            self.mask_layer_norm = None

        # -------------------------
        # 建立 LSTM 層
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=mlp_out_size,
            hidden_size=mlp_out_size,
            batch_first=True,
        )

        # -------------------------
        # 建立最終輸出層：映射 LSTM 輸出至動作 logits
        # -------------------------
        self.logits_layer = nn.Linear(mlp_out_size, num_outputs)
        nn.init.kaiming_normal_(self.logits_layer.weight, nonlinearity='linear')

        # -------------------------
        # 建立 Value function 分支
        # -------------------------
        self.value_branch = nn.Linear(mlp_out_size, 1)
        nn.init.kaiming_normal_(self.value_branch.weight, nonlinearity='linear')

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
        mask = inputs[:, :self.mask_dim]        # shape: [B, mask_dim]
        real_obs = inputs[:, self.mask_dim:]      # shape: [B, real_obs_dim]

        # 2. 經過 MLP 產生特徵向量（來自真實觀測）
        mlp_out = self.mlp(real_obs)              # shape: [B, mlp_out_size]

        # 3. 處理 mask 分支，將 mask 映射至相同維度
        mask_feature = self.mask_processor(mask)  # shape: [B, mlp_out_size]
        if self.mask_layer_norm is not None:
            mask_feature = self.mask_layer_norm(mask_feature)

        # 4. 融合 MLP 特徵與 mask 特徵（採用加法融合，亦可改為其他融合方式）
        combined_feature = mlp_out + mask_feature  # shape: [B, mlp_out_size]

        # 5. 為 LSTM 增加時間軸：變為 [B, 1, mlp_out_size]
        lstm_input = combined_feature.unsqueeze(1)

        # 6. 調整 state 的 batch 大小以匹配 inputs
        if B_state != B_input:
            if B_state == 1:
                h_in = h_in.expand(B_input, -1)
                c_in = c_in.expand(B_input, -1)
            elif B_input % B_state == 0:
                repeats = B_input // B_state
                h_in = h_in.repeat(repeats, 1)
                c_in = c_in.repeat(repeats, 1)
            else:
                raise RuntimeError(
                    f"State batch size ({B_state}) cannot be matched to inputs batch size ({B_input})."
                )
        # LSTM 要求的 state shape 為 [num_layers, B, hidden_dim]，所以需要 unsqueeze 第一個維度
        h_in = h_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)

        # 7. LSTM 前向傳播
        lstm_out, [h_out, c_out] = self.lstm(lstm_input, [h_in, c_in])
        lstm_out = lstm_out.squeeze(1)  # 形狀變為 [B, hidden_dim]

        # 8. 計算動作 logits
        logits = self.logits_layer(lstm_out)  # shape: [B, num_outputs]

        # 9. 套用動作 mask（對無效動作 logits 加上極大負值）
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        # 10. 儲存特徵供 value_function() 使用
        self._features = lstm_out

        # 11. 回傳新的 state（將 LSTM 層的 layer 維度去除）
        new_state = [h_out.squeeze(0), c_out.squeeze(0)]
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
        # value_branch 輸出 shape 為 [B, 1]，reshape 為 [B]
        return torch.reshape(self.value_branch(self._features), [-1])
    
    
    
class TransformerMaskedNetworkV2(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    修改版 Transformer 模型：
    - 假設觀測空間 (obs) 前 4 維為 mask，其餘為真實特徵；
    - 輸出動作數必須等於 mask 維度；
    - 利用 Transformer Encoder 與一個記憶緩衝區，使模型能夠「記憶」多步資訊，
      並保持輸入、輸出介面與原始版本一致。
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # 假設 obs 為 1D 張量，前 mask_dim(=4) 為 mask
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4
        self.real_obs_dim = self.obs_dim - self.mask_dim

        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        # 取得 MLP 隱藏層設定
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        # 建立 MLP 前處理層：輸入為真實觀測 (real_obs)
        mlp_layers = []
        in_size = self.real_obs_dim
        for hidden_size in fcnet_hiddens:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            in_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        # 使得 Transformer 的輸入維度與 MLP 輸出對齊
        self.d_model = in_size

        # 定義 Transformer Encoder
        # 採用 batch_first=True，輸入 shape 為 [B, T, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=self.d_model * 2,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 使用 learnable 的位置編碼，大小為 [max_seq_len, d_model]
        self.max_seq_len = model_config.get("max_seq_len", 10)  # 總序列長度（包含當前步）
        # 記憶緩衝區長度為 max_seq_len - 1
        self.memory_length = self.max_seq_len - 1
        self.pos_embedding = nn.Parameter(torch.zeros(self.max_seq_len, self.d_model))

        # 最終輸出層：將 Transformer 輸出映射到動作 logits
        self.logits_layer = nn.Linear(self.d_model, num_outputs)

        # Value function 分支
        self.value_branch = nn.Linear(self.d_model, 1)

        # 儲存 forward 過程中產生的中間特徵（供 value_function 使用）
        self._features = None

    @override(TorchModelV2)
    def get_initial_state(self):
        """
        回傳單一樣本的初始 state。
        注意：RLlib 要求 recurrent state 為扁平化的 2D 張量，因此這裡回傳 shape 為
              [memory_length * d_model] 的單樣本 state。
        """
        # 根據當前 d_model 與 memory_length 決定 state 大小
        return [torch.zeros(self.memory_length * self.d_model, dtype=torch.float32)]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        前向傳播（Transformer 部分）：
        - 輸入 inputs: [B, obs_dim]
        - 使用 state 作為記憶緩衝區（扁平化後，形狀 [B, memory_length * d_model]），
          在此處轉換成 [B, memory_length, d_model]。如果尺寸不符合，則重置記憶。
        - 將當前步特徵與記憶拼接後送入 Transformer Encoder，
          最終取最後一個時間步作為輸出
        - 輸出 masked_logits: [B, num_outputs]
        - 更新 state 為新一輪的記憶緩衝區（保留最近 memory_length 步的資訊，
          並扁平化回 2D，形狀 [B, memory_length * d_model]）
        """
        B = inputs.size(0)

        # 1. 分離 mask 與真實觀測
        mask = inputs[:, :self.mask_dim]         # [B, mask_dim]
        real_obs = inputs[:, self.mask_dim:]       # [B, real_obs_dim]

        # 2. 經由 MLP 產生當前步特徵 (current embedding)
        current_feat = self.mlp(real_obs)          # [B, d_model]

        # 3. 從 state 取出記憶緩衝區；如果 state 不存在或尺寸不符，則重新初始化
        expected_state_size = B * self.memory_length * self.d_model
        if (len(state) == 0 or state[0] is None or 
                state[0].numel() != expected_state_size):
            memory = torch.zeros(B, self.memory_length, self.d_model,
                                 device=inputs.device, dtype=inputs.dtype)
        else:
            memory = state[0].view(B, self.memory_length, self.d_model)

        # 4. 將當前特徵擴展時間軸，變為 [B, 1, d_model]
        current_feat = current_feat.unsqueeze(1)

        # 5. 拼接記憶與當前步：得到整個序列，形狀 [B, T, d_model]，其中 T = memory_length + 1 = max_seq_len
        seq = torch.cat([memory, current_feat], dim=1)

        # 6. 加上位置編碼
        seq = seq + self.pos_embedding.unsqueeze(0)

        # 7. Transformer Encoder 前向傳播
        transformer_out = self.transformer_encoder(seq)  # [B, max_seq_len, d_model]

        # 8. 取最後一個時間步作為最終特徵
        final_feat = transformer_out[:, -1, :]  # [B, d_model]

        # 9. 計算動作 logits，並依據 mask 遮蔽不合法動作
        logits = self.logits_layer(final_feat)
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        # 10. 更新 state：
        # 新 state 為 transformer_out 除去第一個時間步（保留最近 memory_length 步），
        # 並扁平化成 [B, memory_length * d_model] 供 RLlib 使用
        new_memory = transformer_out[:, 1:, :]  # [B, memory_length, d_model]
        new_memory_flat = new_memory.reshape(B, self.memory_length * self.d_model)

        # 儲存中間特徵供 value_function 使用
        self._features = final_feat

        return masked_logits, [new_memory_flat]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 轉交 forward_rnn 處理
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        """
        根據 forward_rnn() 中儲存的特徵計算狀態價值
        """
        assert self._features is not None, "必須先執行 forward()/forward_rnn() 才能取得 value_function!"
        return torch.reshape(self.value_branch(self._features), [-1])



class EnhancedMaskedLSTMNetwork(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    強化版模型：結合多種改進技術
    改進點：
    1. 加入LayerNorm提升訓練穩定性
    2. 使用GELU激活函數
    3. LSTM權重正交初始化 + 遺忘門偏置初始化
    4. 價值函數深度化
    5. 加入Dropout防止過擬合
    6. 自適應特徵縮放
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4  # 假設mask維度為4
        self.real_obs_dim = self.obs_dim - self.mask_dim

        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})"
            )

        # 從配置取得超參數
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        lstm_hidden_size = model_config.get("lstm_hidden_size", 256)
        dropout_rate = model_config.get("dropout_rate", 0.2)

        # -------------------------
        # 改進的MLP結構：LayerNorm + GELU
        # -------------------------
        mlp_layers = []
        in_size = self.real_obs_dim
        for idx, hidden_size in enumerate(fcnet_hiddens):
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.LayerNorm(hidden_size))
            mlp_layers.append(nn.GELU())
            
            # 加入Dropout（除最後一層外）
            if idx < len(fcnet_hiddens)-1:
                mlp_layers.append(nn.Dropout(dropout_rate))
            
            in_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_out_size = in_size

        # -------------------------
        # LSTM層改進：正交初始化 + 遺忘門偏置初始化
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=mlp_out_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            dropout=dropout_rate if len(fcnet_hiddens) > 1 else 0  # 僅在有多層時生效
        )
        
        # 自適應特徵縮放
        self.feature_scale = nn.Parameter(torch.ones(1))
        
        # LSTM權重初始化
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                # 遺忘門偏置初始化為1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # -------------------------
        # 深度價值函數網路
        # -------------------------
        self.value_branch = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        # -------------------------
        # 輸出層改進：權重初始化
        # -------------------------
        self.logits_layer = nn.Linear(lstm_hidden_size, num_outputs)
        nn.init.xavier_uniform_(self.logits_layer.weight)
        nn.init.constant_(self.logits_layer.bias, 0)

        # 狀態追蹤
        self._features = None
        self.max_seq_len = model_config.get("max_seq_len", 20)

    @override(TorchModelV2)
    def get_initial_state(self):
        hidden_size = self.lstm.hidden_size
        return [
            torch.zeros(hidden_size, dtype=torch.float),
            torch.zeros(hidden_size, dtype=torch.float)
        ]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # 分離mask與觀測
        mask = inputs[:, :self.mask_dim]
        real_obs = inputs[:, self.mask_dim:]

        # 特徵提取
        mlp_out = self.mlp(real_obs)
        
        # 特徵縮放
        mlp_out = self.feature_scale * mlp_out
        
        # 調整輸入形狀
        lstm_input = mlp_out.unsqueeze(1)  # [B, 1, mlp_out_size]

        # 狀態處理
        h, c = state
        B_input = inputs.size(0)
        
        # 狀態批次擴展
        if h.size(0) != B_input:
            if h.size(0) == 1:
                h = h.expand(B_input, -1)
                c = c.expand(B_input, -1)
            elif B_input % h.size(0) == 0:
                repeats = B_input // h.size(0)
                h = h.repeat(repeats, 1)
                c = c.repeat(repeats, 1)
            else:
                raise RuntimeError(f"狀態批次大小 {h.size(0)} 無法匹配輸入批次 {B_input}")

        # LSTM前傳
        lstm_out, (h_out, c_out) = self.lstm(
            lstm_input,
            (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())
        )
        lstm_out = lstm_out.squeeze(1)
        
        # 應用Dropout
        lstm_out = F.dropout(lstm_out, p=0.2, training=self.training)

        # 計算logits與mask
        logits = self.logits_layer(lstm_out)
        masked_logits = logits + (1.0 - mask) * -1e10

        # 保存特徵供價值函數使用
        self._features = lstm_out

        return masked_logits, [h_out.squeeze(0), c_out.squeeze(0)]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.forward_rnn(input_dict["obs"], state, seq_lens)

    @override(TorchModelV2)
    def value_function(self):
        return self.value_branch(self._features).squeeze(-1)

    def extra_repr(self) -> str:
        return f"Feature Scale: {self.feature_scale.item():.2f}"
    
    
class TransformerMaskedNetwork(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    以 Transformer Encoder 取代 LSTM 的示範模型。
    假設觀測空間 (obs) 前 4 維為 mask，其餘為真實特徵，
    且動作輸出數必須等於 mask 維度。
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # 假設 obs 為 1D 張量，前 4 維為 mask
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4
        self.real_obs_dim = self.obs_dim - self.mask_dim

        # 檢查動作輸出數必須與 mask 維度相同
        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        # 取得 MLP 隱藏層設定
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])

        # 建立 MLP 前處理層：輸入為真實觀測(real_obs)
        mlp_layers = []
        in_size = self.real_obs_dim
        for hidden_size in fcnet_hiddens:
            mlp_layers.append(nn.Linear(in_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            in_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_out_size = in_size

        # 定義 Transformer Encoder
        # d_model 必須與 MLP 輸出維度對齊
        self.d_model = mlp_out_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=mlp_out_size * 2,
            activation="relu",
            batch_first=True,  # 保證輸入形狀為 [B, T, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 簡易可學習位置編碼（本範例僅處理單步輸入，因此序列長度為 1）
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 最終輸出層：將 Transformer 輸出映射到動作 logits
        self.logits_layer = nn.Linear(self.d_model, num_outputs)

        # Value function 分支
        self.value_branch = nn.Linear(self.d_model, 1)

        # 儲存 forward() 過程中的中間特徵，供 value_function() 使用
        self._features = None

    @override(TorchModelV2)
    def get_initial_state(self):
        """
        Transformer 模型不使用傳統 RNN 的隱藏狀態，
        但必須符合 RecurrentNetwork 介面，因此回傳空列表。
        """
        return [torch.zeros(1)]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        # 確保 inputs 為 torch.Tensor，且至少 2D
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        # print("inputs shape:", inputs.shape)

        # 1. 分離 mask 與真實觀測
        mask = inputs[:, :self.mask_dim]
        real_obs = inputs[:, self.mask_dim:]

        # 2. 經由 MLP 產生特徵向量
        mlp_out = self.mlp(real_obs)  # [B, mlp_out_size]

        # 3. 增加時間維度，變為 [B, 1, d_model]
        features = mlp_out.unsqueeze(1)

        # 4. 加上位置編碼
        features = features + self.pos_embedding  # 廣播到 [B, 1, d_model]

        # 5. Transformer Encoder 前向傳播
        transformer_out = self.transformer_encoder(features)  # [B, 1, d_model]

        # 6. 移除時間維度：改用顯式索引取得時間步 0 的輸出，避免 squeeze() 導致降維過度
        final_feat = transformer_out[:, 0, :]

        # 7. 計算動作 logits
        logits = self.logits_layer(final_feat)

        # 8. 套用動作 mask：無效動作 logits 加上極大負值
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        # 9. 儲存中間特徵以供 value_function() 使用
        self._features = final_feat

        # 10. Transformer 無隱藏狀態，回傳空列表
        return masked_logits, []

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        """
        根據 forward_rnn() 儲存的特徵計算狀態價值
        """
        assert self._features is not None, "必須先執行 forward()/forward_rnn() 才能取得 value_function!"
        # 將 [B, 1] 輸出 reshape 成 [B]
        return torch.reshape(self.value_branch(self._features), [-1])

        
class HRLMaskedLSTMNetwork(RecurrentNetwork, TorchModelV2, nn.Module):
    """
    簡易的兩層階層式 RL 示範。
    - Meta LSTM 產生一個 subgoal embedding
    - Low-level LSTM 根據 (real_obs + subgoal_emb) 輸出最終動作 logits
    - 最後再套用 mask
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # 假設 obs 空間為 1D 張量，前 4 維為 mask
        self.obs_dim = obs_space.shape[0]
        self.mask_dim = 4
        self.real_obs_dim = self.obs_dim - self.mask_dim

        if num_outputs != self.mask_dim:
            raise ValueError(
                f"num_outputs ({num_outputs}) 必須等於 mask 維度 ({self.mask_dim})，請檢查觀測設計。"
            )

        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        hidden_size = fcnet_hiddens[-1] if len(fcnet_hiddens) > 0 else 256

        # ---- Meta LSTM (上層) ----
        # 先做 MLP，再接一個 LSTM 產生 subgoal embedding
        meta_mlp_layers = []
        in_size = self.real_obs_dim
        for hs in fcnet_hiddens:
            meta_mlp_layers.append(nn.Linear(in_size, hs))
            meta_mlp_layers.append(nn.ReLU())
            in_size = hs
        self.meta_mlp = nn.Sequential(*meta_mlp_layers)

        self.meta_lstm = nn.LSTM(
            input_size=in_size, 
            hidden_size=in_size,
            batch_first=True
        )
        # subgoal 大小可自訂，這裡簡單用和 hidden_size 相同
        self.subgoal_size = in_size

        # ---- Low-level LSTM (下層) ----
        # 輸入為 (real_obs_dim + subgoal_size)
        low_mlp_in = self.real_obs_dim + self.subgoal_size
        self.low_lstm = nn.LSTM(
            input_size=low_mlp_in,
            hidden_size=low_mlp_in,
            batch_first=True
        )

        # 最終動作 logits
        self.logits_layer = nn.Linear(low_mlp_in, num_outputs)

        # Value branch
        self.value_branch = nn.Linear(low_mlp_in, 1)

        # 儲存中間特徵
        self._features = None

    @override(TorchModelV2)
    def get_initial_state(self):
        """
        需要同時維護 Meta-LSTM 與 Low-level LSTM 的隱藏狀態。
        """
        meta_hidden_size = self.subgoal_size
        low_hidden_size = self.real_obs_dim + self.subgoal_size

        # meta_lstm state
        meta_h0 = torch.zeros(meta_hidden_size, dtype=torch.float)
        meta_c0 = torch.zeros(meta_hidden_size, dtype=torch.float)

        # low_lstm state
        low_h0 = torch.zeros(low_hidden_size, dtype=torch.float)
        low_c0 = torch.zeros(low_hidden_size, dtype=torch.float)

        return [meta_h0, meta_c0, low_h0, low_c0]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        HRL 分為兩段：
        1) Meta LSTM => subgoal embedding
        2) Low-level LSTM => final logits
        """
        mask = inputs[:, :self.mask_dim]
        real_obs = inputs[:, self.mask_dim:]

        B_input = inputs.size(0)

        # 拆解隱藏層
        meta_h_in, meta_c_in, low_h_in, low_c_in = state
        B_state_meta = meta_h_in.size(0)
        B_state_low = low_h_in.size(0)

        # 1) Meta MLP -> Meta LSTM
        meta_mlp_out = self.meta_mlp(real_obs)
        meta_lstm_in = meta_mlp_out.unsqueeze(1)  # [B, 1, hidden_size]

        # 對齊 batch size
        if B_input != B_state_meta:
            # 只示範簡單處理，概念同前
            if B_state_meta == 1:
                meta_h_in = meta_h_in.expand(B_input, -1)
                meta_c_in = meta_c_in.expand(B_input, -1)
            else:
                repeats = B_input // B_state_meta
                meta_h_in = meta_h_in.repeat(repeats, 1)
                meta_c_in = meta_c_in.repeat(repeats, 1)

        meta_h_in = meta_h_in.unsqueeze(0)  # [1, B, hidden_size]
        meta_c_in = meta_c_in.unsqueeze(0)

        meta_out, [meta_h_out, meta_c_out] = self.meta_lstm(meta_lstm_in, [meta_h_in, meta_c_in])
        meta_out = meta_out.squeeze(1)  # [B, hidden_size] -> subgoal_emb

        # 2) Low-level LSTM
        #    將 (real_obs, subgoal_emb) 拼接
        low_input = torch.cat([real_obs, meta_out], dim=1)  # [B, real_obs_dim + subgoal_size]
        low_input = low_input.unsqueeze(1)                 # [B, 1, ...]

        if B_input != B_state_low:
            if B_state_low == 1:
                low_h_in = low_h_in.expand(B_input, -1)
                low_c_in = low_c_in.expand(B_input, -1)
            else:
                repeats = B_input // B_state_low
                low_h_in = low_h_in.repeat(repeats, 1)
                low_c_in = low_c_in.repeat(repeats, 1)

        low_h_in = low_h_in.unsqueeze(0)
        low_c_in = low_c_in.unsqueeze(0)

        low_out, [low_h_out, low_c_out] = self.low_lstm(low_input, [low_h_in, low_c_in])
        low_out = low_out.squeeze(1)  # [B, low_mlp_in]

        # logits
        logits = self.logits_layer(low_out)

        # mask
        inf_mask = (1.0 - mask) * -1e10
        masked_logits = logits + inf_mask

        # 儲存特徵給 value_function 使用
        self._features = low_out

        # 新 state
        new_state = [
            meta_h_out.squeeze(0),
            meta_c_out.squeeze(0),
            low_h_out.squeeze(0),
            low_c_out.squeeze(0),
        ]
        return masked_logits, new_state

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits, new_state = self.forward_rnn(obs, state, seq_lens)
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "必須先執行 forward() 才能取得 value_function!"
        return torch.reshape(self.value_branch(self._features), [-1])
    