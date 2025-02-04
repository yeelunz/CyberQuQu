from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
import torch
from ray.rllib.models.catalog import ModelCatalog


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
