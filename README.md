# 專案簡介

本專案利用 [RLlib](https://docs.ray.io/en/latest/rllib.html) ，針對簡易回合制對戰環境訓練強化學習模型。專案內建多款預設模型及簡單的前端介面，讓你能快速上手模型訓練，同時附有基本測試及分析指標，協助你全面掌握模型在回合制對戰環境中的表現。

---

# 環境設置

建議使用 [Conda](https://docs.conda.io/en/latest/) 管理專案環境，請依照下列步驟操作：

```bash
# 建立並啟動新的 Conda 環境（Python 版本 3.10.5）
conda create --name CyberQuQu python=3.10.5
conda activate CyberQuQu

# 進入專案資料夾並安裝必要套件
cd [專案資料夾路徑]
pip install -r requirements.txt
```

---

# 執行專案

安裝完成後，即可使用以下指令啟動專案進行測試：

```bash
python run.py
```

---

# 資料管理說明

目前前端尚未實作文件刪除或重新命名功能，若需進行資料管理，請直接操作以下目錄中的檔案（例如改名或刪除）：

- **模型**：`data/saved_models`
- **模型對戰資料 (模型 vs 模型)**：`data/model_vs`
- **模型對戰環境評估**：`data/cross_validation_ai`
- **PC 對戰環境評估**：`data/cross_validation_pc`

---

# 常見問題與解決方案

1. **CUDA 與 Torch 版本設定**

   如果你已安裝 CUDA 並使用對應的 Torch 版本，請到 `config/gv.json` 中將 `num_gpus_per_env_runner` 設定為 `1`：

   ```json
   "num_gpus_per_env_runner": 1
   ```

2. **Sample Timeout 問題**

   如遇到 sample timeout，可於 `config/gv.json` 中調整 `sameple_time_out_s` 參數，例如：

   ```json
   "sameple_time_out_s": 240
   ```

3. **預設模型說明**

   專案中提供三個訓練完成的預設模型，供你與自訓模型進行對比與評估。

   > **注意**：這些模型均基於初始數值訓練，環境變更可能影響其表現。若需還原至初始環境，請使用 `config/config_default` 中的 `profession_var.json` 覆蓋 `config/profession_var.json`。

   | 模型名稱 | ELO 評級 |
   | -------- | -------- |
   | level 1  | 約 1500  |
   | level 2  | 約 1630  |
   | level 3  | 約 1720  |

4. **是否可以自行新增職業？**

   新增職業較為複雜，因為涉及模型輸出觀測空間的重新設計。若希望新增職業，可嘗試將現有不必要的職業進行修改，但同時需要調整 `profession_var` 及其他相關部分，工作量會較大。

5. **如何調整其他訓練超參數？**

   前端僅提供基本超參數調整。如你熟悉相關流程或希望進一步調整超參數，請參考 `utils/train_methods.py` 中的 `multi_agent_cross_train` 函式進行設定。

---

# 回合制對戰流程

此回合制對戰環境的基本流程如下（先攻與後攻順序為隨機決定，一般不會造成顯著差異）：

1. **回合開始**
2. **雙方同時選擇技能**
3. **環境模擬技能使用效果**
4. **回合結束**

---

# 預設模型介紹

下表列出了本專案提供的各種預設模型，說明其基礎架構、主要組件及適用場景。

> **提醒**：以下描述由大語言模型生成，僅供參考；實際模型表現可能略有出入。  
>
> 若你希望開發及使用自定義模型，請在 `utils/train_method.py` 中透過以下方式註冊（或建議直接修改現有模型架構）：
>
> ```python
> ModelCatalog.register_custom_model("your_register_model_name", your_register_model_class)
> ```

| 模型名稱                             | 基礎架構               | 關鍵組件                                             | 嵌入處理                  | 注意力機制 | 狀態處理             | 共享嵌入 | 適用場景                         | 備註                                       |
| ------------------------------------ | ---------------------- | ---------------------------------------------------- | ------------------------- | ---------- | -------------------- | -------- | ---------------------------------- | ------------------------------------------ |
| **MaskedLSTMNetwork**                | MLP + LSTM             | 基礎 LSTM、動作 mask 處理                             | 無                        | 無         | 單向 LSTM 狀態       | 無       | 基礎 RL 場景                      | 假設觀測前 4 維為 mask                        |
| **MaskedLSTMNetworkV2**              | MLP + LSTM             | Mask 處理分支、特徵融合                               | 無                        | 無         | 單向 LSTM 狀態       | 無       | 無需 mask 特徵強化的場景           | 採用獨立 mask 處理                          |
| **MaskedLSTMNetworkWithEmb**         | MLP + LSTM             | 職業/技能/特效嵌入、分離玩家/敵方嵌入表                | 獨立嵌入表（玩家/敵方）    | 無         | 單向 LSTM 狀態       | 否       | 包含類別特徵的複雜觀測             | 需要手動進行特徵工程                          |
| **MaskedLSTMNetworkWithEmbV2**       | MLP + LSTM             | 職業嵌入融合額外屬性、技能類型融合                   | 獨立嵌入表 + 線性融合      | 無         | 單向 LSTM 狀態       | 否       | 需要特徵融合的場景                 | 改進了嵌入資訊利用率                          |
| **MaskedLSTMNetworkWithMergedEmb**   | MLP + LSTM             | 統一玩家/敵方嵌入表                                   | 共享嵌入表                | 無         | 單向 LSTM 狀態       | 是       | 針對減少參數量的類別特徵場景        | 索引需與新版觀測對齊                          |
| **MaskedLSTMNetworkWithMergedEmbV2** | MLP + LSTM             | 技能類型融合、共享嵌入                               | 共享嵌入表 + 技能類型融合   | 無         | 單向 LSTM 狀態       | 是       | 平衡參數量與特徵表達的場景         | 包含技能類型融合層                           |
| **MaskedLSTMNetworkWithAttention**   | MLP + BiLSTM + Attention | 雙向 LSTM、多頭注意力、LayerNorm                     | 無                        | 有         | 雙向 LSTM 狀態       | 無       | 適用於長序列依賴的場景             | 採用正交初始化與自適應特徵縮放                |
| **TransformerMaskedNetwork**         | MLP + Transformer      | Transformer Encoder、可學習位置編碼                   | 無                        | 有         | 無狀態/記憶緩衝區    | 無       | 用於全局依賴建模                  | 不使用傳統 RNN 狀態                           |
| **TransformerMaskedNetworkV2**       | MLP + Transformer      | Transformer Encoder、可學習位置編碼                   | 無                        | 有         | 有狀態/記憶緩衝區    | 無       | 用於全局依賴建模                  | 不使用傳統 RNN 狀態                           |
| **EnhancedMaskedLSTMNetwork**        | MLP + 強化 LSTM         | GELU 激活、LSTM 正交初始化、深度價值網路               | 無                        | 無         | 單向 LSTM 狀態       | 無       | 複雜場景中的穩定訓練              | 包含遺忘門偏置初始化與特徵縮放                 |
| **HRLMaskedLSTMNetwork**             | 階層式 LSTM            | Meta LSTM 生成 subgoal、Low-level LSTM 執行           | 無                        | 無         | 雙層 LSTM 狀態       | 無       | 適用於階層決策任務                | 兩層 LSTM 協同運作                           |
