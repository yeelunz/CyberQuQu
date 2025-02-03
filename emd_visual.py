import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 使用 Windows 的中文字體 "Microsoft JhengHei"
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 1. 建立映射表
# 技能 ID 與名稱的映射
skill_id_to_name = {
    0: "聖光斬",
    1: "堅守防禦",
    2: "神聖治療",
    3: "火焰之球",
    4: "冰霜箭",
    5: "全域爆破",
    6: "致命暗殺",
    7: "毒爆",
    8: "毒刃襲擊",
    9: "五連矢",
    10: "箭矢補充",
    11: "吸血箭",
    12: "狂暴之力",
    13: "熱血",
    14: "血怒之泉",
    15: "神龍之息",
    16: "龍血之泉",
    17: "神龍燎原",
    18: "血刀",
    19: "血脈祭儀",
    20: "轉生",
    21: "剛毅打擊",
    22: "不屈意志",
    23: "絕地反擊",
    24: "吞裂",
    25: "巨口吞世",
    26: "堅硬皮膚",
    27: "續戰攻擊",
    28: "埋伏防禦",
    29: "荒原抗性",
    30: "雷霆護甲",
    31: "凍燒雷",
    32: "雷擊術",
    33: "枯骨",
    34: "荒原",
    35: "生命逆流",
    36: "災厄隕星",
    37: "光輝流星",
    38: "虛擬創星圖"
}

# 職業 ID 與名稱的映射
profession_id_to_name = {
    0: "聖騎士",
    1: "法師",
    2: "刺客",
    3: "弓箭手",
    4: "狂戰士",
    5: "龍神",
    6: "血神",
    7: "剛毅武士",
    8: "鯨吞",
    9: "荒原遊俠",
    10: "元素法師",
    11: "荒神",
    12: "星神"
}

# -----------------------------
# 2. 讀取 JSON 檔案，取得嵌入權重
with open("data/saved_models/vis_test/embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# print this embedding data's size and each key's size
print("data size: ", len(data))
for key in data:
    print(key, "size: ", len(data[key]))
    # and th
    

profession_embeds_p = np.array(data["profession_p"])  # shape [13, emb_dim]
profession_embeds_e = np.array(data["profession_e"])  # shape [13, emb_dim]
skill_embeds_p = np.array(data["skill_p"])            # shape [39, emb_dim]
skill_embeds_e = np.array(data["skill_e"])            # shape [39, emb_dim]

# -----------------------------
# 3. t-SNE 降維 (2D)
# 職業：合併我方與敵方 (共26筆)
combined_profession = np.concatenate([profession_embeds_p, profession_embeds_e], axis=0)
tsne_profession = TSNE(n_components=2, perplexity=5, random_state=42)
profession_2d = tsne_profession.fit_transform(combined_profession)

# 技能：合併我方與敵方 (共78筆)
combined_skill = np.concatenate([skill_embeds_p, skill_embeds_e], axis=0)
tsne_skill = TSNE(n_components=2, perplexity=30, random_state=42)
skill_2d = tsne_skill.fit_transform(combined_skill)

# -----------------------------
# 4. 視覺化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子圖1：職業嵌入
axes[0].scatter(profession_2d[:13, 0], profession_2d[:13, 1],
                c='b', s=50, label='我方職業')
for i in range(13):
    name = profession_id_to_name.get(i, str(i))
    axes[0].text(profession_2d[i, 0], profession_2d[i, 1],
                 name, fontsize=10)
axes[0].scatter(profession_2d[13:, 0], profession_2d[13:, 1],
                c='r', s=50, label='敵方職業')
for i in range(13, 26):
    name = profession_id_to_name.get(i-13, str(i-13))
    axes[0].text(profession_2d[i, 0], profession_2d[i, 1],
                 name, fontsize=10)
axes[0].set_title("職業 Embeddings (t-SNE 2D)")
axes[0].set_xlabel("維度 1")
axes[0].set_ylabel("維度 2")
axes[0].legend()

# 子圖2：技能嵌入
axes[1].scatter(skill_2d[:39, 0], skill_2d[:39, 1],
                c='g', s=50, label='我方技能')
for i in range(39):
    name = skill_id_to_name.get(i, str(i))
    axes[1].text(skill_2d[i, 0], skill_2d[i, 1],
                 name, fontsize=8)
axes[1].scatter(skill_2d[39:, 0], skill_2d[39:, 1],
                c='orange', s=50, label='敵方技能')
for i in range(39, 78):
    name = skill_id_to_name.get(i-39, str(i-39))
    axes[1].text(skill_2d[i, 0], skill_2d[i, 1],
                 name, fontsize=8)
axes[1].set_title("技能 Embeddings (t-SNE 2D)")
axes[1].set_xlabel("維度 1")
axes[1].set_ylabel("維度 2")
axes[1].legend()

plt.tight_layout()
plt.show()
