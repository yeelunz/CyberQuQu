# freeze_effects.py

import random

def try_trigger_freeze(user, target):
    """
    當 user 對 target 造成攻擊時，如果 target 有 freeze層數>0，
    則有 15%*freeze_layer 機率 觸發「行動被封鎖1回合」 => target["skip_turn"]=True
    並移除 1 層 freeze。
    """
    freeze_layers = target["status"].get("freeze", 0)
    if freeze_layers > 0:
        chance = 0.15 * freeze_layers
        if random.random() < chance:
            # 觸發凍結 => 下回合跳過行動
            target["skip_turn"] = True
            # 移除一層
            target["status"]["freeze"] = freeze_layers - 1
            if target["status"]["freeze"] <= 0:
                del target["status"]["freeze"]
