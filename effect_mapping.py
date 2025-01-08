# effect_mapping.py

# 定義效果ID到觀察向量索引的映射
# 每個效果佔用3個位置：[存在 (0或1), 堆疊數, 剩餘持續回合]
# 自定義效果的效果ID 不會有mapping來處理
# 因為自定義效果依靠source來區分不同狀態
EFFECT_MAPPING = {
    1: 0,   # DamageMultiplier
    2: 3,   # DefenseMultiplier
    3: 6,   # HealMultiplier
    4: 9,   # Burn
    5: 12,  # Poison
    6: 15,  # Freeze
    7: 18,  # ImmuneDamage
    8: 21,  # ImmuneControl
    9: 24,  # BleedEffect
    10: 27, # 暈眩
    11: 30, # HP Recovery
    12: 33, # Max HP Increase

    
    98: 39, # CustomEffect Dot
    999: 42, # Track
}

# 99: 39, # CustomEffect
# 100: 42, # CustomEffect

# 計算觀察向量中效果相關部分的總長度
EFFECT_VECTOR_LENGTH = max(EFFECT_MAPPING.values()) + 3  # 每個效果佔3個位置
