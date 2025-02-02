# profession_var.py is used to store the default value of each profession 

# 基本調用格式為
# PALADIN_VAR['PALADIN_BASE_HP'][0] <= current value：目前的數值
# PALADIN_VAR['PALADIN_BASE_HP'][1] <= default value：預設的數值
# PALADIN_VAR['PALADIN_BASE_HP'][2] <= lowwer bound：沒有指定=0
# PALADIN_VAR['PALADIN_BASE_HP'][3] <= upper bound：沒有指定=0
# PALADIN_VAR['PALADIN_BASE_HP'][4] <= 用來判斷要越高越好或是越小越好
# 若是PALADIN_VAR['PALADIN_BASE_HP'][4]='high'
# =>則代表說進行buff時需要增加這個數值
# 若是PALADIN_VAR['PALADIN_BASE_HP'][4]='low'
# =>則代表說進行buff時需要減少這個數值
# 若是PALADIN_VAR['PALADIN_BASE_HP'][4]='nd'
# =>則代表說這個數值有著不明確的變化
# 若是PALADIN_VAR['PALADIN_BASE_HP'][4]='no'
# =>則代表說這個數值不應該被更動
# 如果不存在[4]或是[4]='no'，則代表說這個數值不應該被更動。不要更動這個數值

PALADIN_VAR = {
    # BASE STATS
    'PALADIN_BASE_HP': [397, 397, 100, 999, 'high'],
    'PALADIN_BASE_ATK': [1.1, 1.1, 0.1, 999, 'high'],
    'PALADIN_BASE_DEF': [1.1, 1.1, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'PALADIN_PASSIVE_TRIGGER_RATE': [0.3, 0.3, 0.01, 1, 'high'],
    # PASSIVE HEAL AMOUNT RATE
    'PALADIN_PASSIVE_HEAL_RATE': [0.15, 0.15, 0.01, 1, 'high'],
    # OVERHEADLING DAMAGE RATE
    'PALADIN_PASSIVE_OVERHEADLINGE_RATE': [1, 1, 0.01, 999, 'high'],

    # SKILL 1 DAMAGE AMOUNT
    'PALADIN_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    
    # PALADING SKILL 2 HEAL AMOUNT AND COOLDOWN
    'PALADIN_SKILL_1_HEAL': [15, 15, 1, 999, 'high'],
    'PALADIN_SKILL_1_COOLDOWN': [3, 3, 1, 999, 'low'],

    # SKILL 3 DAMAGE AMOUNT
    # FIRST HEAL AMOUNT FIRST/SECOND/or more
    'PALADIN_SKILL_2_FIRST_HEAL': [60, 50, 1, 999, 'high'],
    'PALADIN_SKILL_2_SECOND_HEAL': [45, 35, 1, 999, 'high'],
    'PALADIN_SKILL_2_MORE_HEAL': [25, 15, 1, 999, 'high'],

}

MAGE_VAR = {
    # BASE STATS
    'MAGE_BASE_HP': [249, 249, 100, 999, 'high'],
    'MAGE_BASE_ATK': [1.65, 1.65, 0.1, 999, 'high'],
    'MAGE_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'MAGE_PASSIVE_TRIGGER_RATE': [0.5, 0.5, 0.01, 1, 'high'],

    # SKILL 3 DAMAGE AMOUNT AND BURN DAMAGE
    'MAGE_SKILL_0_DAMAGE': [35, 35, 1, 999, 'high'],
    'MAGE_SKILL_0_BURN_DAMAGE': [5, 5, 1, 999, 'high'],

    # SKILL 4 DAMAGE AMOUNT
    'MAGE_SKILL_1_DAMAGE': [35, 35, 1, 999, 'high'],

    # SKILL 5 DAMAGE AND STATUS MULTIPLIER
    'MAGE_SKILL_2_BASE_DAMAGE': [25, 25, 1, 999, 'high'],
    'MAGE_SKILL_2_STATUS_MULTIPLIER': [40, 40, 1, 999, 'high'],
}

ASSASSIN_VAR = {
    # BASE STATS
    'ASSASSIN_BASE_HP': [279, 279, 100, 999, 'high'],
    'ASSASSIN_BASE_ATK': [1.2, 1.2, 0.1, 999, 'high'],
    'ASSASSIN_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'ASSASSIN_PASSIVE_TRIGGER_RATE': [0.35, 0.35, 0.01, 1, 'high'],

    # PASSIVE BONUS DAMAGE RATE
    'ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE': [0.05, 0.05, 0.01, 1, 'high'],

    # SKILL 0 DAMAGE AMOUNT AND CRIT RATE
    'ASSASSIN_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    'ASSASSIN_SKILL_0_CRIT_RATE': [0.35, 0.35, 0.01, 1, 'high'],

    # SKILL 1 DAMAGE PER POISON LAYER AND HEAL PER LAYER
    'ASSASSIN_SKILL_1_DAMAGE_PER_LAYER': [20, 20, 1, 999, 'high'],
    'ASSASSIN_SKILL_1_HEAL_PER_LAYER': [20, 20, 1, 999, 'high'],

    # SKILL 2 DAMAGE AMOUNT AND POISON STACKS
    'ASSASSIN_SKILL_2_DAMAGE': [15, 15, 1, 999, 'high'],
    'ASSASSIN_SKILL_2_POISON_STACKS_1_WEIGHT': [4,4,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_2_WEIGHT': [3,3,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_3_WEIGHT': [2,2,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_4_WEIGHT': [1,1,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_5_WEIGHT': [1,1,0,999,'nd'],
    
    'ASSASSIN_SKILL_2_POISON_DAMAGE': [3, 3, 1, 999, 'high'],
}

ARCHER_VAR = {
    # BASE STATS
    'ARCHER_BASE_HP': [275, 275, 100, 999, 'high'],
    'ARCHER_BASE_ATK': [1.12, 1.12, 0.1, 999, 'high'],
    'ARCHER_BASE_DEF': [1.03, 1.03, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'ARCHER_PASSIVE_BASE_TRIGGER_RATE': [0.05, 0.05, 0.01, 1, 'high'],
    'ARCHER_PASSIVE_TRIGGER_RATE_BONUS': [2.0, 2.0, 0.1, 5, 'high'],  # 每單位防禦力增加的觸發率
    'ARCHER_PASSIVE_TRIGGER_RATE_MAX': [0.5, 0.5, 0.1, 1, 'high'],  # 最大觸發率
    # 觸發後的傷害倍率
    'ARCHER_PASSIVE_DAMAGE_MULTIPLIER': [2, 2, 0.1, 5, 'high'],

    # SKILL 9 DAMAGE AND DEFENSE DEBUFF
    'ARCHER_SKILL_0_DAMAGE': [60, 60, 1, 999, 'high'],
    'ARCHER_SKILL_0_DEFENSE_DEBUFF': [0.895, 0.895, 0.1, 1, 'low'],  # 防禦力降低比例
    'ARCHER_SKILL_0_DURATION': [2, 2, 1, 5, 'high'],

    # SKILL 10 MULTIPLIERS AND DURATIONS
    'ARCHER_SKILL_1_DAMAGE_MULTIPLIER': [3.5, 3.5, 1, 5, 'high'],  # 提升攻擊倍率
    'ARCHER_SKILL_1_DEFENSE_MULTIPLIER': [0.5, 0.5, 0.1, 1, 'high'],  # 降低防禦倍率
    'ARCHER_SKILL_1_DURATION': [2, 2, 1, 5, 'high'],
    'ARCHER_SKILL_1_SUCESS_RATIO': [0.6, 0.6, 0.01, 1, 'high'],  # 成功率

    # SKILL 11 DAMAGE AND HEAL
    'ARCHER_SKILL_2_DAMAGE': [35, 35, 1, 999, 'high'],
    'ARCHER_SKILL_2_HEAL': [20, 20, 1, 999, 'high'],
}

BERSERKER_VAR = {
    # BASE STATS
    'BERSERKER_BASE_HP': [395, 395, 100, 999, 'high'],
    'BERSERKER_BASE_ATK': [1.15, 1.15, 0.1, 999, 'high'],
    'BERSERKER_BASE_DEF': [0.8, 0.8, 0.1, 999, 'high'],

    # PASSIVE EXTRA DAMAGE RATE
    'BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE': [0.35, 0.35, 0.01, 1, 'high'],
    # 這邊是用來判斷是否要進行額外傷害的門檻值
    'BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD': [0.6, 0.6, 0.01, 1, 'high'],

    # SKILL 0 DAMAGE AND SELF-MUTILATION
    'BERSERKER_SKILL_0_DAMAGE': [30, 30, 1, 999, 'high'],
    'BERSERKER_SKILL_0_SELF_MUTILATION_RATE': [0.15, 0.15, 0.01, 1, 'low'],

    # SKILL 1 HP COST AND HEAL OVER TIME
    'BERSERKER_SKILL_1_HP_COST': [150, 150, 1, 999, 'low'],
    'BERSERKER_SKILL_1_HEAL_PER_TURN': [40, 40, 1, 999, 'high'],
    'BERSERKER_SKILL_1_DURATION': [5, 5, 1, 999, 'no'],
    # cool down 必須至少是'BERSERKER_SKILL_1_DURATION'+1
    'BERSERKER_SKILL_1_COOLDOWN': [5, 5, 1, 999, 'no'],

    # SKILL 2 HP COST, DEFENSE BUFF, AND CONTROL IMMUNITY
    'BERSERKER_SKILL_2_HP_COST': [30, 30, 1, 999, 'low'],
    'BERSERKER_SKILL_2_DEFENSE_BUFF': [1.35, 1.35, 0.1, 999, 'high'],
    'BERSERKER_SKILL_2_DURATION': [2, 2, 1, 999, 'high'],
}

DRAGONGOD_VAR = {
    # BASE STATS
    'DRAGONGOD_BASE_HP': [314, 314, 100, 999, 'high'],
    'DRAGONGOD_BASE_ATK': [1.0, 1.0, 0.1, 999, 'high'],
    'DRAGONGOD_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE EFFECT
    'DRAGONGOD_PASSIVE_ATK_MULTIPLIER': [1.05, 1.05, 0.01, 2, 'high'],
    'DRAGONGOD_PASSIVE_DEF_MULTIPLIER': [1.05, 1.05, 0.01, 2, 'high'],

    # SKILL 0 DAMAGE
    'DRAGONGOD_SKILL_0_BASE_DAMAGE': [25, 25, 1, 999, 'high'],
    'DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK': [10, 10, 1, 999, 'high'],

    # SKILL 1 HEAL AND BLEED
    'DRAGONGOD_SKILL_1_HEAL_AMOUNT': [120, 120, 1, 999, 'high'],
    'DRAGONGOD_SKILL_1_BLEED_PER_TURN': [30, 30, 1, 999, 'high'],
    'DRAGONGOD_SKILL_1_BLEED_DURATION': [3, 3, 1, 999, 'no'],
    'DRAGONGOD_SKILL_1_COOLDOWN': [4, 4, 1, 999, 'no'],

    # SKILL 2 DAMAGE AND STACK CONSUMPTION
    'DRAGONGOD_SKILL_2_DAMAGE_PER_STACK': [45, 45, 1, 999, 'high'],
    # 這邊是消耗的疊加層數比例 如果是0.5代表消耗一半
    'DRAGONGOD_SKILL_2_STACK_CONSUMPTION': [0.5, 0.5, 0.1, 1, 'nd'],
}


BLOODGOD_VAR = {
    # BASE STATS
    'BLOODGOD_BASE_HP': [266, 266, 100, 999, 'high'],
    'BLOODGOD_BASE_ATK': [1.6, 1.6, 0.1, 999, 'high'],
    'BLOODGOD_BASE_DEF': [1.6, 1.6, 0.1, 999, 'high'],

    # PASSIVE EFFECT
    'BLOODGOD_PASSIVE_DAMAGE_THRESHOLD': [0.25, 0.25, 0.01, 1, 'high'],  # 每受到最大血量的 25% 傷害
    'BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION': [0.895, 0.895, 0.01, 1, 'low'],  # 攻擊、防禦、治癒力降低比例

    # SKILL 18: DAMAGE, BLEED EFFECT, HEAL PER BLEED STACK
    'BLOODGOD_SKILL_0_DAMAGE': [60, 60, 1, 999, 'high'],
    'BLOODGOD_SKILL_0_BLEED_DURATION': [5, 5, 1, 999, 'no'],
    'BLOODGOD_SKILL_0_HEAL_PER_BLEED_STACK': [3, 3, 1, 999, 'high'],

    # SKILL 19: BLEED STACK EFFECTS
    'BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 每層流血降低累積傷害的倍率
    'BLOODGOD_SKILL_1_HEAL_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 每層流血恢復的生命值
    'BLOODGOD_SKILL_1_BLEED_STACK_MULTIPLIER': [2, 2, 1, 999, 'high'],  # 流血層數翻倍

    # SKILL 20: RESURRECTION
    'BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO': [0.15, 0.15, 0.01, 1, 'low'],  # 消耗當前生命值的比例
    'BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO': [0.25, 0.25, 0.01, 1, 'high'],  # 回復最大生命值的比例
    'BLOODGOD_SKILL_2_BLOOD_ACCUMULATION_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 致死累積傷害倍率
    'BLOODGOD_SKILL_2_DURATION': [2, 2, 1, 999, 'no'],  # 轉生效果持續時間
    'BLOODGOD_SKILL_2_COOLDOWN': [5, 5, 1, 999, 'no'],  # 技能冷卻時間
}

STEADFASTWARRIOR_VAR = {
    # BASE STATS
    'STEADFASTWARRIOR_BASE_HP': [253, 253, 100, 999, 'high'],
    'STEADFASTWARRIOR_BASE_ATK': [0.95, 0.95, 0.1, 999, 'high'],
    'STEADFASTWARRIOR_BASE_DEF': [1.2, 1.2, 0.1, 999, 'high'],

    # PASSIVE: HEAL LOST HP PERCENTAGE
    'STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT': [0.07, 0.07, 0.01, 1, 'high'],  # 每回合恢復損失生命值的百分比

    # SKILL 21: DAMAGE AND DEFENSE DEBUFF
    'STEADFASTWARRIOR_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    'STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF': [0.745, 0.745, 0.01, 1, 'low'],  # 防禦降低比例
    'STEADFASTWARRIOR_SKILL_0_DURATION': [3, 3, 1, 5, 'high'],  # 持續回合數

    # SKILL 22: DEFENSE BUFF AND HEAL
    'STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF': [1.3, 1.3, 0.1, 2, 'high'],  # 防禦力增加比例
    'STEADFASTWARRIOR_SKILL_1_DURATION': [1, 1, 1, 5, 'high'],  # 防禦力持續時間
    'STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT': [25, 25, 1, 999, 'high'],  # 恢復的生命值

    # SKILL 23: COUNTER DAMAGE MULTIPLIER
    'STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER': [3, 3, 0.1, 10, 'high'],  # 反擊傷害倍率
    'STEADFASTWARRIOR_SKILL_2_COOLDOWN': [3, 3, 1, 10, 'low'],  # 技能冷卻時間
}


DEVOUR_VAR = {
    # BASE STATS
    'DEVOUR_BASE_HP': [800, 800, 100, 999, 'high'],
    'DEVOUR_BASE_ATK': [1.0, 1.0, 0.1, 999, 'high'],
    'DEVOUR_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE: SELF DAMAGE PERCENTAGE
    'DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT': [0.08, 0.08, 0.01, 1, 'low'],  # 攻擊時消耗的生命值比例

    # SKILL 24: DAMAGE AND FAILURE RATE
    'DEVOUR_SKILL_0_DAMAGE': [65, 65, 1, 999, 'high'],
    'DEVOUR_SKILL_0_FAILURE_RATE': [0.5, 0.5, 0.01, 1, 'low'],  # 技能失敗的機率

    # SKILL 25: DAMAGE BASED ON HP DIFFERENCE
    'DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER': [0.15, 0.15, 0.01, 1, 'high'],  # 已損血量傷害倍率
    'DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER': [0.15, 0.15, 0.01, 1, 'high'],  # 當前血量傷害倍率
    'DEVOUR_SKILL_1_COOLDOWN': [2, 2, 1, 10, 'low'],  # 技能冷卻時間

    # SKILL 26: DEFENSE BUFF
    'DEVOUR_SKILL_2_DEFENSE_MULTIPLIER': [1.455, 1.455, 0.01, 2, 'high'],  # 防禦力提升倍率
    'DEVOUR_SKILL_2_DURATION': [3, 3, 1, 5, 'high'],  # 防禦力提升持續時間
}

RANGER_VAR = {
    # BASE STATS
    'RANGER_BASE_HP': [279, 279, 100, 999, 'high'],
    'RANGER_BASE_ATK': [1.14, 1.14, 0.1, 999, 'high'],
    'RANGER_BASE_DEF': [0.95, 0.95, 0.1, 999, 'high'],

    # PASSIVE: COLD ARROW
    'RANGER_PASSIVE_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],  # 被動觸發機率
    'RANGER_PASSIVE_DAMAGE': [35, 35, 1, 999, 'high'],  # 被動傷害

    # SKILL 27: DAMAGE AND STACKED BONUS
    'RANGER_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    'RANGER_SKILL_0_BONUS_DAMAGE_PER_USE': [15, 15, 1, 999, 'high'],  # 每次連續使用增加的傷害

    # SKILL 28: DEFENSE BUFF
    'RANGER_SKILL_1_DEFENSE_BUFF': [1.3, 1.3, 0.1, 2, 'high'],  # 防禦力提升倍率
    'RANGER_SKILL_1_DURATION': [3, 3, 1, 5, 'nd'],  # 持續時間
    'RANGER_SKILL_1_AMBUSH_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],  # 埋伏成功觸發機率
    'RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER': [0.5, 0.5, 0.01, 1, 'high'],  # 埋伏反擊傷害倍率

    # SKILL 29: IMMUNE EFFECT
    'RANGER_SKILL_2_HP_COST': [30, 30, 1, 999, 'low'],  # 消耗生命值
    'RANGER_SKILL_2_DURATION': [2, 2, 1, 5, 'high'],  # 免疫效果持續時間
    'RANGER_SKILL_2_COOLDOWN': [5, 5, 1, 10, 'no'],  # 冷卻時間
}

ELEMENTALMAGE_VAR = {
    # BASE STATS
    'ELEMENTALMAGE_BASE_HP': [248, 248, 100, 999, 'high'],
    'ELEMENTALMAGE_BASE_ATK': [1.45, 1.45, 0.1, 999, 'high'],
    'ELEMENTALMAGE_BASE_DEF': [0.95, 0.95, 0.1, 999, 'high'],

    # PASSIVE: ELEMENTAL FORCE
    'ELEMENTALMAGE_PASSIVE_TRIGGER_RATE': [0.3, 0.3, 0.01, 1, 'high'],  # 元素效果觸發機率

    # SKILL 30: LIGHTNING ARMOR
    'ELEMENTALMAGE_SKILL_0_DEFENSE_BUFF': [1.5, 1.5, 0.1, 2, 'high'],  # 防禦力提升倍率
    'ELEMENTALMAGE_SKILL_0_DURATION': [2, 2, 1, 5, 'nd'],  # 持續時間
    'ELEMENTALMAGE_SKILL_0_HEAL_PERCENT': [0.05, 0.05, 0.01, 0.1, 'high'],  # 恢復生命比例
    'ELEMENTALMAGE_SKILL_0_PARALYSIS_TRIGGER_RATE': [0.3, 0.3, 0.01, 1, 'high'],  # 麻痺觸發機率

    # SKILL 31: ELEMENTAL BURST
    'ELEMENTALMAGE_SKILL_1_DAMAGE': [55, 55, 1, 999, 'high'],  # 基礎傷害
    'ELEMENTALMAGE_SKILL_1_ADDITIONAL_DAMAGE': [20, 20, 1, 999, 'high'],  # 每層狀態額外傷害

    # SKILL 32: THUNDERSTRIKE
    'ELEMENTALMAGE_SKILL_2_DAMAGE': [35, 35, 1, 999, 'high'],  # 基礎傷害
    'ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE': [0.35, 0.35, 0.01, 1, 'high'],  # 暈眩觸發機率
    'ELEMENTALMAGE_SKILL_2_PARALYSIS_DURATIONS': [2, 3, 4],  # 暈眩持續時間選項
    'ELEMENTALMAGE_SKILL_2_PARALYSIS_DURATIONS_WEIGHTS': [0.6, 0.3, 0.1],  # 暈眩持續時間機率
}

HUANGSHEN_VAR = {
    # BASE STATS
    'HUANGSHEN_BASE_HP': [248, 248, 100, 999, 'high'],
    'HUANGSHEN_BASE_ATK': [1.23, 1.23, 0.1, 999, 'high'],
    'HUANGSHEN_BASE_DEF': [1.15, 1.15, 0.1, 999, 'high'],

    # PASSIVE: WITHER BLADE
    'HUANGSHEN_PASSIVE_EXTRA_HIT_THRESHOLD': [2, 2, 1, 5, 'no'],  # 每造成兩次傷害增加一次追打機會
    'HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT': [0.05, 0.05, 0.01, 0.1, 'high'],  # 追打造成敵方當前生命的百分比傷害

    # SKILL 33: MULTI-HIT DAMAGE
    'HUANGSHEN_SKILL_0_DAMAGE': [35, 35, 1, 999, 'high'],
    'HUANGSHEN_SKILL_0_DAMAGE_REDUCTION_PER_HIT': [0.25, 0.25, 0.01, 1, 'low'],  # 每次額外攻擊傷害減少比例
    'HUANGSHEN_SKILL_0_HIT_RANGE': [1, 3],  # 隨機攻擊次數範圍

    # SKILL 34: CYCLIC EFFECT
    'HUANGSHEN_SKILL_1_ATK_BUFF': [1.25, 1.25, 0.01, 2, 'high'],  # 攻擊力提升倍率
    'HUANGSHEN_SKILL_1_HEAL_BUFF': [1.25, 1.25, 0.01, 2, 'high'],  # 治癒力提升倍率
    'HUANGSHEN_SKILL_1_DEF_BUFF': [1.25, 1.25, 0.01, 2, 'high'],  # 防禦力提升倍率
    'HUANGSHEN_SKILL_1_BUFF_DURATION': [3, 3, 1, 5, 'high'],  # 持續時間

    # SKILL 35: HEAL BASED ON DAMAGE TIMES
    'HUANGSHEN_SKILL_2_HEAL_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 每次傷害回血比例
    'HUANGSHEN_SKILL_2_COOLDOWN': [3, 3, 1, 10, 'low'],  # 技能冷卻時間
}

GODOFSTAR_VAR = {
    # BASE STATS
    'GODOFSTAR_BASE_HP': [325, 325, 100, 999, 'high'],
    'GODOFSTAR_BASE_ATK': [1, 1, 0.1, 999, 'high'],
    'GODOFSTAR_BASE_DEF': [1, 1, 0.1, 999, 'high'],

    # PASSIVE: STAR PENDULUM
    'GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT': [10, 10, 1, 999, 'high'],  # 每層效果造成的額外傷害
    'GODOFSTAR_PASSIVE_HEAL_PER_EFFECT': [10, 10, 1, 999, 'high'],  # 每層效果恢復的額外生命值

    # SKILL 36: DEBUFF METEOR
    'GODOFSTAR_SKILL_0_DAMAGE': [35, 35, 1, 999, 'high'],
    'GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER': [0.9, 0.9, 0.01, 1, 'low'],  # 減益倍率
    'GODOFSTAR_SKILL_0_DEBUFF_DURATION': [3, 3, 1, 5, 'nd'],  # 持續時間

    # SKILL 37: BUFF METEOR
    'GODOFSTAR_SKILL_1_HEAL': [35, 35, 1, 999, 'high'],
    'GODOFSTAR_SKILL_1_BUFF_MULTIPLIER': [1.1, 1.1, 1.0, 2, 'low'],  # 增益倍率
    'GODOFSTAR_SKILL_1_BUFF_DURATION': [3, 3, 1, 5, 'high'],  # 持續時間

    # SKILL 38: STAR CREATION
    'GODOFSTAR_SKILL_2_DAMAGE': [50, 50, 1, 999, 'high'],  # 基礎傷害
    'GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER': [1.5, 1.5, 0.1, 3, 'high'],  # 被動效果增強倍率
}
