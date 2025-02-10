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
    'PALADIN_BASE_HP': [0, 0, 100, 999, 'high'],
    'PALADIN_BASE_ATK': [1.0, 1.0, 0.1, 999, 'high'],
    'PALADIN_BASE_DEF': [1.1, 1.1, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'PALADIN_PASSIVE_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],
    # PASSIVE HEAL AMOUNT RATE
    'PALADIN_PASSIVE_HEAL_RATE': [0.15, 0.15, 0.01, 1, 'high'],
    # OVERHEADLING DAMAGE RATE
    'PALADIN_PASSIVE_OVERHEADLINGE_RATE': [1, 1, 0.01, 999, 'high'],

    # SKILL 1 DAMAGE AMOUNT
    'PALADIN_SKILL_0_DAMAGE': [30, 30, 1, 999, 'high'],
    
    # PALADING SKILL 2 HEAL AMOUNT AND COOLDOWN
    'PALADIN_SKILL_1_HEAL': [15, 15, 1, 999, 'high'],
    'PALADIN_SKILL_1_COOLDOWN': [3, 3, 1, 999, 'low'],

    # SKILL 3 DAMAGE AMOUNT
    # FIRST HEAL AMOUNT FIRST/SECOND/or more
    'PALADIN_SKILL_2_FIRST_HEAL': [60, 60, 1, 999, 'high'],
    'PALADIN_SKILL_2_SECOND_HEAL': [40, 40, 1, 999, 'high'],
    'PALADIN_SKILL_2_MORE_HEAL': [30, 30, 1, 999, 'high'],
    
    # SKILL 4 DAMAGE AMOUNT AND DEFENSE BUFF
    'PALADIN_SKILL_3_MAX_HP_HEAL': [0.25, 0.25, 1, 999, 'high'],
    'PALADIN_SKILL_3_DAMAGE_BUFF': [2, 2, 0.1, 999, 'high'],
    'PALADIN_SKILL_3_DEFENSE_DEBUFF': [0.75, 0.75, 0.1, 999, 'high'],
    'PALADIN_SKILL_3_DURATION': [3, 3, 1, 999, 'high'],
    'PALADIN_SKILL_3_COOLDOWN': [5, 5, 1, 999, 'low'],

}

MAGE_VAR = {
    # BASE STATS
    'MAGE_BASE_HP': [249, 249, 100, 999, 'high'],
    'MAGE_BASE_ATK': [1.25, 1.25, 0.1, 999, 'high'],
    'MAGE_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'MAGE_PASSIVE_TRIGGER_RATE': [0.5, 0.5, 0.01, 1, 'high'],

    # SKILL 0 DAMAGE AMOUNT AND BURN DAMAGE
    'MAGE_SKILL_0_DAMAGE': [35, 35, 1, 999, 'high'],
    'MAGE_SKILL_0_BURN_DAMAGE': [5, 5, 1, 999, 'high'],

    # SKILL 1 DAMAGE AMOUNT
    'MAGE_SKILL_1_DAMAGE': [35, 35, 1, 999, 'high'],

    # SKILL 2 DAMAGE AND STATUS MULTIPLIER
    'MAGE_SKILL_2_BASE_DAMAGE': [25, 25, 1, 999, 'high'],
    'MAGE_SKILL_2_STATUS_MULTIPLIER': [35, 35, 1, 999, 'high'],
    
    
    # SKILL 3 DAMAGE AND STATUS MULTIPLIER
    'MAGE_SKILL_3_BASE_DAMAGE': [5, 5, 1, 999, 'high'],
    'MAGE_SKILL_3_STATUS_MULTIPLIER': [10, 10, 1, 999, 'high'],
}

ASSASSIN_VAR = {
    # BASE STATS
    'ASSASSIN_BASE_HP': [279, 279, 100, 999, 'high'],
    'ASSASSIN_BASE_ATK': [1.08, 1.08, 0.1, 999, 'high'],
    'ASSASSIN_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'ASSASSIN_PASSIVE_TRIGGER_RATE': [0.35, 0.35, 0.01, 1, 'high'],

    # PASSIVE BONUS DAMAGE RATE
    'ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE': [0.05, 0.05, 0.01, 1, 'high'],

    # SKILL 0 DAMAGE AMOUNT AND CRIT RATE
    'ASSASSIN_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    'ASSASSIN_SKILL_0_CRIT_RATE': [0.3, 0.3, 0.01, 1, 'high'],

    # SKILL 1 DAMAGE PER POISON LAYER AND HEAL PER LAYER
    'ASSASSIN_SKILL_1_DAMAGE_PER_LAYER': [20, 20, 1, 999, 'high'],
    'ASSASSIN_SKILL_1_HEAL_PER_LAYER': [20, 20, 1, 999, 'high'],

    # SKILL 2 DAMAGE AMOUNT AND POISON STACKS
    'ASSASSIN_SKILL_2_DAMAGE': [15, 15, 1, 999, 'high'],
    'ASSASSIN_SKILL_2_POISON_STACKS_1_WEIGHT': [2,4,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_2_WEIGHT': [3,3,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_3_WEIGHT': [2,2,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_4_WEIGHT': [1,1,0,999,'nd'],
    'ASSASSIN_SKILL_2_POISON_STACKS_5_WEIGHT': [1,1,0,999,'nd'],
    
    'ASSASSIN_SKILL_2_POISON_DAMAGE': [3, 3, 1, 999, 'high'],
    
    # SKILL 3
    'ASSASSIN_SKILL_3_DEBUFF_MULTIPLIER': [0.7, 0.7, 1, 999, 'high'],
    'ASSASSIN_SKILL_3_COOLDOWN': [3, 3, 1, 999, 'high'],
}

ARCHER_VAR = {
    # BASE STATS
    'ARCHER_BASE_HP': [275, 275, 100, 999, 'high'],
    'ARCHER_BASE_ATK': [1.03, 1.03, 0.1, 999, 'high'],
    'ARCHER_BASE_DEF': [1.03, 1.03, 0.1, 999, 'high'],

    # PASSIVE TRIGGER RATE
    'ARCHER_PASSIVE_BASE_TRIGGER_RATE': [0.05, 0.05, 0.01, 1, 'high'],
    'ARCHER_PASSIVE_TRIGGER_RATE_BONUS': [2.0, 2.0, 0.1, 5, 'high'],  # 每單位防禦力增加的觸發機率
    'ARCHER_PASSIVE_TRIGGER_RATE_MAX': [0.5, 0.5, 0.1, 1, 'high'],  # 最大觸發機率
    # 觸發後的傷害倍率
    'ARCHER_PASSIVE_DAMAGE_MULTIPLIER': [2, 2, 0.1, 5, 'high'],

    # SKILL 9 DAMAGE AND DEFENSE DEBUFF
    'ARCHER_SKILL_0_DAMAGE': [48, 48, 1, 999, 'high'],
    'ARCHER_SKILL_0_DEFENSE_DEBUFF': [0.9, 0.9, 0.1, 1, 'low'],  # 防禦力降低比例
    'ARCHER_SKILL_0_DURATION': [2, 2, 1, 5, 'high'],

    # SKILL 10 MULTIPLIERS AND DURATIONS
    'ARCHER_SKILL_1_DAMAGE_MULTIPLIER': [3.5, 3.5, 1, 5, 'high'],  # 提升攻擊倍率
    'ARCHER_SKILL_1_DEFENSE_MULTIPLIER': [0.5, 0.5, 0.1, 1, 'high'],  # 降低防禦倍率
    'ARCHER_SKILL_1_DURATION': [2, 2, 1, 5, 'high'],
    'ARCHER_SKILL_1_SUCESS_RATIO': [0.6, 0.6, 0.01, 1, 'high'],  # 成功率

    # SKILL 11 DAMAGE AND HEAL
    'ARCHER_SKILL_2_DAMAGE': [30, 30, 1, 999, 'high'],
    'ARCHER_SKILL_2_HEAL': [20, 20, 1, 999, 'high'],
    
    # skill 3 
    'ARCHER_SKILL_3_DAMAGE': [40, 40, 1, 999, 'high'],
    'ARCHER_SKILL_3_IGN_DEFEND': [0.8, 0.8, 0.1, 0.95, 'high'],
}

BERSERKER_VAR = {
    # BASE STATS
    'BERSERKER_BASE_HP': [378, 378, 100, 999, 'high'],
    'BERSERKER_BASE_ATK': [1.03, 1.03, 0.1, 999, 'high'],
    'BERSERKER_BASE_DEF': [0.8, 0.8, 0.1, 999, 'high'],

    # PASSIVE EXTRA DAMAGE RATE
    'BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE': [0.3, 0.3, 0.01, 1, 'high'],
    # 這邊是用來判斷是否要進行額外傷害的門檻值
    'BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD': [0.55, 0.55, 0.01, 1, 'high'],

    # SKILL 0 DAMAGE AND SELF-MUTILATION
    'BERSERKER_SKILL_0_DAMAGE': [35, 35, 1, 999, 'high'],
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
    'BERSERKER_SKILL_2_DURATION': [3, 3, 1, 999, 'high'],
    
    'BERSERKER_SKILL_3_BASE_HEAL_RATE': [0.1, 0.1, 0.1, 999, 'high'],
    'BERSERKER_SKILL_3_BONUS_RATE': [2.5, 2.5, 0.1, 999, 'high'],
    'BERSERKER_SKILL_3_COOLDOWN': [5, 5, 1, 999, 'low'],
}

DRAGONGOD_VAR = {
    # BASE STATS
    'DRAGONGOD_BASE_HP': [276, 276, 100, 999, 'high'],
    'DRAGONGOD_BASE_ATK': [1.0, 1.0, 0.1, 999, 'high'],
    'DRAGONGOD_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE EFFECT
    'DRAGONGOD_PASSIVE_ATK_MULTIPLIER': [1.05, 1.05, 0.01, 2, 'high'],
    'DRAGONGOD_PASSIVE_DEF_MULTIPLIER': [1.05, 1.05, 0.01, 2, 'high'],

    # SKILL 0 DAMAGE
    'DRAGONGOD_SKILL_0_BASE_DAMAGE': [20, 20, 1, 999, 'high'],
    'DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK': [5, 5, 1, 999, 'high'],

    # SKILL 1 HEAL AND BLEED
    'DRAGONGOD_SKILL_1_HEAL_AMOUNT': [120, 120, 1, 999, 'high'],
    'DRAGONGOD_SKILL_1_BLEED_PER_TURN': [30, 30, 1, 999, 'high'],
    'DRAGONGOD_SKILL_1_BLEED_DURATION': [3, 3, 1, 999, 'no'],
    'DRAGONGOD_SKILL_1_COOLDOWN': [4, 4, 1, 999, 'no'],

    # SKILL 2 DAMAGE AND STACK CONSUMPTION
    'DRAGONGOD_SKILL_2_DAMAGE_PER_STACK': [35, 35, 1, 999, 'high'],
    # 這邊是消耗的疊加層數比例 如果是0.5代表消耗一半
    'DRAGONGOD_SKILL_2_STACK_CONSUMPTION': [0.5, 0.5, 0.1, 1, 'nd'],
    
    'DRAGONGOD_SKILL_3_ADD_STACK': [3, 3, 1, 999, 'high'],
    'DRAGONGOD_SKILL_3_COOLDOWN': [5, 5, 1, 999, 'no'],
}


BLOODGOD_VAR = {
    # BASE STATS
    'BLOODGOD_BASE_HP': [276, 276, 100, 999, 'high'],
    'BLOODGOD_BASE_ATK': [1.6, 1.6, 0.1, 999, 'high'],
    'BLOODGOD_BASE_DEF': [1.6, 1.6, 0.1, 999, 'high'],

    # PASSIVE EFFECT
    'BLOODGOD_PASSIVE_DAMAGE_THRESHOLD': [0.25, 0.25, 0.01, 1, 'high'],  # 每受到最大血量的 25% 傷害
    'BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION': [0.9, 0.9, 0.01, 1, 'low'],  # 攻擊、防禦、治癒力降低比例

    # SKILL 18: DAMAGE, BLEED EFFECT, HEAL PER BLEED STACK
    'BLOODGOD_SKILL_0_DAMAGE': [45, 45, 1, 999, 'high'],
    'BLOODGOD_SKILL_0_BLEED_DURATION': [5, 5, 1, 999, 'no'],
    'BLOODGOD_SKILL_0_HEAL_PER_BLEED_STACK': [3, 3, 1, 999, 'high'],

    # SKILL 19: BLEED STACK EFFECTS
    'BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 每層流血降低累積傷害的倍率
    'BLOODGOD_SKILL_1_HEAL_MULTIPLIER': [5, 5, 1, 999, 'high'],  # 每層流血恢復的生命值
    'BLOODGOD_SKILL_1_BLEED_STACK_MULTIPLIER': [2, 2, 1, 999, 'high'],  # 流血層數翻倍

    # SKILL 20: RESURRECTION
    'BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO': [0.15, 0.15, 0.01, 1, 'low'],  # 消耗當前生命值的比例
    'BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO': [0.3, 0.3, 0.01, 1, 'high'],  # 回復最大生命值的比例
    'BLOODGOD_SKILL_2_BLOOD_ACCUMULATION_MULTIPLIER': [3, 3, 1, 999, 'high'],  # 致死累積傷害倍率
    'BLOODGOD_SKILL_2_DURATION': [2, 2, 1, 999, 'no'],  # 轉生效果持續時間
    'BLOODGOD_SKILL_2_COOLDOWN': [5, 5, 1, 999, 'no'],  # 技能冷卻時間
    
    # SKILL 3
    'BLOODGOD_SKILL_3_REDUCE_DAMAGE': [0.9, 0.9, 0.1, 1, 'high'],
    'BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER': [0.8, 0.8, 0.1, 1, 'high'],
    'BLOODGOD_SKILL_3_COOLDOWN': [7, 7, 1, 999, 'no'],
}

STEADFASTWARRIOR_VAR = {
    # BASE STATS
    'STEADFASTWARRIOR_BASE_HP': [263, 263, 100, 999, 'high'],
    'STEADFASTWARRIOR_BASE_ATK': [0.92, 0.92, 0.1, 999, 'high'],
    'STEADFASTWARRIOR_BASE_DEF': [1.2, 1.2, 0.1, 999, 'high'],

    # PASSIVE: HEAL LOST HP PERCENTAGE
    'STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT': [0.07, 0.07, 0.01, 1, 'high'],  # 每回合恢復損失生命值的百分比

    # SKILL 21: DAMAGE AND DEFENSE DEBUFF
    'STEADFASTWARRIOR_SKILL_0_DAMAGE': [33, 33, 1, 999, 'high'],
    'STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF': [0.75, 0.75, 0.01, 1, 'low'],  # 防禦降低比例
    'STEADFASTWARRIOR_SKILL_0_DURATION': [3, 3, 1, 5, 'high'],  # 持續回合數

    # SKILL 22: DEFENSE BUFF AND HEAL
    'STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF': [1.3, 1.3, 0.1, 2, 'high'],  # 防禦力增加比例
    'STEADFASTWARRIOR_SKILL_1_DURATION': [1, 1, 1, 5, 'high'],  # 防禦力持續時間
    'STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT': [30, 30, 1, 999, 'high'],  # 恢復的生命值

    # SKILL 23: COUNTER DAMAGE MULTIPLIER
    'STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER': [2.5, 2.5, 0.1, 10, 'high'],  # 反擊傷害倍率
    'STEADFASTWARRIOR_SKILL_2_COOLDOWN': [3, 3, 1, 10, 'low'],  # 技能冷卻時間
    
    # SKILL 3
    'STEADFASTWARRIOR_SKILL_3_DAMAGE': [20, 20, 1, 999, 'high'],
    'STEADFASTWARRIOR_SKILL_3_BONUS_DAMAGE': [10, 10, 1, 999, 'high'],  
    'STEADFASTWARRIOR_SKILL_3_COOLDOWN': [3, 3, 1, 999, 'low'],
    
}

DEVOUR_VAR = {
    # BASE STATS
    'DEVOUR_BASE_HP': [800, 800, 100, 999, 'high'],
    'DEVOUR_BASE_ATK': [1.0, 1.0, 0.1, 999, 'high'],
    'DEVOUR_BASE_DEF': [1.0, 1.0, 0.1, 999, 'high'],

    # PASSIVE: SELF DAMAGE PERCENTAGE
    'DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT': [0.08, 0.08, 0.01, 1, 'low'],  # 攻擊時消耗的生命值比例

    # SKILL 24: DAMAGE AND FAILURE RATE
    'DEVOUR_SKILL_0_DAMAGE': [45, 45, 1, 999, 'high'],
    'DEVOUR_SKILL_0_FAILURE_RATE': [0.4, 0.4, 0.01, 1, 'low'],  # 技能失敗的機率
    'DEVOUR_SKILL_0_FAILURE_ADD_RATE': [0.08, 0.08, 0.01, 1, 'low'],  # 技能失敗的機率增加

    # SKILL 25: DAMAGE BASED ON HP DIFFERENCE
    'DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER': [0.12, 0.12, 0.01, 1, 'high'],  # 已損血量傷害倍率
    'DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER': [0.12, 0.12, 0.01, 1, 'high'],  # 當前血量傷害倍率
    'DEVOUR_SKILL_1_COOLDOWN': [2, 2, 1, 10, 'low'],  # 技能冷卻時間

    # SKILL 26: DEFENSE BUFF
    'DEVOUR_SKILL_2_DEFENSE_MULTIPLIER': [1.45, 1.45, 0.01, 2, 'high'],  # 防禦力提升倍率
    'DEVOUR_SKILL_2_DURATION': [3, 3, 1, 5, 'high'],  # 防禦力提升持續時間
    
    # skill 3
    'DEVOUR_SKILL_3_PARALYSIS_DURATION': [3, 3, 3, 999, 'high'], 
    'DEVOUR_SKILL_3_MUST_SUCCESS_DURATAION': [6, 6, 1, 999, 'high'],
    'DEVOUR_SKILL_3_COOLDOWN': [5, 5, 1, 999, 'low'],
}

RANGER_VAR = {
    # BASE STATS
    'RANGER_BASE_HP': [279, 279, 100, 999, 'high'],
    'RANGER_BASE_ATK': [1.04, 1.04, 0.1, 999, 'high'],
    'RANGER_BASE_DEF': [0.95, 0.95, 0.1, 999, 'high'],

    # PASSIVE: COLD ARROW
    'RANGER_PASSIVE_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],  # 被動觸發機率
    'RANGER_PASSIVE_DAMAGE': [35, 35, 1, 999, 'high'],  # 被動傷害

    # SKILL 27: DAMAGE AND STACKED BONUS
    'RANGER_SKILL_0_DAMAGE': [40, 40, 1, 999, 'high'],
    'RANGER_SKILL_0_BONUS_DAMAGE_PER_USE': [10, 10, 1, 999, 'high'],  # 每次連續使用增加的傷害

    # SKILL 28: DEFENSE BUFF
    'RANGER_SKILL_1_DEFENSE_BUFF': [1.3, 1.3, 0.1, 2, 'high'],  # 防禦力提升倍率
    'RANGER_SKILL_1_DURATION': [3, 3, 1, 5, 'nd'],  # 持續時間
    'RANGER_SKILL_1_AMBUSH_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],  # 埋伏成功觸發機率
    'RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER': [0.5, 0.5, 0.01, 1, 'high'],  # 埋伏反擊傷害倍率

    # SKILL 29: IMMUNE EFFECT
    'RANGER_SKILL_2_HP_COST': [30, 30, 1, 999, 'low'],  # 消耗生命值
    'RANGER_SKILL_2_DURATION': [2, 2, 1, 5, 'high'],  # 免疫效果持續時間
    'RANGER_SKILL_2_COOLDOWN': [5, 5, 1, 10, 'no'],  # 冷卻時間
    
    # SKILL 3
    'RANGER_SKILL_3_DURATION': [3, 3, 1, 999, 'high'],
    'RANGER_SKILL_3_HP_THRESHOLD': [0.6, 0.6, 0.1, 1, 'low'],
    'RANGER_SKILL_3_DAMAGE_RATE_SUCCESS': [0.6, 0.6, 0.1, 999, 'high'],
    'RANGER_SKILL_3_DAMAGE_RATE_FAIL': [0.4, 0.4, 0.1, 999, 'high'],
    'RANGER_SKILL_3_DEBUFF_MULTIPLIER_SUCCESS': [0.6, 0.6, 0.1, 1, 'high'],
    'RANGER_SKILL_3_DEBUFF_MULTIPLIER_FAIL': [0.8, 0.8, 0.1, 1, 'high'],
    'RANGER_SKILL_3_COOLDOWN': [5, 5, 1, 999, 'no'],
}

ELEMENTALMAGE_VAR = {
    # BASE STATS
    'ELEMENTALMAGE_BASE_HP': [248, 248, 100, 999, 'high'],
    'ELEMENTALMAGE_BASE_ATK': [1.12, 1.12, 0.1, 999, 'high'],
    'ELEMENTALMAGE_BASE_DEF': [1.02, 1.02, 0.1, 999, 'high'],
    
    'ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_TRIGGER_RATE': [0.2, 0.2, 0.01, 1, 'high'],
    'ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],
    'ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_DURATION': [2, 2, 1, 999, 'high'],
    'ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION': [2, 2, 1, 999, 'high'],
    'ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE': [0.25, 0.25, 0.1, 1, 'high'],
    
    
    'ELEMENTALMAGE_SKILL_0_DAMAGE': [10, 10, 1, 999, 'high'],
    'ELEMENTALMAGE_SKILL_0_SIGLE_ELEMENT_BONOUS': [2.5, 2.5, 1, 999, 'high'],
    'ELEMENTALMAGE_SKILL_0_MULTI_ELEMENT_BONOUS': [6, 6, 1, 999, 'high'],

    # SKILL 30: LIGHTNING ARMOR
    'ELEMENTALMAGE_SKILL_1_DEFENSE_BUFF': [1.5, 1.5, 0.1, 2, 'high'],  # 防禦力提升倍率
    'ELEMENTALMAGE_SKILL_1_DURATION': [2, 2, 1, 5, 'nd'],  # 持續時間
    'ELEMENTALMAGE_SKILL_1_HEAL_PERCENT': [0.08, 0.08, 0.01, 0.1, 'high'],  # 恢復生命比例
    'ELEMENTALMAGE_SKILL_1_PARALYSIS_TRIGGER_RATE': [0.2, 0.2, 0.01, 1, 'high'],  # 麻痺觸發機率
    'ELEMENTALMAGE_SKILL_1_ELEMENT_ADD': [0.2, 0.2, 0.01, 1, 'high'],  # 麻痺觸發機率

    # SKILL 31: ELEMENTAL BURST
    'ELEMENTALMAGE_SKILL_2_DAMAGE': [25, 25, 1, 999, 'high'],
    'ELEMENTALMAGE_SKILL_2_BONOUS_DAMAGE_MULTIPLIER': [1.5, 1.5, 0.1, 2, 'high'],  # 傷害倍率
    
    'ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE': [0.25, 0.25, 0.01, 1, 'high'],  # 麻痺觸發機率
    
    'ELEMENTALMAGE_SKILL_3_DAMAGE': [30, 30, 1, 999, 'high'],
    'ELEMENTALMAGE_SKILL_3_BONOUS_DAMAGE_MULTIPLIER': [1.5, 1.5, 0.1, 2, 'high'],  # 傷害倍率

}

HUANGSHEN_VAR = {
    # BASE STATS
    'HUANGSHEN_BASE_HP': [248, 248, 100, 999, 'high'],
    'HUANGSHEN_BASE_ATK': [1.05, 1.05, 0.1, 999, 'high'],
    'HUANGSHEN_BASE_DEF': [1.08, 1.08, 0.1, 999, 'high'],

    # PASSIVE: WITHER BLADE
    'HUANGSHEN_PASSIVE_EXTRA_HIT_THRESHOLD': [2, 2, 1, 5, 'no'],  # 每造成兩次傷害增加一次追打機會
    'HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT': [0.05, 0.05, 0.01, 0.1, 'high'],  # 追打造成敵方當前生命的百分比傷害

    # SKILL 33: MULTI-HIT DAMAGE
    'HUANGSHEN_SKILL_0_DAMAGE': [30, 30, 1, 999, 'high'],
    'HUANGSHEN_SKILL_0_DAMAGE_REDUCTION_PER_HIT': [0.35, 0.35, 0.01, 1, 'low'],  # 每次額外攻擊傷害減少35%(2次=70%)
    'HUANGSHEN_SKILL_0_HIT_RANGE': [1, 3],  # 隨機攻擊次數範圍

    # SKILL 34: CYCLIC EFFECT
    'HUANGSHEN_SKILL_1_ATK_BUFF': [1.5, 1.5, 0.01, 2, 'high'],  # 攻擊力提升倍率
    'HUANGSHEN_SKILL_1_HEAL_BUFF': [1.5, 1.5, 0.01, 2, 'high'],  # 治癒力提升倍率
    'HUANGSHEN_SKILL_1_DEF_BUFF': [1.5, 1.5, 0.01, 2, 'high'],  # 防禦力提升倍率
    'HUANGSHEN_SKILL_1_BUFF_DURATION': [4, 4, 1, 5, 'high'],  # 持續時間

    # SKILL 35: HEAL BASED ON DAMAGE TIMES
    'HUANGSHEN_SKILL_2_HEAL_MULTIPLIER': [8, 8, 1, 999, 'high'],  # 每次傷害回血比例
    'HUANGSHEN_SKILL_2_COOLDOWN': [3, 3, 1, 10, 'low'],  # 技能冷卻時間
    
    # SKILL 3
    'HUANGSHEN_SKILL_3_DURATION': [3, 3, 1, 999, 'high'],
    'HUANGSHEN_SKILL_3_REDUCE_MULTIPLIER': [0.98, 0.98, 0.1, 1, 'high'],
    'HUANGSHEN_SKILL_3_COOLDOWN': [6, 6, 1, 999, 'low'],
    'HUANGSHEN_PASSIVE_EXTRA_HIT_HEAL': [0.5, 0.5, 0.5, 2, 'high'],
}

GODOFSTAR_VAR = {
    # BASE STATS
    'GODOFSTAR_BASE_HP': [305, 305, 100, 999, 'high'],
    'GODOFSTAR_BASE_ATK': [1, 1, 0.1, 999, 'high'],
    'GODOFSTAR_BASE_DEF': [1, 1, 0.1, 999, 'high'],

    # PASSIVE: STAR PENDULUM
    'GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT': [10, 10, 1, 999, 'high'],  # 每層效果造成的額外傷害
    'GODOFSTAR_PASSIVE_HEAL_PER_EFFECT': [10, 10, 1, 999, 'high'],  # 每層效果恢復的額外生命值

    # SKILL 36: DEBUFF METEOR
    'GODOFSTAR_SKILL_0_DAMAGE': [25, 25, 1, 999, 'high'],
    'GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER': [0.9, 0.9, 0.01, 1, 'low'],  # 減益倍率
    'GODOFSTAR_SKILL_0_DEBUFF_DURATION': [3, 3, 1, 5, 'nd'],  # 持續時間

    # SKILL 37: BUFF METEOR
    'GODOFSTAR_SKILL_1_HEAL': [25, 25, 1, 999, 'high'],
    'GODOFSTAR_SKILL_1_BUFF_MULTIPLIER': [1.1, 1.1, 1.0, 2, 'low'],  # 增益倍率
    'GODOFSTAR_SKILL_1_BUFF_DURATION': [3, 3, 1, 5, 'high'],  # 持續時間

    # SKILL 38: STAR CREATION
    'GODOFSTAR_SKILL_2_DAMAGE': [45, 45, 1, 999, 'high'],  # 基礎傷害
    'GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER': [1.25, 1.25, 0.1, 3, 'high'],  # 被動效果增強倍率
    
    # skill 3
    'GODOFSTAR_SKILL_3_DAMAGE': [15, 15, 1, 999, 'high'],
    'GODOFSTAR_SKILL_3_HEAL': [15, 15, 1, 999, 'high'],
    'GODOFSTAR_SKILL_3_PASSIVE_MULTIPLIER': [1.05, 1.05, 0.1, 3, 'high'],  # 被動效果增強倍率
    'GODOFSTAR_SKILL_3_COOLDOWN': [3, 3, 1, 999, 'low'],
    
}



VARIABLE_NAME_MAPPING = {
    "PALADIN_VAR": {
        "PALADIN_BASE_HP": "聖騎士基本生命值",
        "PALADIN_BASE_ATK": "聖騎士基本攻擊力係數",
        "PALADIN_BASE_DEF": "聖騎士基本防禦係數",
        "PALADIN_PASSIVE_TRIGGER_RATE": "聖騎士被動觸發機率",
        "PALADIN_PASSIVE_HEAL_RATE": "聖騎士被動治療率",
        "PALADIN_PASSIVE_OVERHEADLINGE_RATE": "聖騎士被動溢出治療率",
        "PALADIN_SKILL_0_DAMAGE": "聖光斬基礎傷害",
        "PALADIN_SKILL_1_HEAL": "堅守防禦基礎恢復量",
        "PALADIN_SKILL_1_COOLDOWN": "堅守防禦冷卻回合",
        "PALADIN_SKILL_2_FIRST_HEAL": "神聖治療首次恢復量",
        "PALADIN_SKILL_2_SECOND_HEAL": "神聖治療第二次恢復量",
        "PALADIN_SKILL_2_MORE_HEAL": "神聖治療後續恢復量",
        "PALADIN_SKILL_3_MAX_HP_HEAL": "決一死戰最大血量恢復比例",
        "PALADIN_SKILL_3_DAMAGE_BUFF": "決一死戰傷害增幅",
        "PALADIN_SKILL_3_DEFENSE_DEBUFF": "決一死戰防禦降低",
        "PALADIN_SKILL_3_DURATION": "決一死戰持續回合",
        "PALADIN_SKILL_3_COOLDOWN": "決一死戰冷卻回合"
    },
    "MAGE_VAR": {
        "MAGE_BASE_HP": "法師基本生命值",
        "MAGE_BASE_ATK": "法師基本攻擊力",
        "MAGE_BASE_DEF": "法師基本防禦力",
        "MAGE_PASSIVE_TRIGGER_RATE": "法師被動觸發機率",
        "MAGE_SKILL_0_DAMAGE": "火焰之球基礎傷害",
        "MAGE_SKILL_0_BURN_DAMAGE": "火焰之球燃燒附加傷害",
        "MAGE_SKILL_1_DAMAGE": "冰霜箭基礎傷害",
        "MAGE_SKILL_2_BASE_DAMAGE": "全域爆破基礎傷害",
        "MAGE_SKILL_2_STATUS_MULTIPLIER": "全域爆破狀態傷害加成",
        "MAGE_SKILL_3_BASE_DAMAGE": "詠唱破棄·全域爆破基礎傷害",
        "MAGE_SKILL_3_STATUS_MULTIPLIER": "詠唱破棄·全域爆破狀態傷害加成"
    },
    "ASSASSIN_VAR": {
        "ASSASSIN_BASE_HP": "刺客基本生命值",
        "ASSASSIN_BASE_ATK": "刺客基本攻擊力",
        "ASSASSIN_BASE_DEF": "刺客基本防禦力",
        "ASSASSIN_PASSIVE_TRIGGER_RATE": "刺客被動觸發機率",
        "ASSASSIN_PASSIVE_BONUS_DAMAGE_RATE": "刺客被動額外傷害率",
        "ASSASSIN_SKILL_0_DAMAGE": "致命暗殺基礎傷害",
        "ASSASSIN_SKILL_0_CRIT_RATE": "致命暗殺暴擊率",
        "ASSASSIN_SKILL_1_DAMAGE_PER_LAYER": "毒爆每層傷害",
        "ASSASSIN_SKILL_1_HEAL_PER_LAYER": "毒爆每層治療量",
        "ASSASSIN_SKILL_2_DAMAGE": "毒刃襲擊基礎傷害",
        "ASSASSIN_SKILL_2_POISON_STACKS_1_WEIGHT": "毒刃襲擊毒疊1權重",
        "ASSASSIN_SKILL_2_POISON_STACKS_2_WEIGHT": "毒刃襲擊毒疊2權重",
        "ASSASSIN_SKILL_2_POISON_STACKS_3_WEIGHT": "毒刃襲擊毒疊3權重",
        "ASSASSIN_SKILL_2_POISON_STACKS_4_WEIGHT": "毒刃襲擊毒疊4權重",
        "ASSASSIN_SKILL_2_POISON_STACKS_5_WEIGHT": "毒刃襲擊毒疊5權重",
        "ASSASSIN_SKILL_2_POISON_DAMAGE": "毒刃襲擊中毒傷害",
        "ASSASSIN_SKILL_3_DEBUFF_MULTIPLIER": "致命藥劑減益倍率",
        "ASSASSIN_SKILL_3_COOLDOWN": "致命藥劑冷卻回合"
    },
    "ARCHER_VAR": {
        "ARCHER_BASE_HP": "弓箭手基本生命值",
        "ARCHER_BASE_ATK": "弓箭手基本攻擊力",
        "ARCHER_BASE_DEF": "弓箭手基本防禦力",
        "ARCHER_PASSIVE_BASE_TRIGGER_RATE": "弓箭手被動基礎觸發機率",
        "ARCHER_PASSIVE_TRIGGER_RATE_BONUS": "弓箭手被動觸發機率加成",
        "ARCHER_PASSIVE_TRIGGER_RATE_MAX": "弓箭手被動觸發機率上限",
        "ARCHER_PASSIVE_DAMAGE_MULTIPLIER": "弓箭手被動傷害倍率",
        "ARCHER_SKILL_0_DAMAGE": "五連矢基礎傷害",
        "ARCHER_SKILL_0_DEFENSE_DEBUFF": "五連矢防禦降低百分比",
        "ARCHER_SKILL_0_DURATION": "五連矢持續回合",
        "ARCHER_SKILL_1_DAMAGE_MULTIPLIER": "箭矢補充攻擊傷害倍率",
        "ARCHER_SKILL_1_DEFENSE_MULTIPLIER": "箭矢補充防禦倍率",
        "ARCHER_SKILL_1_DURATION": "箭矢補充持續回合",
        "ARCHER_SKILL_1_SUCESS_RATIO": "箭矢補充成功率",
        "ARCHER_SKILL_2_DAMAGE": "吸血箭傷害",
        "ARCHER_SKILL_2_HEAL": "吸血箭治療量",
        "ARCHER_SKILL_3_DAMAGE": "驟雨基礎傷害",
        "ARCHER_SKILL_3_IGN_DEFEND": "驟雨無視防禦百分比"
    },
    "BERSERKER_VAR": {
        "BERSERKER_BASE_HP": "狂戰士基本生命值",
        "BERSERKER_BASE_ATK": "狂戰士基本攻擊力",
        "BERSERKER_BASE_DEF": "狂戰士基本防禦力",
        "BERSERKER_PASSIVE_EXTRA_DAMAGE_RATE": "狂戰士被動額外傷害率",
        "BERSERKER_PASSIVE_EXTRA_DAMAGE_THRESHOLD": "狂戰士被動額外傷害門檻",
        "BERSERKER_SKILL_0_DAMAGE": "狂暴之力傷害",
        "BERSERKER_SKILL_0_SELF_MUTILATION_RATE": "狂暴之力自殘比率",
        "BERSERKER_SKILL_1_HP_COST": "熱血生命消耗",
        "BERSERKER_SKILL_1_HEAL_PER_TURN": "熱血每回合治療量",
        "BERSERKER_SKILL_1_DURATION": "熱血持續回合",
        "BERSERKER_SKILL_1_COOLDOWN": "熱血冷卻回合",
        "BERSERKER_SKILL_2_HP_COST": "血怒之泉生命消耗",
        "BERSERKER_SKILL_2_DEFENSE_BUFF": "血怒之泉防禦提升",
        "BERSERKER_SKILL_2_DURATION": "血怒之泉持續回合",
        "BERSERKER_SKILL_3_BASE_HEAL_RATE": "嗜血本能基礎治療比率",
        "BERSERKER_SKILL_3_BONUS_RATE": "嗜血本能額外治療比率",
        "BERSERKER_SKILL_3_COOLDOWN": "嗜血本能冷卻回合"
    },
    "DRAGONGOD_VAR": {
        "DRAGONGOD_BASE_HP": "龍神基本生命值",
        "DRAGONGOD_BASE_ATK": "龍神基本攻擊力",
        "DRAGONGOD_BASE_DEF": "龍神基本防禦力",
        "DRAGONGOD_PASSIVE_ATK_MULTIPLIER": "龍神被動攻擊倍率",
        "DRAGONGOD_PASSIVE_DEF_MULTIPLIER": "龍神被動防禦倍率",
        "DRAGONGOD_SKILL_0_BASE_DAMAGE": "神龍之息基礎傷害",
        "DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK": "神龍之息每層額外傷害",
        "DRAGONGOD_SKILL_1_HEAL_AMOUNT": "龍血之泉治療量",
        "DRAGONGOD_SKILL_1_BLEED_PER_TURN": "龍血之泉每回合流血傷害",
        "DRAGONGOD_SKILL_1_BLEED_DURATION": "龍血之泉流血持續回合",
        "DRAGONGOD_SKILL_1_COOLDOWN": "龍血之泉冷卻回合",
        "DRAGONGOD_SKILL_2_DAMAGE_PER_STACK": "神龍燎原每層傷害",
        "DRAGONGOD_SKILL_2_STACK_CONSUMPTION": "神龍燎原層數消耗比例",
        "DRAGONGOD_SKILL_3_ADD_STACK": "預借增加層數",
        "DRAGONGOD_SKILL_3_COOLDOWN": "預借冷卻回合"
    },
    "BLOODGOD_VAR": {
        "BLOODGOD_BASE_HP": "血神基本生命值",
        "BLOODGOD_BASE_ATK": "血神基本攻擊力",
        "BLOODGOD_BASE_DEF": "血神基本防禦力",
        "BLOODGOD_PASSIVE_DAMAGE_THRESHOLD": "血神被動傷害門檻",
        "BLOODGOD_PASSIVE_MULTIPLIER_REDUCTION": "血神被動倍率降低",
        "BLOODGOD_SKILL_0_DAMAGE": "血刀傷害",
        "BLOODGOD_SKILL_0_BLEED_DURATION": "血刀流血持續回合",
        "BLOODGOD_SKILL_0_HEAL_PER_BLEED_STACK": "血刀每層流血治療量",
        "BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER": "血脈祭儀流血減免倍率",
        "BLOODGOD_SKILL_1_HEAL_MULTIPLIER": "血脈祭儀治療倍率",
        "BLOODGOD_SKILL_1_BLEED_STACK_MULTIPLIER": "血脈祭儀流血疊加倍率",
        "BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO": "轉生自傷比例",
        "BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO": "轉生復活治療比例",
        "BLOODGOD_SKILL_2_BLOOD_ACCUMULATION_MULTIPLIER": "轉生血量累積倍率",
        "BLOODGOD_SKILL_2_DURATION": "轉生持續回合",
        "BLOODGOD_SKILL_2_COOLDOWN": "轉生冷卻回合",
        "BLOODGOD_SKILL_3_REDUCE_DAMAGE": "新生減傷比例",
        "BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER": "新生減益倍率",
        "BLOODGOD_SKILL_3_COOLDOWN": "新生冷卻回合"
    },
    "STEADFASTWARRIOR_VAR": {
        "STEADFASTWARRIOR_BASE_HP": "剛毅武士基本生命值",
        "STEADFASTWARRIOR_BASE_ATK": "剛毅武士基本攻擊力",
        "STEADFASTWARRIOR_BASE_DEF": "剛毅武士基本防禦力",
        "STEADFASTWARRIOR_PASSIVE_HEAL_PERCENT": "剛毅武士被動治療百分比",
        "STEADFASTWARRIOR_SKILL_0_DAMAGE": "剛毅打擊傷害",
        "STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF": "剛毅打擊防禦降低幅度",
        "STEADFASTWARRIOR_SKILL_0_DURATION": "剛毅打擊持續回合",
        "STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF": "不屈意志防禦提升",
        "STEADFASTWARRIOR_SKILL_1_DURATION": "不屈意志持續回合",
        "STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT": "不屈意志治療量",
        "STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER": "絕地反擊傷害倍率",
        "STEADFASTWARRIOR_SKILL_2_COOLDOWN": "絕地反擊冷卻回合",
        "STEADFASTWARRIOR_SKILL_3_DAMAGE": "破魂斬傷害",
        "STEADFASTWARRIOR_SKILL_3_BONUS_DAMAGE": "破魂斬額外傷害",
        "STEADFASTWARRIOR_SKILL_3_COOLDOWN": "破魂斬冷卻回合"
    },
    "DEVOUR_VAR": {
        "DEVOUR_BASE_HP": "吞噬者基本生命值",
        "DEVOUR_BASE_ATK": "吞噬者基本攻擊力",
        "DEVOUR_BASE_DEF": "吞噬者基本防禦力",
        "DEVOUR_PASSIVE_SELF_DAMAGE_PERCENT": "吞噬者被動自傷百分比",
        "DEVOUR_SKILL_0_DAMAGE": "吞裂傷害",
        "DEVOUR_SKILL_0_FAILURE_RATE": "吞裂失敗率",
        "DEVOUR_SKILL_0_FAILURE_ADD_RATE": "吞裂失敗累加率",
        "DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER": "巨口吞世流失生命傷害倍率",
        "DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER": "巨口吞世當前生命傷害倍率",
        "DEVOUR_SKILL_1_COOLDOWN": "巨口吞世冷卻回合",
        "DEVOUR_SKILL_2_DEFENSE_MULTIPLIER": "堅硬皮膚防禦倍率",
        "DEVOUR_SKILL_2_DURATION": "堅硬皮膚持續回合",
        "DEVOUR_SKILL_3_PARALYSIS_DURATION": "觸電反應麻痺持續回合",
        "DEVOUR_SKILL_3_MUST_SUCCESS_DURATAION": "觸電反應必定成功持續回合",
        "DEVOUR_SKILL_3_COOLDOWN": "觸電反應冷卻回合"
    },
    "RANGER_VAR": {
        "RANGER_BASE_HP": "遊俠基本生命值",
        "RANGER_BASE_ATK": "遊俠基本攻擊力",
        "RANGER_BASE_DEF": "遊俠基本防禦力",
        "RANGER_PASSIVE_TRIGGER_RATE": "遊俠被動觸發機率",
        "RANGER_PASSIVE_DAMAGE": "遊俠被動傷害",
        "RANGER_SKILL_0_DAMAGE": "續戰攻擊基礎傷害",
        "RANGER_SKILL_0_BONUS_DAMAGE_PER_USE": "續戰攻擊每次額外傷害",
        "RANGER_SKILL_1_DEFENSE_BUFF": "埋伏防禦防禦提升",
        "RANGER_SKILL_1_DURATION": "埋伏防禦持續回合",
        "RANGER_SKILL_1_AMBUSH_TRIGGER_RATE": "埋伏防禦伏擊觸發機率",
        "RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER": "埋伏防禦伏擊傷害倍率",
        "RANGER_SKILL_2_HP_COST": "荒原抗性能量消耗",
        "RANGER_SKILL_2_DURATION": "荒原抗性持續回合",
        "RANGER_SKILL_2_COOLDOWN": "荒原抗性冷卻回合",
        "RANGER_SKILL_3_DURATION": "地雷持續回合",
        "RANGER_SKILL_3_HP_THRESHOLD": "地雷血量門檻",
        "RANGER_SKILL_3_DAMAGE_RATE_SUCCESS": "地雷成功傷害比例",
        "RANGER_SKILL_3_DAMAGE_RATE_FAIL": "地雷失敗傷害比例",
        "RANGER_SKILL_3_DEBUFF_MULTIPLIER_SUCCESS": "地雷成功減益倍率",
        "RANGER_SKILL_3_DEBUFF_MULTIPLIER_FAIL": "地雷失敗減益倍率",
        "RANGER_SKILL_3_COOLDOWN": "地雷冷卻回合"
    },
    "ELEMENTALMAGE_VAR": {
        "ELEMENTALMAGE_BASE_HP": "元素法師基本生命值",
        "ELEMENTALMAGE_BASE_ATK": "元素法師基本攻擊力",
        "ELEMENTALMAGE_BASE_DEF": "元素法師基本防禦力",
        "ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_TRIGGER_RATE": "元素法師單體麻痺觸發機率",
        "ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_TRIGGER_RATE": "元素法師群體麻痺觸發機率",
        "ELEMENTALMAGE_PASSIVE_SINGLE_PARALYSIS_DURATION": "元素法師單體麻痺持續回合",
        "ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_DURATION": "元素法師群體麻痺持續回合",
        "ELEMENTALMAGE_PASSIVE_MULTI_PARALYSIS_ONE_MORE_DURATION_RATE": "元素法師群體麻痺額外延長機率",
        "ELEMENTALMAGE_SKILL_0_DAMAGE": "融合傷害基礎值",
        "ELEMENTALMAGE_SKILL_0_SIGLE_ELEMENT_BONOUS": "融合單一屬性加成",
        "ELEMENTALMAGE_SKILL_0_MULTI_ELEMENT_BONOUS": "融合多屬性加成",
        "ELEMENTALMAGE_SKILL_1_DEFENSE_BUFF": "雷霆護甲防禦提升",
        "ELEMENTALMAGE_SKILL_1_DURATION": "雷霆護甲持續回合",
        "ELEMENTALMAGE_SKILL_1_HEAL_PERCENT": "雷霆護甲治療百分比",
        "ELEMENTALMAGE_SKILL_1_PARALYSIS_TRIGGER_RATE": "雷霆護甲麻痺觸發機率",
        "ELEMENTALMAGE_SKILL_1_ELEMENT_ADD": "雷霆護甲附加元素機率",
        "ELEMENTALMAGE_SKILL_2_DAMAGE": "寒星墜落基礎傷害",
        "ELEMENTALMAGE_SKILL_2_BONOUS_DAMAGE_MULTIPLIER": "寒星墜落傷害加成倍率",
        "ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE": "寒星墜落麻痺觸發機率",
        "ELEMENTALMAGE_SKILL_3_DAMAGE": "焚天基礎傷害",
        "ELEMENTALMAGE_SKILL_3_BONOUS_DAMAGE_MULTIPLIER": "焚天傷害加成倍率"
    },
    "HUANGSHEN_VAR": {
        "HUANGSHEN_BASE_HP": "荒神基本生命值",
        "HUANGSHEN_BASE_ATK": "荒神基本攻擊力",
        "HUANGSHEN_BASE_DEF": "荒神基本防禦力",
        "HUANGSHEN_PASSIVE_EXTRA_HIT_THRESHOLD": "荒神被動額外命中門檻",
        "HUANGSHEN_PASSIVE_EXTRA_HIT_DAMAGE_PERCENT": "荒神被動額外命中傷害百分比",
        "HUANGSHEN_SKILL_0_DAMAGE": "枯骨基礎傷害",
        "HUANGSHEN_SKILL_0_DAMAGE_REDUCTION_PER_HIT": "枯骨每次命中減傷",
        "HUANGSHEN_SKILL_0_HIT_RANGE": "枯骨攻擊次數範圍",
        "HUANGSHEN_SKILL_1_ATK_BUFF": "荒原攻擊提升百分比",
        "HUANGSHEN_SKILL_1_HEAL_BUFF": "荒原治療提升百分比",
        "HUANGSHEN_SKILL_1_DEF_BUFF": "荒原防禦提升百分比",
        "HUANGSHEN_SKILL_1_BUFF_DURATION": "荒原增益持續回合",
        "HUANGSHEN_SKILL_2_HEAL_MULTIPLIER": "生命逆流治療倍率",
        "HUANGSHEN_SKILL_2_COOLDOWN": "生命逆流冷卻回合",
        "HUANGSHEN_SKILL_3_DURATION": "風化持續回合",
        "HUANGSHEN_SKILL_3_REDUCE_MULTIPLIER": "風化減傷倍率",
        "HUANGSHEN_SKILL_3_COOLDOWN": "風化冷卻回合",
        "HUANGSHEN_PASSIVE_EXTRA_HIT_HEAL": "荒神被動額外命中治療"
    },
    "GODOFSTAR_VAR": {
        "GODOFSTAR_BASE_HP": "星神基本生命值",
        "GODOFSTAR_BASE_ATK": "星神基本攻擊力",
        "GODOFSTAR_BASE_DEF": "星神基本防禦力",
        "GODOFSTAR_PASSIVE_DAMAGE_PER_EFFECT": "星神被動每次效果傷害",
        "GODOFSTAR_PASSIVE_HEAL_PER_EFFECT": "星神被動每次效果治療",
        "GODOFSTAR_SKILL_0_DAMAGE": "災厄隕星基礎傷害",
        "GODOFSTAR_SKILL_0_DEBUFF_MULTIPLIER": "災厄隕星減益倍率",
        "GODOFSTAR_SKILL_0_DEBUFF_DURATION": "災厄隕星減益持續回合",
        "GODOFSTAR_SKILL_1_HEAL": "光輝流星治療量",
        "GODOFSTAR_SKILL_1_BUFF_MULTIPLIER": "光輝流星增益倍率",
        "GODOFSTAR_SKILL_1_BUFF_DURATION": "光輝流星增益持續回合",
        "GODOFSTAR_SKILL_2_DAMAGE": "創星圖錄傷害",
        "GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER": "創星圖錄被動倍率",
        "GODOFSTAR_SKILL_3_DAMAGE": "無序聯星傷害",
        "GODOFSTAR_SKILL_3_HEAL": "無序聯星治療量",
        "GODOFSTAR_SKILL_3_PASSIVE_MULTIPLIER": "無序聯星被動倍率",
        "GODOFSTAR_SKILL_3_COOLDOWN": "無序聯星冷卻回合"
    }
}
