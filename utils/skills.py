# skills.py

from .profession_var import *


class Skill:
    """
    定義一個技能物件
    - skill_id: 唯一的技能ID (整數)
    - name: 技能的名稱 (在戰鬥log中顯示)
    - desc: 技能的詳細描述 (在職業介紹、或玩家選技能時顯示)
    - type: 技能類型 ('damage', 'heal', 'effect')
    """

    def __init__(self, skill_id: int, name: str, desc: str, type: str, cool_down: int = 0):
        self.skill_id = skill_id
        self.name = name
        self.desc = desc
        self.type = type
        self.cool_down = cool_down


class SkillManager:
    """
    集中管理所有技能對應:
    skill_id -> Skill(name, desc, type)
    """

    def __init__(self):
        self.skills = {}

    def add_skill(self, skill: Skill):
        self.skills[skill.skill_id] = skill

    def get_skill_name(self, skill_id: int) -> str:
        skill = self.skills.get(skill_id, None)
        if skill:
            return skill.name
        return f"(無效技能ID={skill_id})"

    def get_skill_desc(self, skill_id: int) -> str:
        skill = self.skills.get(skill_id, None)
        if skill:
            return skill.desc
        return f"(無法找到技能描述 ID={skill_id})"

    def get_skill_type(self, skill_id: int) -> str:
        skill = self.skills.get(skill_id, None)
        if skill:
            return skill.type
        return f"(無法找到技能類型 ID={skill_id})"

    def get_skill_cooldown(self, skill_id: int) -> int:
        skill = self.skills.get(skill_id, None)
        if skill:
            return skill.cool_down
        return f"(無法找到技能冷卻時間 ID={skill_id})"


# 初始化技能管理器並添加技能
sm = SkillManager()


def build_skill_manager():
    return sm
# PASSIVE SKILLS idx = -1 龍神被動 佔位
# 以下是具名技能的定義，根據提供的職業表進行定義

# PASSIVE SKILLS idx = -1 龍神被動 佔位
# 以下是具名技能的定義，根據提供的職業表進行定義
# backend/static/skills.py

# backend/static/skills.py

sm.add_skill(Skill(
    0, "聖光斬", f"對敵方造成 {PALADIN_VAR['PALADIN_SKILL_0_DAMAGE'][0]} 點傷害。", 'damage'))
sm.add_skill(Skill(
    1, "堅守防禦", f"本回合迴避所有攻擊，並恢復 {PALADIN_VAR['PALADIN_SKILL_1_HEAL'][0]} 點生命值。", 'effect', PALADIN_VAR['PALADIN_SKILL_1_COOLDOWN'][0]))
sm.add_skill(Skill(
    2, "神聖治療", f"首次使用恢復 {PALADIN_VAR['PALADIN_SKILL_2_FIRST_HEAL'][0]} 點生命值；第二次使用恢復 {PALADIN_VAR['PALADIN_SKILL_2_SECOND_HEAL'][0]} 點生命值；其後使用恢復 {PALADIN_VAR['PALADIN_SKILL_2_MORE_HEAL'][0]} 點生命值。", 'heal'))
sm.add_skill(Skill(
    3, "決一死戰", f"持續 2 回合，自身受到致死傷害時，回復最大血量的 {int(PALADIN_VAR['PALADIN_SKILL_3_MAX_HP_HEAL'][0]*100)}%。效果觸發時，持續 3 回合，自身提升 {int((PALADIN_VAR['PALADIN_SKILL_3_DAMAGE_BUFF'][0]-1) *100)}%的攻擊力，以及自身降低 {int((1-PALADIN_VAR['PALADIN_SKILL_3_DEFENSE_DEBUFF'][0])*100)}%的防禦力。", 'effect',PALADIN_VAR['PALADIN_SKILL_3_COOLDOWN'][0]))

# 火焰領域
sm.add_skill(Skill(
    4, "火焰之球", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_0_DAMAGE'][0]} 點傷害，並疊加 1 層燃燒狀態（最多 3 層）。燃燒時會額外對敵方造成 {MAGE_VAR['MAGE_SKILL_0_BURN_DAMAGE'][0]} 點傷害。", 'damage'))
sm.add_skill(Skill(
    5, "冰霜箭", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_1_DAMAGE'][0]} 點傷害，並疊加 1 層冰凍狀態（最多 3 層）。", 'damage'))
sm.add_skill(Skill(
    6, "全域爆破", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_2_BASE_DAMAGE'][0]} 點傷害，並引爆累積的狀態。每層燃燒或冰凍效果額外對敵方造成 {MAGE_VAR['MAGE_SKILL_2_STATUS_MULTIPLIER'][0]} 點傷害。", 'damage'))
sm.add_skill(Skill(
    7, "詠唱破棄．全域爆破", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_3_BASE_DAMAGE'][0]} 點傷害，並引爆累積的狀態。每層燃燒或冰凍效果額外對敵方造成 {MAGE_VAR['MAGE_SKILL_3_STATUS_MULTIPLIER'][0]} 點傷害，此技能攻擊順序必定最優先。", 'effect'))



# 暗影刺殺
sm.add_skill(Skill(
    8, "致命暗殺", f"對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_0_DAMAGE'][0]} 點傷害，{int(ASSASSIN_VAR['ASSASSIN_SKILL_0_CRIT_RATE'][0] * 100)}% 機率使傷害翻倍。", 'damage'))
sm.add_skill(Skill(
    9, "毒爆", f"引爆目標的中毒狀態。每層中毒對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_1_DAMAGE_PER_LAYER'][0]} 點傷害，並為自身恢復 {ASSASSIN_VAR['ASSASSIN_SKILL_1_HEAL_PER_LAYER'][0]} 點生命值。", 'effect'))
sm.add_skill(Skill(
    10, "毒刃襲擊", f"對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_2_DAMAGE'][0]} 點傷害，並疊加 1~5 層中毒狀態（機率由權重決定）。每層中毒對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_DAMAGE'][0]} 點傷害。", 'damage'))
# 致命藥劑 :根據敵方當前中毒層數降低治癒力，敵方每層中毒降低30%的治癒力。持續時間等同於中毒的剩餘持續時間
sm.add_skill(Skill(
    11, "致命藥劑", f"根據敵方當前中毒層數降低治癒力，敵方每層中毒降低 {int((1-ASSASSIN_VAR['ASSASSIN_SKILL_3_DEBUFF_MULTIPLIER'][0])*100)}% 治癒力。持續時間等同於中毒的剩餘持續時間。", 'effect', ASSASSIN_VAR['ASSASSIN_SKILL_3_COOLDOWN'][0]))   

    

# 弓箭手
sm.add_skill(Skill(
    12, "五連矢", f"對敵方造成 {ARCHER_VAR['ARCHER_SKILL_0_DAMAGE'][0]} 點傷害，並使其防禦力降低 {int((1 - ARCHER_VAR['ARCHER_SKILL_0_DEFENSE_DEBUFF'][0]) * 100)}%， 持續 {ARCHER_VAR['ARCHER_SKILL_0_DURATION'][0]} 回合。", 'damage'))
sm.add_skill(Skill(
    13, "箭矢補充", f"持續 {ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0]} 回合，提升 {int(ARCHER_VAR['ARCHER_SKILL_1_DAMAGE_MULTIPLIER'][0] * 100 - 100)}% 攻擊力，或降低 {int((1 - ARCHER_VAR['ARCHER_SKILL_1_DEFENSE_MULTIPLIER'][0]) * 100)}% 防禦力（成功率：{int(ARCHER_VAR['ARCHER_SKILL_1_SUCESS_RATIO'][0] * 100)}%）。", 'effect'))
sm.add_skill(Skill(
    14, "吸血箭", f"對敵方造成 {ARCHER_VAR['ARCHER_SKILL_2_DAMAGE'][0]} 點傷害，並為自身恢復 {ARCHER_VAR['ARCHER_SKILL_2_HEAL'][0]} 點生命值。", 'damage'))
# 驟雨 對敵方造成50(10*5)傷害，無視目標80%防禦力。
sm.add_skill(Skill(
    15, "驟雨", f"對敵方造成 {ARCHER_VAR['ARCHER_SKILL_3_DAMAGE'][0]} 點傷害，無視目標 {int(ARCHER_VAR['ARCHER_SKILL_3_IGN_DEFEND'][0]*100)}% 防禦力。", 'damage'))

    


# 狂戰士
sm.add_skill(Skill(
    16,
    "狂暴之力",
    f"對敵方造成 {BERSERKER_VAR['BERSERKER_SKILL_0_DAMAGE'][0]} 點傷害，並自身反嗜 {int(BERSERKER_VAR['BERSERKER_SKILL_0_SELF_MUTILATION_RATE'][0] * 100)}% 的攻擊傷害。",
    'damage'
))

sm.add_skill(Skill(
    17,
    "熱血",
    f"消耗 {BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0]} 點生命值，持續 {BERSERKER_VAR['BERSERKER_SKILL_1_DURATION'][0]} 回合，每回合恢復 {BERSERKER_VAR['BERSERKER_SKILL_1_HEAL_PER_TURN'][0]} 點生命值。",
    'effect',
    BERSERKER_VAR['BERSERKER_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    18,
    "血怒之泉",
    f"消耗 {BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0]} 點生命值，持續 {BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0]} 回合內免疫控制，並提升自身 {int((BERSERKER_VAR['BERSERKER_SKILL_2_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力。",
    'effect'
))

# 浴血 CD 4
# 3回合間獲得”吸血”效果，攻擊敵人會回復 15%傷害造成的血量。自身血量比例越低時，吸血效果額外增加，最大增加200%。
sm.add_skill(Skill(
    19,"嗜血本能",f"持續 3 回合，攻擊敵人時會恢復 {int(BERSERKER_VAR['BERSERKER_SKILL_3_BASE_HEAL_RATE'][0]*100)}% 傷害造成的血量。自身血量比例越低時，吸血效果額外增加，最大增加 {int(BERSERKER_VAR['BERSERKER_SKILL_3_BONUS_RATE'][0]*100)}% 。",'effect',BERSERKER_VAR['BERSERKER_SKILL_3_COOLDOWN'][0]))
             



# 龍神
sm.add_skill(Skill(
    20,
    "神龍之息",
    f"對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BASE_DAMAGE'][0]} 點傷害，每層龍神狀態額外對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    21,
    "龍血之泉",
    f"恢復 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_HEAL_AMOUNT'][0]} 點生命值，持續 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_DURATION'][0]} 回合，每回合扣除 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_PER_TURN'][0]} 點生命值。",
    'heal',
    DRAGONGOD_VAR['DRAGONGOD_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    22,
    "神龍燎原",
    f"消耗 {int(DRAGONGOD_VAR['DRAGONGOD_SKILL_2_STACK_CONSUMPTION'][0] * 100)}% 的龍神層數，每層對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_2_DAMAGE_PER_STACK'][0]} 點傷害。",
    'damage'
))
# 預借 CD 6
# 龍神層數立即疊加 4 層，但在接下來的4回合內不會疊加層數。
sm.add_skill(Skill(
    23,"預借",f"龍神層數立即疊加 {DRAGONGOD_VAR['DRAGONGOD_SKILL_3_ADD_STACK'][0]} 層，但在接下來的{DRAGONGOD_VAR['DRAGONGOD_SKILL_3_ADD_STACK'][0]}回合內，「龍血」技能會處於無效狀態。",'effect',DRAGONGOD_VAR['DRAGONGOD_SKILL_3_COOLDOWN'][0]))



# 血神
sm.add_skill(Skill(
    24, "血刀",
    f"對敵方造成 {BLOODGOD_VAR['BLOODGOD_SKILL_0_DAMAGE'][0]} 點傷害，對敵方疊加流血狀態，攻擊流血的敵人時恢復生命值。",
    'damage'
))

sm.add_skill(Skill(
    25, "血脈祭儀",
    f"血脈的累積傷害降低敵方流血層數的 {BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER'][0]} 倍；恢復流血層數的 {BLOODGOD_VAR['BLOODGOD_SKILL_1_HEAL_MULTIPLIER'][0]} 倍生命值，並使流血層數翻倍。",
    'effect'
))

sm.add_skill(Skill(
    26, "轉生",
    f"消耗當前生命值的 {int(BLOODGOD_VAR['BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO'][0] * 100)}%，持續 {BLOODGOD_VAR['BLOODGOD_SKILL_2_DURATION'][0]} 回合內，受到致死傷害時，復活並回復最大血量的 {int(BLOODGOD_VAR['BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO'][0] * 100)}%。",
    'effect',
    BLOODGOD_VAR['BLOODGOD_SKILL_2_COOLDOWN'][0]
))
# 新生 CD8
# 立即降低90%的血脈中的累積傷害，但是獲得一個永久的降低攻/防/治 20%的效果
sm.add_skill(Skill(
    27,"新生",f"立即降低 {int(BLOODGOD_VAR['BLOODGOD_SKILL_3_REDUCE_DAMAGE'][0]*100)}% 的血脈中的累積傷害，但是獲得一個永久的攻擊力、防禦力、治癒力降低 {int((1-BLOODGOD_VAR['BLOODGOD_SKILL_3_DEBUFF_MULTIPLIER'][0])*100)}%的效果。",'effect',BLOODGOD_VAR['BLOODGOD_SKILL_3_COOLDOWN'][0]))
             


# 剛毅武士 (SteadfastWarrior) 技能定義
sm.add_skill(Skill(
    28, "剛毅打擊",
    f"對敵方造成 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DAMAGE'][0]} 點傷害，並降低其 {int((1 - STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF'][0]) * 100)}% 防禦力，持續 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DURATION'][0]} 回合。",
    'damage'
))

sm.add_skill(Skill(
    29, "不屈意志",
    f"持續 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DURATION'][0]} 回合，自身防禦力增加 {int((STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF'][0] - 1) * 100)}%，並恢復 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT'][0]} 點生命值。",
    'effect'
))

sm.add_skill(Skill(
    30, "絕地反擊",
    f"對攻擊者立即造成其本次攻擊傷害的 {int(STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER'][0] * 100)}% 傷害。",
    'damage',
    STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_COOLDOWN'][0]
))
# 破魂斬

# 傷害：對敵方造成25點傷害
# 特殊效果：若敵方擁有增益效果，則移除一個增益效果並額外造成15點傷害，若我方有減益效果，則移除一個減益效果並額外造成15點傷害
# 冷卻時間：3回合
sm.add_skill(Skill(
    31,"破魂斬",f"對敵方造成 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_DAMAGE'][0]} 點傷害，若敵方擁有增益效果，則移除一個增益效果並額外對敵方造成 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_BONUS_DAMAGE'][0]} 點傷害，若我方有減益效果，則移除一個減益效果並額外對敵方造成 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_BONUS_DAMAGE'][0]} 點傷害。",'damage',STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_3_COOLDOWN'][0]))





# 鯨吞 技能定義
sm.add_skill(Skill(
    32, "吞裂",
    f"對敵方造成 {DEVOUR_VAR['DEVOUR_SKILL_0_DAMAGE'][0]} 點傷害，{int(DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_RATE'][0] * 100)}% 機率使用失敗，使用失敗後，提升 {int(DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_ADD_RATE'][0]*100)}% 機率成功施放，機率會被重複疊加，直到技能成功使用。",
    'damage'
))

sm.add_skill(Skill(
    33, "巨口吞世",
    f"當敵方當前血量比例較我方高時，對敵方造成已損血量 {int(DEVOUR_VAR['DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER'][0] * 100)}% 的傷害，否則，造成當前血量 {int(DEVOUR_VAR['DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER'][0] * 100)}% 的傷害。",
    'damage',
    DEVOUR_VAR['DEVOUR_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    34, "堅硬皮膚",
    f"提升 {int((DEVOUR_VAR['DEVOUR_SKILL_2_DEFENSE_MULTIPLIER'][0] - 1) * 100)}% 防禦力，持續 {DEVOUR_VAR['DEVOUR_SKILL_2_DURATION'][0]} 回合。",
    'effect'
))

# 新技能： 觸電反應 CD 3
# 麻痺自己3回合，在接下來的3回合間吞裂的使用成功機率提升至100%
sm.add_skill(Skill(
    35,"觸電反應",f"麻痺自己 {DEVOUR_VAR['DEVOUR_SKILL_3_PARALYSIS_DURATION'][0]} 回合。持續 {DEVOUR_VAR['DEVOUR_SKILL_3_MUST_SUCCESS_DURATAION'][0]} 回合，吞裂的使用成功機率提升至100%。",'effect',DEVOUR_VAR['DEVOUR_SKILL_3_COOLDOWN'][0]))




# 荒原遊俠 (Ranger) 技能定義
sm.add_skill(Skill(
    36, "續戰攻擊",
    f"對敵方造成 {RANGER_VAR['RANGER_SKILL_0_DAMAGE'][0]} 點傷害，每次連續使用攻擊技能時額外造成 {RANGER_VAR['RANGER_SKILL_0_BONUS_DAMAGE_PER_USE'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    37, "埋伏防禦",
    f"持續 {RANGER_VAR['RANGER_SKILL_1_DURATION'][0]} 回合，提升 {int((RANGER_VAR['RANGER_SKILL_1_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力，期間受到攻擊時 {int(RANGER_VAR['RANGER_SKILL_1_AMBUSH_TRIGGER_RATE'][0] * 100)}% 機率反擊傷害的 {int(RANGER_VAR['RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER'][0] * 100)}%。",
    'effect'
))

sm.add_skill(Skill(
    38, "荒原抗性",
    f"消耗 {RANGER_VAR['RANGER_SKILL_2_HP_COST'][0]} 點生命值，免疫 {RANGER_VAR['RANGER_SKILL_2_DURATION'][0]} 回合的傷害以及控制效果。",
    'effect',
    RANGER_VAR['RANGER_SKILL_2_COOLDOWN'][0]
))

# 施放一個地雷，地雷在受到最大等同於自身當前血量的50%累積傷害，或是3回合後會引爆。
# 引爆後2回合內會降低敵方40%防禦力，並受到地雷累積60%的傷害。
# 若是未達成條件即引爆時，則降低20%防禦力，並受到地雷累積40%的傷害。
sm.add_skill(Skill(
    39,"地雷",f"施放一個地雷，地雷在受到最大等同於自身當前血量的 {int(RANGER_VAR['RANGER_SKILL_3_HP_THRESHOLD'][0]*100)}% 累積傷害，或是 {RANGER_VAR['RANGER_SKILL_3_DURATION'][0] +1} 回合後會引爆。引爆後，持續 2 回合，會讓敵方降低 {int((1 - RANGER_VAR['RANGER_SKILL_3_DEBUFF_MULTIPLIER_SUCCESS'][0])*100)}% 防禦力，並受到地雷累積 {int(RANGER_VAR['RANGER_SKILL_3_DAMAGE_RATE_SUCCESS'][0]*100)}% 的傷害。若是未達成傷害累積條件即引爆時，則降低 {int((1 - RANGER_VAR['RANGER_SKILL_3_DEBUFF_MULTIPLIER_FAIL'][0])*100)}% 防禦力，並受到地雷累積 {int(RANGER_VAR['RANGER_SKILL_3_DAMAGE_RATE_FAIL'][0]*100)}% 的傷害。",'effect',RANGER_VAR['RANGER_SKILL_3_COOLDOWN'][0]))


# 元素法師 (ElementalMage) 技能定義
sm.add_skill(Skill(
    40, "雷霆護甲",
    f"持續 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DURATION'][0]} 回合，受到傷害時有 {int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_PARALYSIS_TRIGGER_RATE'][0] * 100)}% 機率直接麻痺敵人，並提升 {int((ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力，恢復最大生命的 {int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_HEAL_PERCENT'][0] * 100)}%。",
    'effect'
))

sm.add_skill(Skill(
    41, "凍燒雷",
    f"對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DAMAGE'][0]} 點傷害，每層燃燒、冰凍、麻痺效果額外對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_ADDITIONAL_DAMAGE'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    42, "雷擊術",
    f"對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_DAMAGE'][0]} 點傷害，{int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE'][0] * 100)}% 機率使敵方麻痺 2~4 回合。",
    'damage'
))
# 天啟 CD 8
# 4回合內，將雙方的治癒力係數降低99%
sm.add_skill(Skill(
    43,"天啟",f"持續 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_DURATION'][0]} 回合，雙方降低 {int((1- ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_MULTIPLIER'][0])*100)}% 的治癒力。",'effect',ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_3_COOLDOWN'][0]))


# 荒神 (HuangShen) 技能定義
sm.add_skill(Skill(
    44, "枯骨",
    f"對敵方造成 {HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE'][0]} 點傷害，隨機造成 {HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'][0]}~{HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'][1]} 次的傷害，傷害會隨著額外次數而降低。",
    'damage'
))

sm.add_skill(Skill(
    45, "荒原",
    f"根據技能使用次數，分別循環發動以下效果：\n"
    f"增加自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_ATK_BUFF'][0] - 1) * 100)}% 攻擊力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。\n"
    f"增加自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_HEAL_BUFF'][0] - 1) * 100)}% 治癒力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。\n"
    f"提升自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_DEF_BUFF'][0] - 1) * 100)}% 防禦力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。",
    'effect'
))

sm.add_skill(Skill(
    46, "生命逆流",
    f"恢復造成傷害次數 {int(HUANGSHEN_VAR['HUANGSHEN_SKILL_2_HEAL_MULTIPLIER'][0]*100)}% 的血量。",
    'heal',
    HUANGSHEN_VAR['HUANGSHEN_SKILL_2_COOLDOWN'][0]
))
# 3回合內，枯萎之刃攻擊時，降低敵方2%防禦力及治癒力。並回復等同於追打次數100%的血量
sm.add_skill(Skill(
    47,"風化",f"持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_3_DURATION'][0]} 回合，枯萎之刃攻擊時，敵方降低 {int((1- HUANGSHEN_VAR['HUANGSHEN_SKILL_3_REDUCE_MULTIPLIER'][0])*100)}% 防禦力及治療力。並恢復等同於追打次數 {int(HUANGSHEN_VAR['HUANGSHEN_PASSIVE_EXTRA_HIT_HEAL'][0]*100)}% 的血量。",'effect',HUANGSHEN_VAR['HUANGSHEN_SKILL_3_COOLDOWN'][0]))

# 星神
sm.add_skill(Skill(
    48, "災厄隕星",
    f"對敵方造成 {GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DAMAGE'][0]} 點傷害，並隨機為敵方附加以下一種減益效果，持續 {GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0]} 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。",
    'damage'
))

sm.add_skill(Skill(
    49, "光輝流星",
    f"為自身恢復 {GODOFSTAR_VAR['GODOFSTAR_SKILL_1_HEAL'][0]} 點生命值，並隨機為自身附加以下一種增益效果，持續 {GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0]} 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。",
    'heal'
))

sm.add_skill(Skill(
    50, "虛擬創星圖",
    f"對敵方造成 {GODOFSTAR_VAR['GODOFSTAR_SKILL_2_DAMAGE'][0]} 點傷害，本回合天啟星盤的效果增加 {int((GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0] - 1) * 100)}%。",
    'damage'
))

# 聯星
# 對敵方造成15點傷害
# 回復15點生命值
# 本回合天啟星盤的效果增加 25%。
# 將自身由光輝流星附加的buff，轉為減益效果賦予到對方身上；自身由災厄隕星附加的debuff，轉為增益效果賦予到對方身上
sm.add_skill(Skill(
    51,"無序聯星",f"對敵方造成 {GODOFSTAR_VAR['GODOFSTAR_SKILL_3_DAMAGE'][0]} 點傷害，恢復 {GODOFSTAR_VAR['GODOFSTAR_SKILL_3_HEAL'][0]} 點生命值，本回合天啟星盤的效果增加 {int(GODOFSTAR_VAR['GODOFSTAR_SKILL_3_PASSIVE_MULTIPLIER'][0]*100)-100}%。將自身由光輝流星附加的增益效果，轉為減益效果賦予到對方身上；自身由災厄隕星附加的簡易效果，轉為增益效果賦予自身身上。",'damage',GODOFSTAR_VAR['GODOFSTAR_SKILL_3_COOLDOWN'][0]))

