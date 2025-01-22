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

# 火焰領域
sm.add_skill(Skill(
    3, "火焰之球", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_0_DAMAGE'][0]} 點傷害，並疊加 1 層燃燒狀態（最多 3 層）。燃燒時會額外對敵方造成 {MAGE_VAR['MAGE_SKILL_0_BURN_DAMAGE'][0]} 點傷害。", 'damage'))
sm.add_skill(Skill(
    4, "冰霜箭", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_1_DAMAGE'][0]} 點傷害，並疊加 1 層冰凍狀態（最多 3 層）。", 'damage'))
sm.add_skill(Skill(
    5, "全域爆破", f"對敵方造成 {MAGE_VAR['MAGE_SKILL_2_BASE_DAMAGE'][0]} 點傷害，並引爆累積的狀態。每層燃燒或冰凍效果額外對敵方造成 {MAGE_VAR['MAGE_SKILL_2_STATUS_MULTIPLIER'][0]} 點傷害。", 'damage'))

# 暗影刺殺
sm.add_skill(Skill(
    6, "致命暗殺", f"對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_0_DAMAGE'][0]} 點傷害，{int(ASSASSIN_VAR['ASSASSIN_SKILL_0_CRIT_RATE'][0] * 100)}% 機率使傷害翻倍。", 'damage'))
sm.add_skill(Skill(
    7, "毒爆", f"引爆目標的中毒狀態。每層中毒對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_1_DAMAGE_PER_LAYER'][0]} 點傷害，並為自身恢復 {ASSASSIN_VAR['ASSASSIN_SKILL_1_HEAL_PER_LAYER'][0]} 點生命值。", 'effect'))
sm.add_skill(Skill(
    8, "毒刃襲擊", f"對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_2_DAMAGE'][0]} 點傷害，並疊加 1~5 層中毒狀態（機率由權重決定）。每層中毒對敵方造成 {ASSASSIN_VAR['ASSASSIN_SKILL_2_POISON_DAMAGE'][0]} 點傷害。", 'damage'))

# 弓箭手
sm.add_skill(Skill(
    9, "五連矢", f"對敵方造成 {ARCHER_VAR['ARCHER_SKILL_0_DAMAGE'][0]} 點傷害，並使其防禦力降低 {int((1 - ARCHER_VAR['ARCHER_SKILL_0_DEFENSE_DEBUFF'][0]) * 100)}%， 持續 {ARCHER_VAR['ARCHER_SKILL_0_DURATION'][0]} 回合。", 'damage'))
sm.add_skill(Skill(10, "箭矢補充", f"持續 {ARCHER_VAR['ARCHER_SKILL_1_DURATION'][0]} 回合，提升 {int(ARCHER_VAR['ARCHER_SKILL_1_DAMAGE_MULTIPLIER'][0] * 100 - 100)}% 攻擊力，或降低 {int((1 - ARCHER_VAR['ARCHER_SKILL_1_DEFENSE_MULTIPLIER'][0]) * 100)}% 防禦力（成功率：{int(ARCHER_VAR['ARCHER_SKILL_1_SUCESS_RATIO'][0] * 100)}%）。", 'effect'))
sm.add_skill(Skill(
    11, "吸血箭", f"對敵方造成 {ARCHER_VAR['ARCHER_SKILL_2_DAMAGE'][0]} 點傷害，並為自身恢復 {ARCHER_VAR['ARCHER_SKILL_2_HEAL'][0]} 點生命值。", 'damage'))

# 狂戰士
sm.add_skill(Skill(
    12,
    "狂暴之力",
    f"對敵方造成 {BERSERKER_VAR['BERSERKER_SKILL_0_DAMAGE'][0]} 點傷害，並自身反嗜 {int(BERSERKER_VAR['BERSERKER_SKILL_0_SELF_MUTILATION_RATE'][0] * 100)}% 的攻擊傷害。",
    'damage'
))

sm.add_skill(Skill(
    13,
    "熱血",
    f"消耗 {BERSERKER_VAR['BERSERKER_SKILL_1_HP_COST'][0]} 點生命值，持續 {BERSERKER_VAR['BERSERKER_SKILL_1_DURATION'][0]} 回合，每回合恢復 {BERSERKER_VAR['BERSERKER_SKILL_1_HEAL_PER_TURN'][0]} 點生命值。",
    'effect',
    BERSERKER_VAR['BERSERKER_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    14,
    "血怒之泉",
    f"消耗 {BERSERKER_VAR['BERSERKER_SKILL_2_HP_COST'][0]} 點生命值，持續 {BERSERKER_VAR['BERSERKER_SKILL_2_DURATION'][0]} 回合內免疫控制，並提升自身 {int((BERSERKER_VAR['BERSERKER_SKILL_2_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力。",
    'effect'
))

# 龍神
sm.add_skill(Skill(
    15,
    "神龍之息",
    f"對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BASE_DAMAGE'][0]} 點傷害，每層龍神狀態額外對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_0_BONUS_DAMAGE_PER_STACK'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    16,
    "龍血之泉",
    f"恢復 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_HEAL_AMOUNT'][0]} 點生命值，持續 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_DURATION'][0]} 回合，每回合扣除 {DRAGONGOD_VAR['DRAGONGOD_SKILL_1_BLEED_PER_TURN'][0]} 點生命值。",
    'heal',
    DRAGONGOD_VAR['DRAGONGOD_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    17,
    "神龍燎原",
    f"消耗 {int(DRAGONGOD_VAR['DRAGONGOD_SKILL_2_STACK_CONSUMPTION'][0] * 100)}% 的龍神層數，每層對敵方造成 {DRAGONGOD_VAR['DRAGONGOD_SKILL_2_DAMAGE_PER_STACK'][0]} 點傷害。",
    'damage'
))

# 血神
sm.add_skill(Skill(
    18, "血刀",
    f"對敵方造成 {BLOODGOD_VAR['BLOODGOD_SKILL_0_DAMAGE'][0]} 點傷害，對敵方疊加流血狀態，攻擊流血的敵人時恢復生命值。",
    'damage'
))

sm.add_skill(Skill(
    19, "血脈祭儀",
    f"血脈的累積傷害降低敵方流血層數的 {BLOODGOD_VAR['BLOODGOD_SKILL_1_BLEED_REDUCTION_MULTIPLIER'][0]} 倍；恢復流血層數的 {BLOODGOD_VAR['BLOODGOD_SKILL_1_HEAL_MULTIPLIER'][0]} 倍生命值，並使流血層數翻倍。",
    'effect'
))

sm.add_skill(Skill(
    20, "轉生",
    f"消耗當前生命值的 {int(BLOODGOD_VAR['BLOODGOD_SKILL_2_SELF_DAMAGE_RATIO'][0] * 100)}%，持續 {BLOODGOD_VAR['BLOODGOD_SKILL_2_DURATION'][0]} 回合內，免於致死傷害並回復最大血量的 {int(BLOODGOD_VAR['BLOODGOD_SKILL_2_RESURRECT_HEAL_RATIO'][0] * 100)}%。",
    'effect',
    BLOODGOD_VAR['BLOODGOD_SKILL_2_COOLDOWN'][0]
))
# 剛毅武士 (SteadfastWarrior) 技能定義
sm.add_skill(Skill(
    21, "剛毅打擊",
    f"對敵方造成 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DAMAGE'][0]} 點傷害，並降低其 {int((1 - STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DEFENSE_DEBUFF'][0]) * 100)}% 防禦力，持續 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_0_DURATION'][0]} 回合。",
    'damage'
))

sm.add_skill(Skill(
    22, "不屈意志",
    f"持續 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DURATION'][0]} 回合，自身防禦力增加 {int((STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_DEFENSE_BUFF'][0] - 1) * 100)}%，並恢復 {STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_1_HEAL_AMOUNT'][0]} 點生命值。",
    'effect'
))

sm.add_skill(Skill(
    23, "絕地反擊",
    f"對攻擊者立即造成其本次攻擊傷害的 {int(STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_DAMAGE_MULTIPLIER'][0] * 100)}% 傷害。",
    'damage',
    STEADFASTWARRIOR_VAR['STEADFASTWARRIOR_SKILL_2_COOLDOWN'][0]
))

# 鯨吞 技能定義
sm.add_skill(Skill(
    24, "吞裂",
    f"對敵方造成 {DEVOUR_VAR['DEVOUR_SKILL_0_DAMAGE'][0]} 點傷害，{int(DEVOUR_VAR['DEVOUR_SKILL_0_FAILURE_RATE'][0] * 100)}% 機率使用失敗。",
    'damage'
))

sm.add_skill(Skill(
    25, "巨口吞世",
    f"當敵方當前血量比例較我方高時，對敵方造成已損血量 {int(DEVOUR_VAR['DEVOUR_SKILL_1_LOST_HP_DAMAGE_MULTIPLIER'][0] * 100)}% 的傷害，否則，造成當前血量 {int(DEVOUR_VAR['DEVOUR_SKILL_1_CURRENT_HP_DAMAGE_MULTIPLIER'][0] * 100)}% 的傷害。",
    'damage',
    DEVOUR_VAR['DEVOUR_SKILL_1_COOLDOWN'][0]
))

sm.add_skill(Skill(
    26, "堅硬皮膚",
    f"提升 {int((DEVOUR_VAR['DEVOUR_SKILL_2_DEFENSE_MULTIPLIER'][0] - 1) * 100)}% 防禦力，持續 {DEVOUR_VAR['DEVOUR_SKILL_2_DURATION'][0]} 回合。",
    'effect'
))


# 荒原遊俠 (Ranger) 技能定義
sm.add_skill(Skill(
    27, "續戰攻擊",
    f"對敵方造成 {RANGER_VAR['RANGER_SKILL_0_DAMAGE'][0]} 點傷害，每次連續使用攻擊技能時額外造成 {RANGER_VAR['RANGER_SKILL_0_BONUS_DAMAGE_PER_USE'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    28, "埋伏防禦",
    f"持續 {RANGER_VAR['RANGER_SKILL_1_DURATION'][0]} 回合，提升 {int((RANGER_VAR['RANGER_SKILL_1_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力，期間受到攻擊時 {int(RANGER_VAR['RANGER_SKILL_1_AMBUSH_TRIGGER_RATE'][0] * 100)}% 機率反擊傷害的 {int(RANGER_VAR['RANGER_SKILL_1_AMBUSH_DAMAGE_MULTIPLIER'][0] * 100)}%。",
    'effect'
))

sm.add_skill(Skill(
    29, "荒原抗性",
    f"消耗 {RANGER_VAR['RANGER_SKILL_2_HP_COST'][0]} 點生命值，免疫 {RANGER_VAR['RANGER_SKILL_2_DURATION'][0]} 回合的傷害以及控制效果。",
    'effect',
    RANGER_VAR['RANGER_SKILL_2_COOLDOWN'][0]
))


# 元素法師 (ElementalMage) 技能定義
sm.add_skill(Skill(
    30, "雷霆護甲",
    f"持續 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DURATION'][0]} 回合，受到傷害時有 {int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_PARALYSIS_TRIGGER_RATE'][0] * 100)}% 機率直接麻痺敵人，並提升 {int((ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_DEFENSE_BUFF'][0] - 1) * 100)}% 防禦力，恢復最大生命的 {int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_0_HEAL_PERCENT'][0] * 100)}%。",
    'effect'
))

sm.add_skill(Skill(
    31, "凍燒雷",
    f"對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_DAMAGE'][0]} 點傷害，每層燃燒、冰凍、麻痺效果額外對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_1_ADDITIONAL_DAMAGE'][0]} 點傷害。",
    'damage'
))

sm.add_skill(Skill(
    32, "雷擊術",
    f"對敵方造成 {ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_DAMAGE'][0]} 點傷害，{int(ELEMENTALMAGE_VAR['ELEMENTALMAGE_SKILL_2_PARALYSIS_TRIGGER_RATE'][0] * 100)}% 機率使敵方暈眩 2~4 回合。",
    'damage'
))

# 荒神 (HuangShen) 技能定義
sm.add_skill(Skill(
    33, "枯骨",
    f"對敵方造成 {HUANGSHEN_VAR['HUANGSHEN_SKILL_0_DAMAGE'][0]} 點傷害，隨機造成 {HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'][0]}~{HUANGSHEN_VAR['HUANGSHEN_SKILL_0_HIT_RANGE'][1]} 次的傷害，傷害會隨著額外次數而降低。",
    'damage'
))

sm.add_skill(Skill(
    34, "荒原",
    f"根據技能使用次數，分別循環發動以下效果：\n"
    f"增加自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_ATK_BUFF'][0] - 1) * 100)}% 攻擊力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。\n"
    f"增加自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_HEAL_BUFF'][0] - 1) * 100)}% 治癒力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。\n"
    f"提升自身 {int((HUANGSHEN_VAR['HUANGSHEN_SKILL_1_DEF_BUFF'][0] - 1) * 100)}% 防禦力，持續 {HUANGSHEN_VAR['HUANGSHEN_SKILL_1_BUFF_DURATION'][0]} 回合。",
    'effect'
))

sm.add_skill(Skill(
    35, "生命逆流",
    f"恢復造成傷害次數 {int(HUANGSHEN_VAR['HUANGSHEN_SKILL_2_HEAL_MULTIPLIER'][0]*100)}% 的血量。",
    'heal',
    HUANGSHEN_VAR['HUANGSHEN_SKILL_2_COOLDOWN'][0]
))


# 星神
sm.add_skill(Skill(
    36, "災厄隕星",
    f"對敵方造成 {GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DAMAGE'][0]} 點傷害，並隨機為敵方附加以下一種減益效果，持續 {GODOFSTAR_VAR['GODOFSTAR_SKILL_0_DEBUFF_DURATION'][0]} 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。",
    'damage'
))

sm.add_skill(Skill(
    37, "光輝流星",
    f"為自身恢復 {GODOFSTAR_VAR['GODOFSTAR_SKILL_1_HEAL'][0]} 點生命值，並隨機為自身附加以下一種增益效果，持續 {GODOFSTAR_VAR['GODOFSTAR_SKILL_1_BUFF_DURATION'][0]} 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。",
    'heal'
))

sm.add_skill(Skill(
    38, "虛擬創星圖",
    f"對敵方造成 {GODOFSTAR_VAR['GODOFSTAR_SKILL_2_DAMAGE'][0]} 點傷害，本回合天啟星盤的效果增加 {int((GODOFSTAR_VAR['GODOFSTAR_SKILL_2_PASSIVE_MULTIPLIER'][0] - 1) * 100)}%。",
    'damage'
))
