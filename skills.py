# skills.py

class Skill:
    """
    定義一個技能物件
    - skill_id: 唯一的技能ID (整數)
    - name: 技能的名稱 (在戰鬥log中顯示)
    - desc: 技能的詳細描述 (在職業介紹、或玩家選技能時顯示)
    - type: 技能類型 ('damage', 'heal', 'effect')
    """
    def __init__(self, skill_id: int, name: str, desc: str, type: str,cool_down:int = 0):
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
sm.add_skill(Skill(0, "聖光斬", "對單體造成 40 點傷害。", 'damage'))
sm.add_skill(Skill(1, "堅守防禦", "本回合迴避所有攻擊，並恢復 10 點生命值。冷卻 3 回合。", 'effect',3))
sm.add_skill(Skill(2, "神聖治療", "恢復血量：第一次恢復 50 點，第二次恢復 35 點，第三次及以後恢復 10 點。", 'heal'))

# 
sm.add_skill(Skill(3, "火焰之球", "對單體造成 35 點傷害，並疊加 1 層燃燒狀態（最多 3 層）。每層燃燒造成 5 點額外傷害。", 'damage'))
sm.add_skill(Skill(4, "冰霜箭", "對單體造成 35 點傷害，並疊加 1 層冰凍狀態（最多 3 層）。", 'damage'))
sm.add_skill(Skill(5, "全域爆破", "對敵方全體造成 25 點傷害，並引爆累積的狀態。每層燃燒或冰凍效果額外造成 35 點傷害。", 'damage'))

# 
sm.add_skill(Skill(6, "致命暗殺", "對單體造成 45 點傷害，35% 機率使傷害翻倍。", 'damage'))
sm.add_skill(Skill(7, "毒爆", "引爆目標的中毒狀態。每層中毒造成 10 點傷害，並為自身恢復 8 點生命值。", 'effect'))
sm.add_skill(Skill(8, "毒刃襲擊", "對單體造成 10 點傷害，並疊加 1~3 層中毒狀態（最多 5 層）。每層中毒狀態造成 3 點傷害。", 'damage'))

# 
sm.add_skill(Skill(9, "五連矢", "對單體造成 50 點傷害，並使其防禦力降低 15%。", 'damage'))
sm.add_skill(Skill(10, "箭矢補充", "2 回合間，增加 150% 攻擊力，或降低自身防禦力 50%。", 'effect'))
sm.add_skill(Skill(11, "吸血箭", "對單體造成 30 點傷害，並為自身恢復 15 點生命值。", 'damage'))

# 狂戰
sm.add_skill(Skill(12, "狂暴之力", "對單體造成 30 點傷害，並自身反嗜 15% 的攻擊傷害。", 'damage'))
sm.add_skill(Skill(13, "熱血", "消耗 150 點生命值，接下來 5 回合，每回合恢復 35 點生命值。冷卻 5 回合。", 'heal',5))
sm.add_skill(Skill(14, "血怒之泉", "犧牲 30 點生命值，接下來 2 回合內免疫控制，並提升自身 35% 防禦力。", 'effect'))

# 
sm.add_skill(Skill(15, "神龍之息", "對單體造成 35 點傷害，每層龍神狀態增加 3 點額外傷害。", 'damage'))
sm.add_skill(Skill(16, "龍血之淵", "恢復 100 點生命值，接下來 3 回合，每回合扣除 30 點生命值。冷卻 5 回合。", 'heal',4))
sm.add_skill(Skill(17, "神龍燎原", "消除一半的龍神層數，對敵方造成每層 25 點傷害。", 'damage'))

# 血神
sm.add_skill(Skill(18, "血刀", "對單體造成 45 點傷害，並疊加 1 層流血狀態。", 'damage'))
sm.add_skill(Skill(19, "飲血", "消耗敵方一半的流血層數。每層流血造成 5 點傷害，並為自身恢復 5 點生命值。", 'damage'))
sm.add_skill(Skill(20, "赤紅之災", "隨機為敵方單體疊加 3~5 層流血狀態。", 'damage'))

# SteadfastWarrior (剛毅武士) 技能定義
sm.add_skill(Skill(21, "剛毅打擊", "對單體造成 35 點傷害，並降低其 20% 防禦力，持續 2 回合。", 'damage'))
sm.add_skill(Skill(22, "不屈意志", "本回合防禦力增加 30%，並回復 25 點生命值。", 'effect'))
sm.add_skill(Skill(23, "絕地反擊", "對攻擊者立即造成其本次攻擊傷害的 200%。此技能需冷卻 3 回合。", 'damage',3))

# SunWarrior (烈陽勇士) 技能定義
sm.add_skill(Skill(24, "吞裂", "造成65點傷害，50%機率使用失敗。", 'damage'))
sm.add_skill(Skill(25, "巨口吞世", "當敵方當前血量比例較我方高時，對敵方造成已損血量15%的傷害，否則，造成當前血量10%的傷害。冷卻3回合。", 'damage',3))
sm.add_skill(Skill(26, "堅硬皮膚", "提升25%防禦力，持續3回合。", 'effect'))

# Ranger (荒原遊俠) 技能定義
sm.add_skill(Skill(27, "續戰攻擊", "造成 35 點傷害，每次連續使用攻擊技能時額外增加 20 點傷害。", 'damage'))
sm.add_skill(Skill(28, "埋伏防禦", "一回合內，提升 50% 防禦力。", 'effect'))
sm.add_skill(Skill(29, "荒原抗性", "消耗 15 點生命力，免疫一回合的傷害。", 'effect'))

# ElementalMage (元素法師) 技能定義
sm.add_skill(Skill(30, "雷霆護甲", "2 回合內，受到傷害時有 30% 機率直接麻痺敵人。", 'effect'))
sm.add_skill(Skill(31, "凍燒雷", "造成 55 點傷害，對每層麻痺、冰凍、燃燒狀態額外造成 15 點傷害。", 'damage'))
sm.add_skill(Skill(32, "雷擊術", "造成 15 點傷害，35 % 機率使敵方暈眩 1~3 回合。", 'effect'))

# HuangShen (荒神) 技能定義
sm.add_skill(Skill(33, "枯骨的永恆", "基礎傷害30，隨機造成 1 ~ 3 次傷害，傷害會隨著額外次數而降低。", 'damage'))
sm.add_skill(Skill(34, "荒原", "根據技能使用次數，分別循環發動以下效果：\n"
                                "- 增加自身 25% 攻擊力，持續 3 回合。\n"
                                "- 增加自身 25% 治癒力，持續 3 回合。\n"
                                "- 提升自身 25% 防禦力，持續 3 回合。", 'effect'))
sm.add_skill(Skill(35, "生命逆流", "回復造成傷害次數500%的血量，冷卻3回合", 'damage',3))
#  星神
sm.add_skill(Skill(36, "光輝流星", "對敵方單體造成 15 點傷害，並隨機為自身附加以下一種增益效果，持續 3 回合：攻擊力提升 5%，防禦力提升 5%，治癒效果提升 5%。", 'damage'))
sm.add_skill(Skill(37, "災厄隕星", "為自身恢復 15 點生命值，並隨機為敵方附加以下一種減益效果，持續 3 回合：攻擊力降低 5%，防禦力降低 5%，治癒效果降低 5%。", 'damage'))
sm.add_skill(Skill(38, "虛擬創星圖", "對敵方單體造成 45 點傷害，本回合天啟星盤的效果增加 25% 。", 'damage'))

