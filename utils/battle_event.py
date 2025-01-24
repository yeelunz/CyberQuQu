# battle_event.py
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class BattleEvent:
        """
            這個取代原本的add battle_log
            
            - type: str, 事件類型
                其中包含以下事件類型:
                - damage: 傷害事件
                - heal: 治療事件
                - status_apply: 異常狀態
                - status_tick: 異常狀態持續
                - status_remove: 異常狀態移除
                - status_set 異常狀態設置stack
                - passive_trigger: 被動技能觸發播報
                - skill: 技能事件
                - text : 純文字事件
                - round_start: 回合開始
                - round_end: 回合結束
                - other: 其他事件
            - user: str, 使用者名稱
                此項只在 damage/heal/skill中被使用。
            - target: str, 目標名稱
                此項為動畫中的目標
            - animation: str, 動畫名稱
                如果沒有指定的話，則從type中自動選擇
            - appendix: DICT, 附加資訊
            - text: str, 文字事件的文字 若設置此項的話，則text直接套用此項，部會自動生成
        """
        type: str
        
        
        user: Optional[str] = None
        target: Optional[str] = None
        animation: Optional[str] = None

        text: Optional[str] = None
        appendix: Optional[Dict] = None