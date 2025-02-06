from app.main import create_app

import logging
import ray
import os
import sys

from ray.rllib.algorithms.ppo import PPOConfig
from utils.global_var import globalVar as gv
import json
from utils.profession_var import PALADIN_VAR, MAGE_VAR, ASSASSIN_VAR, ARCHER_VAR, BERSERKER_VAR, DRAGONGOD_VAR, BLOODGOD_VAR, STEADFASTWARRIOR_VAR, DEVOUR_VAR, RANGER_VAR, ELEMENTALMAGE_VAR, HUANGSHEN_VAR, GODOFSTAR_VAR
def init_global_var():
    global PALADIN_VAR, MAGE_VAR, ASSASSIN_VAR, ARCHER_VAR, BERSERKER_VAR, DRAGONGOD_VAR, BLOODGOD_VAR, STEADFASTWARRIOR_VAR, DEVOUR_VAR, RANGER_VAR, ELEMENTALMAGE_VAR, HUANGSHEN_VAR, GODOFSTAR_VAR
    global gv
    # in config/gv.json
    init_path = os.path.join(os.path.dirname(__file__), 'config/gv.json')
    # init gv.global_var from gv.json
    with open(init_path, 'r') as f:
        up = json.load(f)
        gv.update(up)
    
    
    init_path_profession = os.path.join(os.path.dirname(__file__), 'config/profession_var.json')
    with open(init_path_profession, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    PALADIN_VAR.clear()
    PALADIN_VAR.update(data.get("PALADIN_VAR", {}))

    # 更新 MAGE_VAR
    MAGE_VAR.clear()
    MAGE_VAR.update(data.get("MAGE_VAR", {}))

    # 更新 ASSASSIN_VAR
    ASSASSIN_VAR.clear()
    ASSASSIN_VAR.update(data.get("ASSASSIN_VAR", {}))

    # 更新 ARCHER_VAR
    ARCHER_VAR.clear()
    ARCHER_VAR.update(data.get("ARCHER_VAR", {}))

    # 更新 BERSERKER_VAR
    BERSERKER_VAR.clear()
    BERSERKER_VAR.update(data.get("BERSERKER_VAR", {}))

    # 更新 DRAGONGOD_VAR
    DRAGONGOD_VAR.clear()
    DRAGONGOD_VAR.update(data.get("DRAGONGOD_VAR", {}))

    # 更新 BLOODGOD_VAR
    BLOODGOD_VAR.clear()
    BLOODGOD_VAR.update(data.get("BLOODGOD_VAR", {}))

    # 更新 STEADFASTWARRIOR_VAR
    STEADFASTWARRIOR_VAR.clear()
    STEADFASTWARRIOR_VAR.update(data.get("STEADFASTWARRIOR_VAR", {}))

    # 更新 DEVOUR_VAR
    DEVOUR_VAR.clear()
    DEVOUR_VAR.update(data.get("DEVOUR_VAR", {}))

    # 更新 RANGER_VAR
    RANGER_VAR.clear()
    RANGER_VAR.update(data.get("RANGER_VAR", {}))

    # 更新 ELEMENTALMAGE_VAR
    ELEMENTALMAGE_VAR.clear()
    ELEMENTALMAGE_VAR.update(data.get("ELEMENTALMAGE_VAR", {}))

    # 更新 HUANGSHEN_VAR
    HUANGSHEN_VAR.clear()
    HUANGSHEN_VAR.update(data.get("HUANGSHEN_VAR", {}))

    # 更新 GODOFSTAR_VAR
    GODOFSTAR_VAR.clear()
    GODOFSTAR_VAR.update(data.get("GODOFSTAR_VAR", {}))


app = create_app()


if __name__ == '__main__':
    init_global_var()
    # test
    app.run(host='0.0.0.0', port=5000, debug=False)



