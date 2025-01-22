from flask import Blueprint, request, jsonify, render_template
from .models.main import (
    multi_agent_cross_train,
    version_test_random_vs_random,
    high_level_test_ai_vs_ai,
    compute_ai_elo,
    computer_vs_computer,
    ai_vs_computer,
    ai_vs_ai,
    show_profession_info,
    build_skill_manager,
    build_professions,
    get_professions_data
)

from .models.data_stamp import Gdata
import json
from .models.global_var import globalVar
import os

main_routes = Blueprint('main', __name__)

# 建立全域的 manager & professions，避免每次端點都要重建
skill_mgr = build_skill_manager()
professions = build_professions()

@main_routes.route('/')
def index():
    """
    回傳前端的 index.html
    """
    return render_template('index.html')

@main_routes.route('/api/train', methods=['POST'])
def api_train():
    """
    (1) 交叉疊代訓練 (多智能體)
    用 POST 傳遞要訓練多少次 iteration。
    """
    data = request.get_json()
    iteration = data.get("iteration", 5)
    
    # 這裡簡單示範，不做多餘的檢查
    save_path_1 = "multiagent_ai1.zip"
    save_path_2 = "multiagent_ai2.zip"

    multi_agent_cross_train(
        num_iterations=iteration,
        professions=professions,
        skill_mgr=skill_mgr,
        save_path_1=save_path_1,
        save_path_2=save_path_2
    )
    return jsonify({"message": f"訓練完成 {iteration} 次疊代。"})
    
@main_routes.route("/api/computer_vs_computer", methods=["GET"])
def api_computer_vs_computer():
    """
    (5) 電腦 VS 電腦
    """
    battle_log = computer_vs_computer(skill_mgr, professions)
    return jsonify({"battle_log": battle_log})
    
@main_routes.route("/api/version_test", methods=["GET"])
def api_version_test():
    """
    (2) 版本環境測試 => 隨機 VS 隨機
    """
    # find in data/cross_validation to find if the version same as globalVar
    # traverse all the data in cross_validation
    hasData = False
    for i in os.listdir("data/cross_validation_pc"):
        print(f"data/cross_validation_pc/{i}")
        with open(f"data/cross_validation_pc/{i}", 'r') as f:
            data = json.load(f)
            print('dv',data["version"],'gv',globalVar["version"])
            if data["version"] == globalVar["version"]:
                hasData = True
                
    
    
    # if not find the same version
    if not hasData:
        res = version_test_random_vs_random(professions, skill_mgr, num_battles=100)
        resGdata = Gdata(res,globalVar["version"],"cross_validation_pc")
        resGdata.save()

    all_data = []
    for i in os.listdir("data/cross_validation_pc"):
        with open(f"data/cross_validation_pc/{i}", 'r') as f:
            data = json.load(f)
            all_data.append(data)
    return jsonify({"message": all_data})

    
    
@main_routes.route("/api/show_professions", methods=["GET"])
def api_show_professions():
    """
    (8) 各職業介紹
    回傳結構化的職業資料
    """
    professions_data = get_professions_data(professions, skill_mgr)
    return jsonify({"professions_info": professions_data})

# 你還可以繼續加其它端點 ...
