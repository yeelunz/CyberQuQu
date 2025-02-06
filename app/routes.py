from sklearn.manifold import TSNE
import random
import logging
import numpy as np
import os
from flask import Blueprint, request, jsonify, render_template, Response, stream_with_context
from utils.versus import (

    computer_vs_computer,
    ai_vs_ai,

)
from utils.skills import build_skill_manager
from utils.professions import build_professions
from utils.train_methods import multi_agent_cross_train, stop_training_flag , make_env_config
from utils.cross_model_evaluation import version_test_model_vs_model_generate_sse
from utils.single_model_evaluation import version_test_random_vs_random_sse_ai, compute_ai_elo
from utils.pc_evaluaiton import version_test_random_vs_random_sse
import time
from utils.battle_env import BattleEnv
from utils.data_stamp import Gdata
import json
import uuid
import queue
from utils.global_var import globalVar
from utils.profession_var import (
    PALADIN_VAR, MAGE_VAR, ASSASSIN_VAR, ARCHER_VAR, BERSERKER_VAR,
    DRAGONGOD_VAR, BLOODGOD_VAR, STEADFASTWARRIOR_VAR, DEVOUR_VAR, RANGER_VAR,
    ELEMENTALMAGE_VAR, HUANGSHEN_VAR, GODOFSTAR_VAR
)
import copy

# 將所有職業的變數整理到一個全域 dict 中 (注意：此處只是在記憶體中修改，重新啟動後會回復原始設定)


main_routes = Blueprint('main', __name__)




# 建立全域的 manager & professions，避免每次端點都要重建



@main_routes.route("/api/list_models", methods=["GET"])
def list_models():
    """
    列出 data/saved_models/ 下所有模型資料夾及其 training_meta.json 內容
    回傳:
    [
      {
        "folder_name": <str>,
        "meta": { ... }
      },
      ...
    ]
    """
    base_path = os.path.join("data", "saved_models")
    if not os.path.exists(base_path):
        return jsonify([]), 200

    folder_list = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        meta_path = os.path.join(folder_path, "training_meta.json")
        if not os.path.exists(meta_path):
            # 若沒有 meta 檔，略過
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        folder_list.append({
            "folder_name": folder_name,
            "meta": meta_data
        })

    return jsonify(folder_list), 200


# route.py

@main_routes.route("/api/compute_elo_sse", methods=["GET"])
def compute_elo_sse():
    folder_name = request.args.get("folder_name")
    if not folder_name:
        return jsonify({"success": False, "message": "缺少 folder_name"}), 400

    base_path = os.path.join("data", "saved_models")
    model_path = os.path.join(base_path, folder_name)
    meta_path = os.path.join(model_path, "training_meta.json")

    if not os.path.exists(meta_path):
        return jsonify({"success": False, "message": "找不到該模型"}), 404

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    def generate_stream():
        professions = build_professions()
        skill_mgr = build_skill_manager()

        elo_generator = compute_ai_elo(
            model_path_1=model_path,
            professions=professions,
            skill_mgr=skill_mgr,
            base_elo=1500,
            opponent_elo=1500,
            num_battles=100,
            K=32
        )

        elo_result = None  # 先預設 None，等收到 final_result 後再記下來

        try:
            for event in elo_generator:
                # 如果偵測到 "type": "final_result"
                if event.get("type") == "final_result":
                    elo_result = event.get("elo_result", None)
                    # 也一併把事件繼續往前端丟（或你也可選擇不丟，自己內部處理就好）
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                else:
                    # 其他事件 ("progress"等) 照樣往前端丟
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 走到這裡代表 compute_ai_elo 已經結束迭代了
            # 如果剛剛有拿到 final_result，就寫進 meta_data
            if elo_result is not None:
                meta_data["elo_result"] = elo_result
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)

                # 最後再yield一個 "done" 事件
                done_event = {
                    'type': 'done',
                    'message': 'ELO 計算完成',
                    'new_elo_result': elo_result
                }
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        except Exception as e:
            # 任意其他例外
            error_event = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')


@main_routes.route('/api/train_sse')
def train_sse():
    """
    SSE 端點: 接收 Query Params，開始多智能體訓練，
    每個 iteration 完成後馬上 send 一個 event 到前端。
    """

    model_name = request.args.get("model_name", "my_multiagent_ai")
    iteration = int(request.args.get("iteration", 5))
    hyperparams_json = request.args.get("hyperparams_json", "{}")
    hyperparams = json.loads(hyperparams_json)

    def generate_stream():
        train_generator = multi_agent_cross_train(
            num_iterations=iteration,
            model_name=model_name,
            hyperparams=hyperparams
        )
        # 迭代拿到每筆資訊 => SSE 格式送到前端
        for result in train_generator:
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

    # 在一個新的執行緒中運行 SSE 生成器，以避免阻塞主線程
    return Response(generate_stream(), mimetype='text/event-stream')


@main_routes.route('/api/stop_train', methods=['POST'])
def stop_train():
    """
    終止訓練的端點
    """
    if not stop_training_flag.is_set():
        stop_training_flag.set()
        return jsonify({"message": "終止訓練請求已收到。"}), 200
    else:
        return jsonify({"message": "訓練已經在終止中。"}), 400




@main_routes.route("/api/list_professions", methods=["GET"])
def list_professions():
    """
    回傳所有職業的 name (list[str])
    """
    prof_objs = build_professions()
    names = [p.name for p in prof_objs]  # 例如 ["聖騎士", "狂戰士", ...]
    return jsonify({"professions": names}), 200


@main_routes.route("/api/list_saved_models_simple", methods=["GET"])
def list_saved_models_simple():
    """
    列出 data/saved_models/ 下所有模型的資料夾名稱 => ["modelA", "modelB", ...]
    """
    base_path = os.path.join("data", "saved_models")
    if not os.path.exists(base_path):
        return jsonify({"models": []}), 200

    model_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            model_folders.append(item)
    return jsonify({"models": model_folders}), 200


def find_profession_by_name(name: str, prof_list):
    """
    從 prof_list (list of class物件) 中，根據 p.name == name 找出對應的職業物件。
    找不到就回傳 None
    """
    for p in prof_list:
        if p.name == name:
            return p
    return None


@main_routes.route("/api/computer_vs_computer", methods=["GET"])
def api_computer_vs_computer():
    """
    PC vs PC
    GET 參數: pr1, pr2 (職業名稱 or "Random")
    """
    pr1_name = request.args.get("pr1", "Random")
    pr2_name = request.args.get("pr2", "Random")

    professions = build_professions()  # [Paladin(), Mage(), ...]

    if pr1_name == "Random":
        pr1_obj = random.choice(professions)
    else:
        pr1_obj = find_profession_by_name(pr1_name, professions)
        if pr1_obj is None:
            pr1_obj = random.choice(professions)

    if pr2_name == "Random":
        pr2_obj = random.choice(professions)
    else:
        pr2_obj = find_profession_by_name(pr2_name, professions)
        if pr2_obj is None:
            pr2_obj = random.choice(professions)

    battle_log = computer_vs_computer(
        skill_mgr=build_skill_manager(),
        professions=professions,  # optional
        pr1=pr1_obj,
        pr2=pr2_obj
    )

    return jsonify({"battle_log": battle_log})


@main_routes.route("/api/ai_vs_ai", methods=["GET"])
def api_ai_vs_ai():
    """
    AI vs AI
    GET 參數: pr1, pr2 (職業名稱 or "Random"), model1, model2 (模型名稱)
    """
    pr1_name = request.args.get("pr1", "Random")
    pr2_name = request.args.get("pr2", "Random")
    print("pr1_name", pr1_name, "pr2_name", pr2_name)
    model1 = request.args.get("model1", "")  # 可能是空
    model2 = request.args.get("model2", "")  # 可能是空

    professions = build_professions()

    if pr1_name == "Random":
        pr1_obj = random.choice(professions)
    else:
        pr1_obj = find_profession_by_name(pr1_name, professions)
        if pr1_obj is None:
            print("pr1_obj is None")
            pr1_obj = random.choice(professions)

    if pr2_name == "Random":
        pr2_obj = random.choice(professions)
    else:
        pr2_obj = find_profession_by_name(pr2_name, professions)
        if pr2_obj is None:
            print("pr2_obj is None")
            pr2_obj = random.choice(professions)

    # ---- 若使用者沒選擇模型，就回傳 400，避免 NoneType trainer ---
    if not model1 or not model2:
        return jsonify({
            "error": "必須同時為左右AI選擇模型"
        }), 400

    # --- 同模型 or 不同模型 ---
    same_model = (model1 == model2)

    # 轉成絕對路徑 => 解決 pyarrow fs 無法辨識相對路徑
    model_path_1 = os.path.abspath(
        os.path.join("data", "saved_models", model1))
    model_path_2 = os.path.abspath(
        os.path.join("data", "saved_models", model2))

    # 檢查檔案/資料夾是否存在(避免出錯)
    if not os.path.exists(model_path_1):
        return jsonify({"error": f"模型 {model1} 不存在"}), 400
    if not os.path.exists(model_path_2):
        return jsonify({"error": f"模型 {model2} 不存在"}), 400

    # --- 執行對戰 ---
    try:
        battle_log = ai_vs_ai(
            skill_mgr=build_skill_manager(),
            professions=professions,
            model_path_1=model_path_1,
            model_path_2=model_path_2,
            pr1=pr1_obj,
            pr2=pr2_obj,
            SameModel=same_model
        )
        return jsonify({"battle_log": battle_log}), 200

    except Exception as e:
        # 如果在 restore/checkpoint 時出錯 => return 500
        print("AI vs AI 發生例外：", e)
        return jsonify({"error": str(e)}), 500


@main_routes.route("/api/version_test_generate", methods=["GET"])
def api_version_test_generate_sse():
    """
    SSE 產生交叉對戰數據 (PC vs PC 或 AI vs AI)。
    前端以 GET + QueryString 的方式連線:
      GET /api/version_test_generate?mode=pc&model_path=xxx&num_battles=100
    """

    # 1) 取得參數 (皆為字串，需要自行轉型)
    mode = request.args.get("mode", "pc")       # 預設 pc
    model_path = request.args.get("model_path", "")
    num_battles = request.args.get("num_battles", "100")
    try:
        num_battles = int(num_battles)
    except:
        num_battles = 100  # fallback

    # 2) 依照 mode 決定使用何種對戰生成器
    #    PC vs PC => version_test_random_vs_random_sse
    #    AI vs AI => version_test_random_vs_random_sse_ai (你自己實作好的)

    if mode == "ai":
        # 檢查 model_path 是否真的存在

        base_path = os.path.join("data", "saved_models")
        model_path = os.path.join(base_path, model_path)
        if not os.path.exists(model_path):
            # 若找不到，直接用 SSE 回傳錯誤訊息即可
            def error_stream():
                err = {
                    "type": "error",
                    "message": f"找不到模型: {model_path}"
                }
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return Response(stream_with_context(error_stream()), mimetype='text/event-stream')

        print("model_full_path", model_path)
        generator = version_test_random_vs_random_sse_ai(
            professions = build_professions(),
            skill_mgr = build_skill_manager(),
            num_battles=num_battles,
            model_path_1=model_path
        )
    else:
        # mode == "pc"
        generator = version_test_random_vs_random_sse(
            professions = build_professions(),
            skill_mgr = build_skill_manager(),
            num_battles=num_battles
        )

    # 3) SSE 串流函式
    def sse_stream():
        for info in generator:
            # info 為 dict，例如 { "type": "progress", "progress": 50, "message": "..."}
            yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"

    # 4) 回傳 SSE
    response = Response(stream_with_context(sse_stream()),
                        mimetype='text/event-stream')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"  # 關閉 nginx buffering (若有用到)
    return response


@main_routes.route("/api/version_test", methods=["GET"])
def api_version_test():
    """
    (原本) 顯示交叉對戰數據 => PC
    不再自動產生，而是直接去 data/cross_validation_pc/ 搜尋並顯示。
    """

    all_data = []
    for i in os.listdir("data/cross_validation_pc"):
        with open(f"data/cross_validation_pc/{i}", 'r') as f:
            data = json.load(f)
            all_data.append(data)
    for i in os.listdir("data/cross_validation_ai"):
        with open(f"data/cross_validation_ai/{i}", 'r') as f:
            data = json.load(f)
            all_data.append(data)

    return jsonify({"message": all_data})


def get_professions_data(profession_list, skill_mgr):
    """
    將職業資訊轉換為結構化的資料格式，不包含 skill_id。
    """
    professions_data = []
    for p in profession_list:
        profession = {
            "name": p.name,
            "hp": p.base_hp,
            "attack_coeff": p.baseAtk,
            "defense_coeff": p.baseDef,
            "passive": {
                "name": p.passive_name,
                "description": p.passive_desc
            },
            "skills": []
        }
        skill_ids = p.get_available_skill_ids({0: 0, 1: 0, 2: 0, 3:0})
        for sid in skill_ids:
            skill = skill_mgr.skills.get(sid)
            if skill:
                skill_info = {
                    "name": skill.name,
                    "description": skill.desc,
                    "cooldown": skill.cool_down if skill.cool_down > 0 else None,
                    "type": skill.type
                }
                profession["skills"].append(skill_info)
        professions_data.append(profession)
    return professions_data


@main_routes.route("/api/show_professions", methods=["GET"])
def api_show_professions():
    """
    (8) 各職業介紹
    回傳結構化的職業資料
    """
    
    professions_data = get_professions_data(build_professions(), build_skill_manager())
    return jsonify({"professions_info": professions_data})


@main_routes.route("/api/model_vs_model_result", methods=["GET"])
def api_model_vs_model_result():
    """
    取得先前儲存的模型間對戰資料
    GET 參數：
      result_id (識別碼)
    """
    result_id = request.args.get("result_id", "")
    if not result_id:
        return jsonify({"error": "缺少 result_id"}), 400

    # 假設你有函式 load_model_vs_model_result(result_id) 負責讀取資料
    data = load_model_vs_model_result(result_id)
    if not data:
        return jsonify({"error": "找不到資料"}), 404

    return jsonify(data), 200

# 請自行實作或調整 load_model_vs_model_result 函式


def load_model_vs_model_result(result_id):
    # 範例：從檔案中讀取 (實際請依你存檔邏輯調整)
    result_file = os.path.join("data", "results", f"{result_id}.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@main_routes.route("/api/version_test_generate_model_vs_model", methods=["GET"])
def api_version_test_generate_model_vs_model():
    """
    SSE 產生模型間對戰數據
    GET 參數：
      modelA, modelB (模型名稱)
      num_battles (每組進行場數)
    """
    modelA = request.args.get("modelA", "")
    modelB = request.args.get("modelB", "")
    num_battles = request.args.get("num_battles", "100")
    try:
        num_battles = int(num_battles)
    except:
        num_battles = 100

    if not modelA or not modelB:
        return jsonify({"error": "必須選擇模型A與模型B"}), 400

    model_path_A = os.path.abspath(
        os.path.join("data", "saved_models", modelA))
    model_path_B = os.path.abspath(
        os.path.join("data", "saved_models", modelB))
    if not os.path.exists(model_path_A):
        return jsonify({"error": f"模型 {modelA} 不存在"}), 400
    if not os.path.exists(model_path_B):
        return jsonify({"error": f"模型 {modelB} 不存在"}), 400

    # 取得 professions 與 skill_mgr（請依你專案調整）
    professions = build_professions()
    skill_mgr = build_skill_manager()

    generator = version_test_model_vs_model_generate_sse(
        professions, skill_mgr, num_battles, model_path_A, model_path_B)

    def sse_stream():
        for info in generator:
            yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"

    response = Response(stream_with_context(sse_stream()),
                        mimetype='text/event-stream')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@main_routes.route("/api/list_model_vs_results", methods=["GET"])
def show_list_model_vs_model():
    """
    列出 /data/model_vs 下所有 JSON 檔案，並回傳檔案清單。
    每筆資料包含：檔案名稱 (作為識別 id)、model_A、model_B、timestamp
    """
    # logger.debug("進入 /api/list_model_vs_results")
    results = []
    base_dir = os.path.join(os.getcwd(), "data", "model_vs")
    # logger.debug(f"base_dir: {base_dir}")
    if not os.path.exists(base_dir):
        # logger.debug("目錄不存在")
        return jsonify({"results": []})
    for filename in os.listdir(base_dir):
        # logger.debug(f"處理檔案: {filename}")
        if filename.endswith(".json"):
            filepath = os.path.join(base_dir, filename)
            # logger.debug(f"完整路徑: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                # logger.debug(f"讀取內容: {content}")
                results.append({
                    "id": filename,  # 以檔名作為識別碼
                    "model_A": content.get("model_A", ""),
                    "model_B": content.get("model_B", ""),
                    "timestamp": content.get("timestamp", "")
                })
            except Exception as e:
                logger.error(f"讀取 {filename} 時發生錯誤: {e}")
                continue
    # logger.debug(f"回傳結果: {results}")
    return jsonify({"results": results})


@main_routes.route("/api/model_vs_model_result_json", methods=["GET"])
def show_list_model_vs_model_json():
    """
    根據 query string 中的 result_id，讀取 /data/model_vs 下對應檔案的 JSON 內容回傳。
    """
    result_id = request.args.get("result_id", "").strip()
    # logger.debug(f"收到 result_id: {result_id}")
    if not result_id:
        # logger.debug("缺少 result_id")
        return jsonify({"error": "缺少 result_id"}), 400

    # 強制print 這個result_id
    # logger.debug(f"強制print 這個result_id: {result_id}")

    base_dir = os.path.join(os.getcwd(), "data", "model_vs")
    # logger.debug(f"base_dir: {base_dir}")
    # 列出目前可用的檔案，方便除錯
    available_files = os.listdir(base_dir) if os.path.exists(base_dir) else []
    # logger.debug(f"目前 available_files: {available_files}")

    filepath = os.path.join(base_dir, result_id)
    # logger.debug(f"組成的 filepath: {filepath}")

    if not os.path.exists(filepath):
        logger.debug(f"檔案 {filepath} 不存在")
        return jsonify({"error": f"檔案 {result_id} 不存在. Available files: {available_files}"}), 404
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
        # logger.debug(f"檔案內容: {content}")
        return jsonify(content)
    except Exception as e:
        logger.error(f"讀取檔案 {result_id} 時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


eff_id_to_name = {
    1: "攻擊力變更", 2: "防禦力變更", 3: "治癒力變更", 4: "燃燒", 5: "中毒",
    6: "凍結", 7: "免疫傷害", 8: "免疫控制", 9: "流血", 10: "麻痺",
    11: "生命力持續變更", 12: "無效效果", 13: "追蹤"
}

skill_id_to_name = {
    0: "聖光斬", 1: "堅守防禦", 2: "神聖治療",
    3: "決一死戰", 4: "火焰之球", 5: "冰霜箭", 6: "全域爆破", 7: "詠唱破棄．全域爆破", 8: "致命暗殺",
    9: "毒爆", 10: "毒刃襲擊", 11: "致命藥劑", 12: "五連矢", 13: "箭矢補充", 14: "吸血箭",
    15: "驟雨", 16: "狂暴之力", 17: "熱血", 18: "血怒之泉", 19: "嗜血本能", 20: "神龍之息", 21: "龍血之泉",
    22: "神龍燎原", 23: "預借", 24: "血刀", 25: "血脈祭儀", 26: "轉生", 27: "新生", 28: "剛毅打擊", 29: "不屈意志",
    30: "絕地反擊", 31: "破魂斬", 32: "吞裂", 33: "巨口吞世", 34: "堅硬皮膚", 35: "觸電反應",
    36: "續戰攻擊", 37: "埋伏防禦", 38: "荒原抗性", 39: "地雷", 40: "融合", 41: "雷霆護甲",
    42: "寒星墜落", 43: "焚天", 44: "枯骨",  45: "荒原", 46: "生命逆流", 47: "風化", 48: "災厄隕星",  49: "光輝流星",
    50: "虛擬創星圖", 51: "無序聯星"
}

profession_id_to_name = {
    0: "聖騎士", 1: "法師", 2: "刺客", 3: "弓箭手", 4: "狂戰士",
    5: "龍神", 6: "血神", 7: "剛毅武士", 8: "鯨吞", 9: "荒原遊俠",
    10: "元素法師", 11: "荒神", 12: "星神"
}


@main_routes.route("/api/embedding/list_models", methods=["GET"])
def list_embedding_models():
    """
    列出 data/saved_models/ 下所有擁有 embeddings.json 的模型資料夾。
    """
    base_path = os.path.join("data", "saved_models")
    if not os.path.exists(base_path):
        return jsonify([]), 200

    models = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 檢查是否有 embeddings.json
        embed_path = os.path.join(folder_path, "embeddings.json")
        if not os.path.exists(embed_path):
            continue

        # 讀取 training_meta.json (若存在)
        meta_data = {}
        meta_path = os.path.join(folder_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        meta_data["has_embedding"] = True

        models.append({
            "folder_name": folder_name,
            "meta": meta_data
        })

    return jsonify(models), 200


@main_routes.route("/api/embedding/get", methods=["GET"])
def get_embedding():
    """
    讀取指定模型資料夾中的 embeddings.json，
    並依據 query 參數 dim (2 或 3) 使用 t-SNE 降維後回傳資料。
    """
    model_name = request.args.get("model", "")
    dim_str = request.args.get("dim", "2")
    try:
        dim = int(dim_str)
        if dim not in (2, 3):
            return jsonify({"error": "dim must be 2 or 3"}), 400
    except Exception:
        return jsonify({"error": "dim must be 2 or 3"}), 400

    base_path = os.path.join("data", "saved_models", model_name)
    embed_path = os.path.join(base_path, "embeddings.json")
    if not os.path.exists(embed_path):
        return jsonify({"error": f"embeddings.json not found for model: {model_name}"}), 404

    with open(embed_path, "r", encoding="utf-8") as f:
        embed_data = json.load(f)

    # 根據 JSON 結構決定要處理的 key
    if "profession_p" in embed_data:
        keys = ["profession_p", "profession_e",
                "skill_p", "skill_e", "effect_p", "effect_e"]
    else:
        keys = ["profession", "skill", "effect"]

    result_categories = []
    for key in keys:
        if key not in embed_data:
            continue

        # 將原始資料轉為 numpy 陣列
        arr = np.array(embed_data[key])
        n_samples = arr.shape[0]

        # 根據不同類別設定預設 perplexity (skill 類較多，預設 15，其它預設 5)
        default_perplexity = 15 if "skill" in key else 5
        perplexity = default_perplexity if default_perplexity < n_samples else max(
            1, n_samples - 1)
        try:
            tsne = TSNE(n_components=dim,
                        perplexity=perplexity, random_state=42)
            reduced = tsne.fit_transform(arr)
        except Exception as e:
            return jsonify({"error": f"TSNE failed for {key}: {str(e)}"}), 500

        cat_data = {"name": key}
        if dim == 2:
            cat_data["x"] = reduced[:, 0].tolist()
            cat_data["y"] = reduced[:, 1].tolist()
        else:
            cat_data["x"] = reduced[:, 0].tolist()
            cat_data["y"] = reduced[:, 1].tolist()
            cat_data["z"] = reduced[:, 2].tolist()

        # 根據 key 類型，為每個點附上中文標籤
        labels = []
        for i in range(n_samples):
            # 預設 label 為索引
            label = str(i)
            if "profession" in key:
                label = profession_id_to_name.get(i, str(i))
            elif "skill" in key:
                label = skill_id_to_name.get(i, str(i))
            elif "effect" in key:
                # 這裡假設 effect 的順序與 mapping 中 key 值相差 1 (即 index 0 對應 mapping key 1)
                label = eff_id_to_name.get(i + 1, str(i))
            labels.append(label)
        cat_data["labels"] = labels

        result_categories.append(cat_data)

    return jsonify({"categories": result_categories}), 200



from ray.rllib.algorithms.ppo import PPOConfig

from utils.versus import current_loaded_model_path_1, current_loaded_model_path_2   
from utils.versus import current_trainer_1, current_trainer_2

pva_sessions = {}  # 用 session_id -> {...} 做儲存


def get_skill_info(skill_id, skill_mgr):
    """
    取得技能的資訊，包括名稱、描述、冷卻時間、效果類型。
    """
    skill = skill_mgr.skills.get(skill_id)
    if skill:
        return {
            "name": skill.name,
            "description": skill.desc,
            "cooldown": skill.cool_down if skill.cool_down > 0 else 0,
            "type": skill.type
        }
    return {
        "name": f"Skill-{skill_id}",
        "description": "",
        "cooldown": 0,
        "type": "other"
    }


def create_pva_environment(skill_mgr, professions, model_path_1, model_path_2, pr1, pr2, SameModel=False):
    """
    建立「玩家 vs AI」模式所需的環境與訓練器。
    回傳 (env, trainer_player, state_player, trainer_enemy, state_enemy, obs, info)。
    """
    global current_loaded_model_path_1, current_loaded_model_path_2
    global current_trainer_1, current_trainer_2

    # 讀取 model_path_1
    fc_hiddens_1 = [256, 256, 256]
    max_seq_len_1 = 10
    mask_model_1 = "my_mask_model"
    if model_path_1 is not None:
        cp_path_1 = os.path.abspath(model_path_1)
        if os.path.exists(cp_path_1):
            meta_path1 = os.path.join(cp_path_1, "training_meta.json")
            try:
                with open(meta_path1, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
                    hyperparams = meta_info.get("hyperparams", {})
                    fc_hiddens_1 = hyperparams.get("fcnet_hiddens", [256, 256, 256])
                    max_seq_len_1 = hyperparams.get("max_seq_len", 10)
                    mask_model_1 = hyperparams.get("mask_model", "my_mask_model")
            except FileNotFoundError:
                print(f"[Warn] 找不到 {meta_path1}，使用預設 fcnet_hiddens。")

    # 讀取 model_path_2
    fc_hiddens_2 = [256, 256, 256]
    max_seq_len_2 = 10
    mask_model_2 = "my_mask_model"
    if model_path_2 is not None:
        cp_path_2 = os.path.abspath(model_path_2)
        if os.path.exists(cp_path_2):
            meta_path2 = os.path.join(cp_path_2, "training_meta.json")
            try:
                with open(meta_path2, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
                    hyperparams = meta_info.get("hyperparams", {})
                    fc_hiddens_2 = hyperparams.get("fcnet_hiddens", [256, 256, 256])
                    max_seq_len_2 = hyperparams.get("max_seq_len", 10)
                    mask_model_2 = hyperparams.get("mask_model", "my_mask_model")
            except FileNotFoundError:
                print(f"[Warn] 找不到 {meta_path2}，使用預設 fcnet_hiddens。")

    # 建立環境 config
    beconfig = make_env_config(
        skill_mgr=skill_mgr,
        professions=professions,
        show_battlelog=True,
        pr1=pr1,
        pr2=pr2
    )
    # 先建立一個 env 來拿 obs/act space
    benv = BattleEnv(config=beconfig)

    config_left = (
        PPOConfig()
        .environment(env=BattleEnv, env_config=beconfig)
        .env_runners(num_env_runners=1, sample_timeout_s=120)
        .framework("torch")
        .training(
            model={
                "custom_model": mask_model_1,
                "fcnet_hiddens": fc_hiddens_1,
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": max_seq_len_1
            }
        )
    )
    config_right = (
        PPOConfig()
        .environment(env=BattleEnv, env_config=beconfig)
        .env_runners(num_env_runners=1, sample_timeout_s=120)
        .framework("torch")
        .training(
            model={
                "custom_model": mask_model_2,
                "fcnet_hiddens": fc_hiddens_2,
                "fcnet_activation": "ReLU",
                "vf_share_layers": False,
                "max_seq_len": max_seq_len_2
            }
        )
    )
    # 多代理
    config_left = config_left.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: "shared_policy",
    )
    config_right = config_right.multi_agent(
        policies={
            "shared_policy": (None, benv.observation_space, benv.action_space, {}),
        },
        policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: "shared_policy",
    )
    # 關閉 API
    config_left = config_left.api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
    config_right = config_right.api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )

    # 真正建立 env
    env = BattleEnv(config=beconfig)
    obs, info = env.reset()

    # 判斷是否要重新載入
    global current_trainer_1, current_trainer_2
    need_reload_1 = (model_path_1 is not None and model_path_1 != current_loaded_model_path_1)
    need_reload_2 = (model_path_2 is not None and model_path_2 != current_loaded_model_path_2 and not SameModel)

    if SameModel:
        if need_reload_1:
            current_trainer_1 = config_left.build()
            print("[Info] 載入模型(同一模型):", model_path_1)
            current_trainer_1.restore(model_path_1)
            current_loaded_model_path_1 = model_path_1
        trainer1 = current_trainer_1
        trainer2 = current_trainer_1
        current_loaded_model_path_2 = model_path_1
    else:
        if need_reload_1:
            current_trainer_1 = config_left.build()
            print("[Info] 載入模型(左):", model_path_1)
            current_trainer_1.restore(model_path_1)
            current_loaded_model_path_1 = model_path_1

        if need_reload_2:
            current_trainer_2 = config_right.build()
            print("[Info] 載入模型(右):", model_path_2)
            current_trainer_2.restore(model_path_2)
            current_loaded_model_path_2 = model_path_2

        trainer1 = current_trainer_1
        trainer2 = current_trainer_2

    if trainer1 is None or trainer2 is None:
        print("[Error] 載入模型失敗")
        return None

    policy_left = trainer1.get_policy("shared_policy")
    policy_right = trainer2.get_policy("shared_policy")
    state_left = policy_left.get_initial_state()
    state_right = policy_right.get_initial_state()

    trainer_player = trainer1
    state_player = state_left
    trainer_enemy = trainer2
    state_enemy = state_right

    return env, trainer_player, state_player, trainer_enemy, state_enemy, obs, info


@main_routes.route("/api/player_vs_ai_init", methods=["POST"])
def player_vs_ai_init():
    data = request.json or {}
    pr_name_player = data.get("player_profession", "Random")
    pr_name_enemy = data.get("enemy_profession", "Random")
    model_name_1 = data.get("model1", "")
    model_name_2 = data.get("model2", "")

    if not model_name_1 or not model_name_2:
        return jsonify({"error": "必須提供 model1 和 model2"}), 400

    professions = build_professions()
    if pr_name_player == "Random":
        pr_player = random.choice(professions)
    else:
        pr_player = find_profession_by_name(pr_name_player, professions) or random.choice(professions)

    if pr_name_enemy == "Random":
        pr_enemy = random.choice(professions)
    else:
        pr_enemy = find_profession_by_name(pr_name_enemy, professions) or random.choice(professions)

    model_path_1 = os.path.abspath(os.path.join("data", "saved_models", model_name_1))
    model_path_2 = os.path.abspath(os.path.join("data", "saved_models", model_name_2))
    if not os.path.exists(model_path_1) or not os.path.exists(model_path_2):
        return jsonify({"error": "模型不存在"}), 400

    SameModel = (model_path_1 == model_path_2)
    skill_mgr = build_skill_manager()

    env_result = create_pva_environment(
        skill_mgr=skill_mgr,
        professions=professions,
        model_path_1=model_path_1,
        model_path_2=model_path_2,
        pr1=pr_player,
        pr2=pr_enemy,
        SameModel=SameModel
    )
    if env_result is None:
        return jsonify({"error": "建立環境失敗"}), 500

    env, trainer_player, state_player, trainer_enemy, state_enemy, initial_obs, initial_info = env_result

    session_id = str(uuid.uuid4())
    msg_queue = queue.Queue()

    # 建立 session 資料
    pva_sessions[session_id] = {
        "env": env,
        "trainer_player": trainer_player,
        "state_player": state_player,
        "trainer_enemy": trainer_enemy,
        "state_enemy": state_enemy,
        "skill_mgr": skill_mgr,
        "professions": professions,
        "msg_queue": msg_queue,
        "done": False,
        "last_obs": initial_obs,
        "last_info": initial_info,
    }

    # 這裡以 env.player_team[0] / env.enemy_team[0] 為實際角色資訊
    p = copy.deepcopy(env.player_team[0])   # Player
    e = copy.deepcopy(env.enemy_team[0])    # Enemy
    ppid = p['profession'].profession_id
    epid = e['profession'].profession_id

    # 取得左方角色技能
    left_skill_data = []
    for i in range(4):
        sid = ppid * 4 + i
        left_skill_data.append(get_skill_info(sid, skill_mgr))
    # 取得右方角色技能
    right_skill_data = []
    for i in range(4):
        sid = epid * 4 + i
        right_skill_data.append(get_skill_info(sid, skill_mgr))

    # 組裝初始狀態
    initial_status = {
        "global": {
            "left_profession": p['profession'].name,
            "right_profession": e['profession'].name,
            "round": env.round_count,
            "max_rounds": env.max_rounds,
            "damage_coefficient": 1.0
        },
        "left": {
            "hp": p['hp'],
            "max_hp": p['max_hp'],
            "multiplier": {
                "damage": 1.0,
                "defend": 1.0,
                "heal": 1.0
            },
            "effects": [],
            # cooldowns 為 dict
            "cooldowns": {0:0, 1:0, 2:0, 3:0},
            "skill_info": left_skill_data,
        },
        "right": {
            "hp": e['hp'],         # 修正：原本誤填 e['profession'].name
            "max_hp": e['max_hp'],
            "multiplier": {
                "damage": 1.0,
                "defend": 1.0,
                "heal": 1.0
            },
            "effects": [],
            "cooldowns": {0:0, 1:0, 2:0, 3:0},
            "skill_info": right_skill_data,
        }
    }

    # 推送 SSE
    msg_queue.put({
        "type": "refresh_status",
        "appendix": initial_status
    })

    return jsonify({"session_id": session_id, "msg": "ok"}), 200


from dataclasses import is_dataclass, asdict

def serialize_data(obj):
    """遞迴將 dataclass 物件轉成 dict，其它物件保持不變。"""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(item) for item in obj]
    else:
        return obj


@main_routes.route("/api/player_vs_ai_stream/<session_id>", methods=["GET"])
def player_vs_ai_stream(session_id):
    """SSE: 持續推送 msg_queue 內容給前端"""
    if session_id not in pva_sessions:
        return jsonify({"error": "invalid session_id"}), 400

    def sse_stream():
        q = pva_sessions[session_id]["msg_queue"]
        while True:
            data = q.get()  # 會阻塞
            # 將資料中可能含有 BattleEvent 的物件轉換成 dict（可被 json.dumps 處理）
            serializable_data = serialize_data(data)
            yield f"data: {json.dumps(serializable_data)}\n\n"
            time.sleep(0.01)

    return Response(sse_stream(), mimetype="text/event-stream")


@main_routes.route("/api/player_vs_ai_step/<session_id>", methods=["POST"])
def player_vs_ai_step(session_id):
    if session_id not in pva_sessions:
        return jsonify({"error": "invalid session_id"}), 400

    storage = pva_sessions[session_id]
    if storage["done"]:
        return jsonify({"message": "battle already ended"}), 200

    data = request.json or {}
    skill_idx = data.get("skill_idx", 0)

    env = storage["env"]
    trainer_enemy = storage["trainer_enemy"]
    state_enemy = storage["state_enemy"]

    last_obs = storage["last_obs"]
    last_info = storage["last_info"]

    # 敵方 obs
    enemy_obs = last_obs["enemy"]

    # 我方動作
    p_act = skill_idx

    # AI 動作
    policy_enemy = trainer_enemy.get_policy("shared_policy")
    e_act_package = policy_enemy.compute_single_action(enemy_obs, state_enemy, policy_id="shared_policy")
    e_act = e_act_package[0]
    new_state_enemy = e_act_package[1]

    actions = {"player": p_act, "enemy": e_act}
    new_obs, rew, done_dict, _, info = env.step(actions)
    done = done_dict["__all__"]

    storage["state_enemy"] = new_state_enemy
    storage["last_obs"] = new_obs
    storage["last_info"] = info

    # 本回合 battle log
    cur_log = env.cur_round_battle_log
    storage["msg_queue"].put({
        "type": "round_result",
        "round_battle_log": cur_log,
    })

    if done:
        res = info["__common__"]["result"]
        result_text = "玩家贏了" if res == 1 else ("AI贏了" if res == -1 else "平手/回合耗盡")
        storage["msg_queue"].put({
            "type": "battle_end",
            "winner": res,
            "winner_text": result_text,
        })
        storage["done"] = True
    print("--對戰完成--")
    return jsonify({"done": done}), 200


@main_routes.route("/api/player_vs_ai_hint/<session_id>", methods=["GET"])
def player_vs_ai_hint(session_id):
    """
    提供玩家 Hint: 從 obs["player"] 丟入 trainer_player 做推論
    """
    if session_id not in pva_sessions:
        return jsonify({"error": "invalid session_id"}), 400

    storage = pva_sessions[session_id]
    if storage["done"]:
        return jsonify({"message": "battle ended"}), 200

    env = storage["env"]
    trainer_player = storage["trainer_player"]

    last_obs = storage["last_obs"]
    player_obs = last_obs.get("player")
    if not player_obs:
        return jsonify({"error": "no player obs"}), 400

    policy_hint = trainer_player.get_policy("shared_policy")
    action_package = policy_hint.compute_single_action(player_obs, storage["state_player"], policy_id="shared_policy")
    action = action_package[0]
    storage["state_player"] = action_package[1]

    skill_id = env.config["professions"]["player"].skills[action]
    skill_info = get_skill_info(skill_id, storage["skill_mgr"])

    return jsonify({
        "action": action,
        "skill_name": skill_info["name"],
        "skill_desc": skill_info["description"],
        "probabilities": [0, 0, 0, 0]  # 假設不顯示詳細機率
    }), 200

@main_routes.route('/get_token')
def get_token():
    token = globalVar['token']
    return str(token)

@main_routes.route('/get_version')
def get_version():
    version = globalVar['version']
    return str(version)

@main_routes.route('/')
def index():
    token = get_token()
    version = get_version()
    # 將 token 與 version 傳入模板
    return render_template('index.html', token=token, version=version)