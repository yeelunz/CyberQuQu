from flask import Blueprint, request, jsonify, render_template , Response,stream_with_context
from utils.main import (

    version_test_random_vs_random,
    compute_ai_elo,
    computer_vs_computer,
    ai_vs_ai,
    show_profession_info,
    get_professions_data
)
from utils.skills import build_skill_manager
from utils.professions import build_professions
from utils.train_methods import multi_agent_cross_train , stop_training_flag
from utils.train_methods import version_test_random_vs_random_sse, version_test_random_vs_random_sse_ai

from utils.data_stamp import Gdata
import json
from utils.global_var import globalVar
import os

main_routes = Blueprint('main', __name__)

# 建立全域的 manager & professions，避免每次端點都要重建
skill_mgr = build_skill_manager()
professions = build_professions()


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



@main_routes.route('/')
def index():
    """
    回傳前端的 index.html
    """
    return render_template('index.html')



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

import random

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
    print("pr1_name",pr1_name,"pr2_name",pr2_name)
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
    model_path_1 = os.path.abspath(os.path.join("data", "saved_models", model1))
    model_path_2 = os.path.abspath(os.path.join("data", "saved_models", model2))

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
        
        print("model_full_path",model_path)
        generator = version_test_random_vs_random_sse_ai(
            professions, 
            skill_mgr, 
            num_battles=num_battles, 
            model_path_1=model_path
        )
    else:
        # mode == "pc"
        generator = version_test_random_vs_random_sse(
            professions, 
            skill_mgr, 
            num_battles=num_battles
        )

    # 3) SSE 串流函式
    def sse_stream():
        for info in generator:
            # info 為 dict，例如 { "type": "progress", "progress": 50, "message": "..."}
            yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"

    # 4) 回傳 SSE
    response = Response(stream_with_context(sse_stream()), mimetype='text/event-stream')
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
    
    
@main_routes.route("/api/show_professions", methods=["GET"])
def api_show_professions():
    """
    (8) 各職業介紹
    回傳結構化的職業資料
    """
    professions_data = get_professions_data(professions, skill_mgr)
    return jsonify({"professions_info": professions_data})

