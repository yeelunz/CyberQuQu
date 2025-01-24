from flask import Blueprint, request, jsonify, render_template , Response
from utils.main import (

    version_test_random_vs_random,
    high_level_test_ai_vs_ai,
    compute_ai_elo,
    computer_vs_computer,
    ai_vs_computer,
    ai_vs_ai,
    show_profession_info,

    get_professions_data
)
from utils.skills import build_skill_manager
from utils.professions import build_professions
from utils.train_methods import multi_agent_cross_train , stop_training_flag

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



