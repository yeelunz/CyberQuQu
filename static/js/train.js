// /static/js/train.js
document.addEventListener("DOMContentLoaded", function () {
  const menuTrain = document.getElementById("menu-train");
  const contentArea = document.getElementById("content-area");

  // 自訂打開/關閉 Modal 的函式
  function showModal(message) {
    const modal = document.getElementById("train-complete-modal");
    const modalMsg = document.getElementById("modal-message");
    modalMsg.textContent = message;
    modal.style.display = "block";
  }

  function closeModal() {
    const modal = document.getElementById("train-complete-modal");
    modal.style.display = "none";
  }

  // Modal 的關閉按鈕
  const modalCloseBtn = document.getElementById("modal-close");
  modalCloseBtn.addEventListener("click", closeModal);

  // 當前的 SSE 連線
  let currentSource = null;

  // 點擊左側選單「訓練模型」時的處理
  menuTrain.addEventListener("click", function () {
    contentArea.innerHTML = `
      <div id="train-page">
        <h1>多智能體訓練</h1>
        
        <!-- 超參數說明 -->
        <div class="params-explanation">
          <h2>超參數解說</h2>
          <p>
            <b>Learning Rate (learning_rate)</b>：控制梯度下降時的步伐大小，
            建議介於 <strong>1e-5 ~ 1e-2</strong> 之間。
          </p>
          <p>
            <b>Batch Size (train_batch_size)</b>：每次更新所使用的樣本數，
            值愈大梯度估計愈穩定，但單次 iteration 所需時間也較久，
            建議根據應用情境調整（例如：1000 ~ 10000）。
          </p>
          <p>
            <b>Entropy Coefficient (entropy_coeff)</b>：用於鼓勵策略隨機性，
            建議值通常在 <strong>0.001 ~ 0.1</strong> 範圍內。
          </p>
          <p>
            <b>Entropy Coefficient Schedule</b>：設定訓練過程中 entropy_coeff 的變化，
            請依序填入兩組數值。
          </p>
        </div>
        
        <!-- 表單區 -->
        <div class="form-section">
          <label for="modelNameInput">模型名稱：</label>
          <input type="text" id="modelNameInput" placeholder="my_multiagent_ai_{timestamp}" />

          <label for="iterationInput">Iteration 次數：</label>
          <input type="number" id="iterationInput" placeholder="5" min="1" />

          <label for="lrInput">Learning Rate：</label>
          <input type="number" step="0.0001" id="lrInput" placeholder="0.0001" min="0" />

          <label for="batchInput">Batch Size：</label>
          <input type="number" id="batchInput" placeholder="4000" min="1" />

          <label for="entropyInput">Entropy Coefficient：</label>
          <input type="number" step="0.001" id="entropyInput" placeholder="0.01" min="0" />

          <!-- 將 Entropy Schedule 拆成四個欄位 -->
          <fieldset class="entropy-schedule-group" style="border:1px solid #ccc; padding:10px; border-radius:3px; flex:1 1 100%;">
            <legend style="font-weight:bold;">Entropy Coefficient Schedule</legend>
            <div style="display: flex; align-items: center; gap: 5px; margin-bottom:5px;">
              <label for="entropySchedule1Timestep" style="width: 150px;">Timestep 1:</label>
              <input type="number" id="entropySchedule1Timestep" placeholder="0" style="flex:1 1 200px;" />
            </div>
            <div style="display: flex; align-items: center; gap: 5px; margin-bottom:5px;">
              <label for="entropySchedule1Value" style="width: 150px;">Value 1:</label>
              <input type="number" step="0.001" id="entropySchedule1Value" placeholder="0.01" style="flex:1 1 200px;" />
            </div>
            <div style="display: flex; align-items: center; gap: 5px; margin-bottom:5px;">
              <label for="entropySchedule2Timestep" style="width: 150px;">Timestep 2:</label>
              <input type="number" id="entropySchedule2Timestep" placeholder="1000000" style="flex:1 1 200px;" />
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
              <label for="entropySchedule2Value" style="width: 150px;">Value 2:</label>
              <input type="number" step="0.001" id="entropySchedule2Value" placeholder="0.0" style="flex:1 1 200px;" />
            </div>
          </fieldset>
          
          <!-- 新增 FC Net Hiddens 輸入 -->
          <label for="fcnetHiddensInput">FC Net Hiddens：</label>
          <input type="text" id="fcnetHiddensInput" placeholder="256,256,256" />

          <button id="startTrainBtn">開始訓練</button>
          <button id="stopTrainBtn" disabled>終止訓練</button>
        </div>
        
        <!-- 訓練初始化中狀態 -->
        <div id="initializing-status" style="display:none;">
          <div class="spinner"></div>
          <span>訓練初始化中，請稍候...</span>
        </div>

        <!-- 初始化與訓練中資訊 -->
        <div id="initialized-info" style="display:none; color: green; font-weight: bold;"></div>
        
        <!-- 進度條 -->
        <div class="progress-bar-container">
          <div class="progress-bar" id="trainProgressBar"></div>
        </div>
        
        <!-- 訓練結果 -->
        <div id="iterationResults">
          <h2>訓練結果</h2>
          <div id="results-container"></div>
        </div>
      </div>
    `;

    // 綁定開始與終止訓練按鈕事件
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");
    startTrainBtn.addEventListener("click", startTraining);
    stopTrainBtn.addEventListener("click", stopTraining);

    // 設定預設的模型名稱（依據目前時間戳記）
    const modelNameInput = document.getElementById("modelNameInput");
    const now = new Date();
    const timestamp = `${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, '0')}_${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}_${String(now.getMinutes()).padStart(2, '0')}_${String(now.getSeconds()).padStart(2, '0')}`;
    modelNameInput.value = `my_multiagent_ai_${timestamp}`;
  });

  function startTraining() {
    const modelName =
      document.getElementById("modelNameInput").value.trim() ||
      `my_multiagent_ai_${getCurrentTimestamp()}`;
    const iteration = parseInt(document.getElementById("iterationInput").value) || 5;
    const lr = parseFloat(document.getElementById("lrInput").value) || 0.0001;
    const batchSize = parseInt(document.getElementById("batchInput").value) || 4000;
    const entropy = parseFloat(document.getElementById("entropyInput").value) || 0.01;

    // 從四個欄位取得 Entropy Schedule 的數值，若使用者未填則採用預設值
    const schedule1TimestepRaw = document.getElementById("entropySchedule1Timestep").value;
    const schedule1ValueRaw = document.getElementById("entropySchedule1Value").value;
    const schedule2TimestepRaw = document.getElementById("entropySchedule2Timestep").value;
    const schedule2ValueRaw = document.getElementById("entropySchedule2Value").value;

    const schedule1Timestep = schedule1TimestepRaw !== "" ? parseInt(schedule1TimestepRaw) : 0;
    const schedule1Value = schedule1ValueRaw !== "" ? parseFloat(schedule1ValueRaw) : 0.01;
    const schedule2Timestep =
      schedule2TimestepRaw !== "" ? parseInt(schedule2TimestepRaw) : 1000000;
    const schedule2Value = schedule2ValueRaw !== "" ? parseFloat(schedule2ValueRaw) : 0.0;

    const entropySchedule = [
      [schedule1Timestep, schedule1Value],
      [schedule2Timestep, schedule2Value]
    ];

    // 取得 FC Net Hiddens 的輸入值，並解析為整數陣列（以逗號分隔）
    const fcnetHiddensRaw = document.getElementById("fcnetHiddensInput").value.trim();
    let fcnetHiddens;
    if (fcnetHiddensRaw) {
      fcnetHiddens = fcnetHiddensRaw.split(",")
        .map(item => parseInt(item))
        .filter(num => !isNaN(num));
    } else {
      fcnetHiddens = [256, 256, 256]; // 預設值
    }

    // 基本驗證
    if (iteration < 1) {
      alert("Iteration 次數必須大於 0");
      return;
    }
    if (lr <= 0) {
      alert("Learning Rate 必須大於 0");
      return;
    }
    if (batchSize < 1) {
      alert("Batch Size 必須大於 0");
      return;
    }
    if (entropy < 0) {
      alert("Entropy Coefficient 不能為負數");
      return;
    }

    const hyperparams = {
      learning_rate: lr,
      train_batch_size: batchSize,
      entropy_coeff: entropy,
      entropy_coeff_schedule: entropySchedule,
      fcnet_hiddens: fcnetHiddens
      // ... 其他可自行再擴充的超參數
    };

    // 重置訓練結果區
    const resultsContainer = document.getElementById("results-container");
    resultsContainer.innerHTML = "";

    // 重置進度條
    const progressBar = document.getElementById("trainProgressBar");
    progressBar.style.width = "0%";

    // 顯示初始化狀態
    const initializingStatus = document.getElementById("initializing-status");
    const initializedInfo = document.getElementById("initialized-info");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");

    initializingStatus.style.display = "flex";
    initializedInfo.style.display = "none";

    // 禁用開始按鈕，啟用終止按鈕
    startTrainBtn.disabled = true;
    stopTrainBtn.disabled = false;

    // 建立 SSE 連線並傳送超參數
    const hyperparamsJson = encodeURIComponent(JSON.stringify(hyperparams));
    const url = `/api/train_sse?model_name=${modelName}&iteration=${iteration}&hyperparams_json=${hyperparamsJson}`;
    currentSource = new EventSource(url);

    let currentIteration = 0;

    currentSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE 收到資料:", data);

      switch (data.type) {
        case "initialized":
          // 隱藏初始化狀態，並顯示「環境初始化完成」及「模型訓練中」狀態與轉動圈圈
          initializingStatus.style.display = "none";
          initializedInfo.style.display = "block";
          initializedInfo.innerHTML =
            data.message +
            "<br>模型訓練中 <span class=\"spinner\" style=\"display:inline-block;\"></span>";
          break;

        case "iteration":
          currentIteration = data.iteration;
          const progress = (currentIteration / iteration) * 100;
          progressBar.style.width = progress + "%";

          // 新增每次迭代的結果折疊區塊
          const iterationBlock = document.createElement("div");
          iterationBlock.classList.add("iteration-block");

          const iterationHeader = document.createElement("div");
          iterationHeader.classList.add("iteration-header");
          iterationHeader.textContent = `Iteration ${data.iteration} (點擊展開/收合)`;
          
          const iterationDetails = document.createElement("div");
          iterationDetails.classList.add("iteration-details");
          iterationDetails.style.display = "none";
          iterationDetails.appendChild(generateInfoTable(data));

          iterationHeader.addEventListener("click", () => {
            if (iterationDetails.style.display === "none") {
              iterationDetails.style.display = "block";
              iterationHeader.classList.add("active");
            } else {
              iterationDetails.style.display = "none";
              iterationHeader.classList.remove("active");
            }
          });

          iterationBlock.appendChild(iterationHeader);
          iterationBlock.appendChild(iterationDetails);
          resultsContainer.appendChild(iterationBlock);
          break;

        case "done":
          currentSource.close();
          currentSource = null;
          progressBar.style.width = "100%";
          if (initializedInfo.style.display === "none") {
            initializedInfo.style.display = "block";
            initializedInfo.textContent = "algo = config.build() 完成";
          }
          showModal(data.message || "訓練完成！");
          startTrainBtn.disabled = false;
          stopTrainBtn.disabled = true;
          break;

        case "stopped":
          currentSource.close();
          currentSource = null;
          showModal(data.message || "訓練已被終止。");
          startTrainBtn.disabled = false;
          stopTrainBtn.disabled = true;
          break;

        default:
          console.warn("未知的 SSE 資料類型:", data);
      }
    };

    currentSource.onerror = (err) => {
      console.error("SSE 發生錯誤或連線中斷:", err);
      currentSource.close();
      currentSource = null;
      showModal("訓練過程中發生錯誤或連線中斷。");
      startTrainBtn.disabled = false;
      stopTrainBtn.disabled = true;
    };
  }

  function stopTraining() {
    if (currentSource) {
      fetch('/api/stop_train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: '終止訓練請求' })
      })
        .then(response => response.json())
        .then(data => {
          console.log("終止訓練回應:", data);
          showModal(data.message || "訓練終止請求已發送。");
        })
        .catch(error => {
          console.error("終止訓練時發生錯誤:", error);
          showModal("終止訓練時發生錯誤。");
        });
    }
  }

  // 工具函式：將物件轉成 HTML Table (支援嵌套物件)
  function generateInfoTable(obj) {
    const table = document.createElement("table");
    table.classList.add("info-table");

    for (const key in obj) {
      if (["type", "iteration"].includes(key)) continue;
      const value = obj[key];

      const row = document.createElement("tr");
      const tdKey = document.createElement("td");
      tdKey.textContent = key;
      const tdValue = document.createElement("td");

      if (typeof value === "object" && value !== null) {
        tdValue.appendChild(generateNestedTable(value));
      } else {
        tdValue.textContent = value;
      }
      row.appendChild(tdKey);
      row.appendChild(tdValue);
      table.appendChild(row);
    }
    return table;
  }

  // 遞迴生成嵌套表格
  function generateNestedTable(obj) {
    const nestedTable = document.createElement("table");
    nestedTable.classList.add("nested-info-table");

    for (const key in obj) {
      const value = obj[key];
      const row = document.createElement("tr");
      const tdKey = document.createElement("td");
      tdKey.textContent = key;
      const tdValue = document.createElement("td");

      if (typeof value === "object" && value !== null) {
        tdValue.appendChild(generateNestedTable(value));
      } else {
        tdValue.textContent = value;
      }
      row.appendChild(tdKey);
      row.appendChild(tdValue);
      nestedTable.appendChild(row);
    }
    return nestedTable;
  }

  // 取得目前時間戳記字串
  function getCurrentTimestamp() {
    const now = new Date();
    return `${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, '0')}_${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}_${String(now.getMinutes()).padStart(2, '0')}_${String(now.getSeconds()).padStart(2, '0')}`;
  }
});
