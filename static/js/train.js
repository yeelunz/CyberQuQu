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
            <b>Learning Rate (learning_rate)</b>：控制模型更新時每次梯度下降的步伐大小。較大的值可以加速學習，但可能導致震盪或不穩定；較小的值則使收斂更平穩，但學習速度可能變慢。建議設定值介於 <strong>1e-5 ~ 1e-2</strong> 之間，並根據模型複雜度與資料量進行微調。
          </p>
          <p>
            <b>Batch Size (train_batch_size)</b>：每次模型更新所使用的樣本數。較大的 batch size 可使梯度計算更穩定，但也會增加計算資源的消耗。通常建議值在 <strong>1000 ~ 10000</strong> 之間，可根據硬體資源和任務特性調整。
          </p>
          <p>
            <b>Entropy Coefficient (entropy_coeff)</b>：用於鼓勵模型探索新的策略，增加決策過程中的隨機性。較高的值有助於防止過早收斂，但過高可能導致訓練不穩定。建議值通常在 <strong>0.001 ~ 0.1</strong> 之間。
          </p>
          <p>
            <b>Entropy Coefficient Schedule</b>：設定訓練過程中 entropy_coeff 的動態調整。請依序填入兩組數值，例如：<strong>[0, 0.01]</strong>（初期探索階段）與 <strong>[1000000, 0.0]</strong>（後期收斂階段），以平衡探索與利用。
          </p>
          <p>
            <b>Max Seq Len (max_seq_len)</b>：指定模型中 LSTM 的序列長度，也就是一次輸入的步數。這個值通常與資料中的時間序列長度或上下文窗口大小相匹配，預設為 <strong>10</strong>。根據應用場景，可適當調整此參數。
          </p>
          <p>
            <b>FC Net Hiddens (fcnet_hiddens)</b>：定義全連接層中每層的隱藏單元數量。輸入格式為用逗號分隔的數值，例如預設值 <strong>256,256</strong> 表示兩層，每層各有 256 個神經元。此參數影響模型容量與計算成本，請根據需求調整。最後一層之尺寸相當於lstm_cell_size。
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
          
          <!-- 新增 FC Net Hiddens 輸入 (預設改為 256,256) -->
          <label for="fcnetHiddensInput">FC Net Hiddens：</label>
          <input type="text" id="fcnetHiddensInput" placeholder="256,256" />

          <!-- 新增 max_seq_len 輸入 -->
          <label for="maxSeqLenInput">Max Seq Len：</label>
          <input type="number" id="maxSeqLenInput" placeholder="10" min="1" />

          <button id="startTrainBtn">開始訓練</button>
          <button id="stopTrainBtn" disabled>終止訓練</button>
        </div>
        
        <!-- 訓練初始化中狀態 -->
        <div id="initializing-status" style="display:none; align-items: center;">
          <span style="color: black;">訓練初始化中，請稍候...</span>
          <div class="spinner" style="margin-left:10px;"></div>
        </div>

        <!-- 初始化與訓練中資訊 (訓練中提示文字預設為黑色) -->
        <div id="initialized-info" style="display:none; color: black; font-weight: bold;"></div>
        
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
    
    // 取得 max_seq_len 的數值，預設為 10
    const maxSeqLen = parseInt(document.getElementById("maxSeqLenInput").value) || 10;

    // 取得 Entropy Schedule 四個欄位的數值
    const schedule1TimestepRaw = document.getElementById("entropySchedule1Timestep").value;
    const schedule1ValueRaw = document.getElementById("entropySchedule1Value").value;
    const schedule2TimestepRaw = document.getElementById("entropySchedule2Timestep").value;
    const schedule2ValueRaw = document.getElementById("entropySchedule2Value").value;

    const schedule1Timestep = schedule1TimestepRaw !== "" ? parseInt(schedule1TimestepRaw) : 0;
    const schedule1Value = schedule1ValueRaw !== "" ? parseFloat(schedule1ValueRaw) : 0.01;
    const schedule2Timestep = schedule2TimestepRaw !== "" ? parseInt(schedule2TimestepRaw) : 1000000;
    const schedule2Value = schedule2ValueRaw !== "" ? parseFloat(schedule2ValueRaw) : 0.0;

    const entropySchedule = [
      [schedule1Timestep, schedule1Value],
      [schedule2Timestep, schedule2Value]
    ];

    // 取得 FC Net Hiddens 的輸入值，並解析為整數陣列（以逗號分隔），預設改為 [256,256]
    const fcnetHiddensRaw = document.getElementById("fcnetHiddensInput").value.trim();
    let fcnetHiddens;
    if (fcnetHiddensRaw) {
      fcnetHiddens = fcnetHiddensRaw.split(",")
        .map(item => parseInt(item))
        .filter(num => !isNaN(num));
    } else {
      fcnetHiddens = [256, 256]; // 預設值
    }

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
      fcnet_hiddens: fcnetHiddens,
      max_seq_len: maxSeqLen
    };

    const resultsContainer = document.getElementById("results-container");
    resultsContainer.innerHTML = "";

    const progressBar = document.getElementById("trainProgressBar");
    progressBar.style.width = "0%";

    const initializingStatus = document.getElementById("initializing-status");
    const initializedInfo = document.getElementById("initialized-info");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");

    // 顯示初始化狀態（提示文字預設為黑色）
    initializingStatus.style.display = "flex";
    initializedInfo.style.display = "none";

    startTrainBtn.disabled = true;
    stopTrainBtn.disabled = false;

    const hyperparamsJson = encodeURIComponent(JSON.stringify(hyperparams));
    const url = `/api/train_sse?model_name=${modelName}&iteration=${iteration}&hyperparams_json=${hyperparamsJson}`;
    currentSource = new EventSource(url);

    let currentIteration = 0;

    currentSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE 收到資料:", data);

      switch (data.type) {
        case "initialized":
          initializingStatus.style.display = "none";
          initializedInfo.style.display = "block";
          // 調整：將文字放在左側，轉圈圈圖標放在右側，並修改提示文字內容與顏色（黑色）
          initializedInfo.innerHTML =
            data.message +
            "<br><div style='display:flex; align-items:center;'><span style=\"color: black;\">模型訓練中，請稍後 ...</span><span class='spinner' style='margin-left:10px;'></span></div>";
          break;

        case "iteration":
          currentIteration = data.iteration;
          const progress = (currentIteration / iteration) * 100;
          progressBar.style.width = progress + "%";

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
          // 將訓練中提示文字更新為完成狀態（綠色）
          initializedInfo.style.color = "green";
          if (initializedInfo.style.display === "none") {
            initializedInfo.style.display = "block";
          }
          initializedInfo.innerHTML = data.message || "algo = config.build() 完成";
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

  function getCurrentTimestamp() {
    const now = new Date();
    return `${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, '0')}_${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}_${String(now.getMinutes()).padStart(2, '0')}_${String(now.getSeconds()).padStart(2, '0')}`;
  }
});
