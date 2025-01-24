// /static/js/train.js
document.addEventListener("DOMContentLoaded", function() {

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
  menuTrain.addEventListener("click", function() {
    contentArea.innerHTML = `
      <div id="train-page">
        <h1>多智能體訓練</h1>
        
        <!-- 參數解說區域 -->
        <div class="params-explanation">
          <h2>超參數解說</h2>
          <p>
            <b>Learning Rate (learning_rate)</b>：控制梯度下降時的步伐大小，<br>
            一般建議介於 <strong>1e-5 ~ 1e-2</strong>，<br>
            若值過大容易造成訓練不穩定，值過小則可能導致收斂速度過慢。
          </p>
          <p>
            <b>Batch Size (train_batch_size)</b>：每次更新所使用的樣本數，<br>
            值愈大，梯度估計越穩定，但單次 iteration 所需時間也越久。<br>
            建議範圍根據具體應用調整，例如 <strong>1000 ~ 10000</strong>。
          </p>
          <!-- 可根據需要添加更多超參數的解說 -->
        </div>

        <div class="form-section">
          <label for="modelNameInput">模型名稱：</label>
          <input type="text" id="modelNameInput" placeholder="my_multiagent_ai_{timestamp}" />

          <label for="iterationInput">Iteration 次數：</label>
          <input type="number" id="iterationInput" placeholder="5" min="1" />

          <label for="lrInput">Learning Rate：</label>
          <input type="number" step="0.0001" id="lrInput" placeholder="0.0001" min="0" />

          <label for="batchInput">Batch Size：</label>
          <input type="number" id="batchInput" placeholder="4000" min="1" />

          <button id="startTrainBtn">開始訓練</button>
          <button id="stopTrainBtn" disabled>終止訓練</button>
        </div>
    
        <!-- 顯示訓練初始化中 的載入效果 -->
        <div id="initializing-status" style="display:none;">
          <div class="spinner"></div>
          <span>訓練初始化中，請稍候...</span>
        </div>

        <!-- 顯示初始化完成的資訊 -->
        <div id="initialized-info" style="display:none; color: green; font-weight: bold;"></div>

        <div class="progress-bar-container">
          <div class="progress-bar" id="trainProgressBar"></div>
        </div>
    
        <!-- 訓練結果區塊 -->
        <div id="iterationResults">
          <h2>訓練結果</h2>
          <div id="results-container">
            <!-- 動態新增迭代結果 -->
          </div>
        </div>
      </div>
    `;

    // 綁定開始訓練按鈕事件
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");
    startTrainBtn.addEventListener("click", startTraining);
    stopTrainBtn.addEventListener("click", stopTraining);

    // 設定模型名稱預設值
    const modelNameInput = document.getElementById("modelNameInput");
    const now = new Date();
    const timestamp = `${now.getFullYear()}_${String(now.getMonth()+1).padStart(2, '0')}_${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}_${String(now.getMinutes()).padStart(2, '0')}_${String(now.getSeconds()).padStart(2, '0')}`;
    modelNameInput.value = `my_multiagent_ai_${timestamp}`;
  });

  function startTraining() {
    const modelName = document.getElementById("modelNameInput").value.trim() || `my_multiagent_ai_${getCurrentTimestamp()}`;
    const iteration = parseInt(document.getElementById("iterationInput").value) || 5;
    const lr = parseFloat(document.getElementById("lrInput").value) || 0.0001;
    const batchSize = parseInt(document.getElementById("batchInput").value) || 4000;

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

    const hyperparams = {
      learning_rate: lr,
      train_batch_size: batchSize
      // ... 其他可自行再擴充
    };

    // 重置畫面
    const resultsContainer = document.getElementById("results-container");
    resultsContainer.innerHTML = "";

    const progressBar = document.getElementById("trainProgressBar");
    progressBar.style.width = "0%";

    const initializingStatus = document.getElementById("initializing-status");
    const initializedInfo = document.getElementById("initialized-info");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");

    // 顯示初始化提示
    initializingStatus.style.display = "flex";  // 使用 flex 以水平排列 spinner 和文字
    initializedInfo.style.display = "none";

    // 禁用開始訓練按鈕，啟用終止訓練按鈕
    startTrainBtn.disabled = true;
    stopTrainBtn.disabled = false;

    // 建立 SSE 連線
    const hyperparamsJson = encodeURIComponent(JSON.stringify(hyperparams));
    const url = `/api/train_sse?model_name=${modelName}&iteration=${iteration}&hyperparams_json=${hyperparamsJson}`;
    currentSource = new EventSource(url);

    let currentIteration = 0;

    currentSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE 收到資料:", data);

      // 根據 data.type 來區分
      switch (data.type) {
        case "initialized":
          // 移除初始化提示
          initializingStatus.style.display = "none";
          // 顯示初始化完成資訊
          initializedInfo.style.display = "block";
          initializedInfo.textContent = data.message; // "algo = config.build() 完成"
          break;

        case "iteration":
          currentIteration = data.iteration;  // 用於更新進度

          // 更新進度條
          const progress = (currentIteration / iteration) * 100;
          progressBar.style.width = progress + "%";

          // 在前端新增一個「Iteration i」折疊區塊
          const iterationBlock = document.createElement("div");
          iterationBlock.classList.add("iteration-block");

          const iterationHeader = document.createElement("div");
          iterationHeader.classList.add("iteration-header");
          iterationHeader.textContent = `Iteration ${data.iteration} (點擊展開/收合)`;
          
          const iterationDetails = document.createElement("div");
          iterationDetails.classList.add("iteration-details");
          iterationDetails.style.display = "none";

          // 將 data 中的資訊以表格顯示
          iterationDetails.appendChild(generateInfoTable(data));

          // 點擊標題收合
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
          // 關閉 SSE
          currentSource.close();
          currentSource = null;
          // 進度條直接 100%
          progressBar.style.width = "100%";
          // 顯示初始化完成資訊（如果尚未顯示）
          if (initializedInfo.style.display === "none") {
            initializedInfo.style.display = "block";
            initializedInfo.textContent = "algo = config.build() 完成";
          }
          // 彈窗顯示完成
          showModal(data.message || "訓練完成！");
          // 重置按鈕狀態
          startTrainBtn.disabled = false;
          stopTrainBtn.disabled = true;
          break;

        case "stopped":
          // 訓練被終止
          currentSource.close();
          currentSource = null;
          showModal(data.message || "訓練已被終止。");
          // 重置按鈕狀態
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
      // 重置按鈕狀態
      startTrainBtn.disabled = false;
      stopTrainBtn.disabled = true;
    };
  }

  function stopTraining() {
    if (currentSource) {
      // 發送終止請求到後端
      fetch('/api/stop_train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
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
      // 排除我們不想重複顯示的 key
      if (["type", "iteration"].includes(key)) {
        continue;
      }
      const value = obj[key];

      const row = document.createElement("tr");
      const tdKey = document.createElement("td");
      tdKey.textContent = key;
      const tdValue = document.createElement("td");

      if (typeof value === "object" && value !== null) {
        // 若是物件或陣列 => 遞迴生成表格
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

  // 工具函式：遞迴生成嵌套表格
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

  // 工具函式：取得當前時間的字串格式
  function getCurrentTimestamp() {
    const now = new Date();
    return `${now.getFullYear()}_${String(now.getMonth()+1).padStart(2, '0')}_${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}_${String(now.getMinutes()).padStart(2, '0')}_${String(now.getSeconds()).padStart(2, '0')}`;
  }

});
