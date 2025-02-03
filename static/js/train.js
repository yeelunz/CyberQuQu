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
    const now = new Date();
    const timestamp = `${now.getFullYear()}_${String(
      now.getMonth() + 1
    ).padStart(2, "0")}_${String(now.getDate()).padStart(2, "0")}_${String(
      now.getHours()
    ).padStart(2, "0")}_${String(now.getMinutes()).padStart(2, "0")}_${String(
      now.getSeconds()
    ).padStart(2, "0")}`;
    contentArea.innerHTML = `
<div id="train-page">
  <h1>多智能體訓練</h1>
  
  <!-- 超參數說明 -->
 <div class="params-explanation">
  <h2>超參數解說</h2>
  
  <p>
    <b>Learning Rate (learning_rate)</b>：控制模型在每次參數更新時依據梯度下降法所採取步伐的大小。學習率設定過大，可能會導致訓練過程中參數波動劇烈甚至發散；而設定過小則雖然能使更新更加穩定，但收斂速度可能變得緩慢。合理的學習率能幫助模型更有效地找到最優解。建議設定值介於 <strong>1e-5 ~ 1e-2</strong> 之間，以平衡收斂速度與穩定性。
  </p>
  
  <p>
    <b>LR Schedule</b>：透過設定一系列 [Timestep, Value] 的數值對，來動態調整學習率在不同訓練階段的取值（例如：[0, 0.0001] 與 [1000000, 0.0]）。這種調度機制通常用於在訓練初期採用較高的學習率以促進快速探索，隨著訓練進展逐漸降低學習率以精細調整參數。此為默認選項，默認值為 <strong>[0, 0.0001] , [1000000, 0.0]</strong>。
  </p>
  
  <p>
    <b>Batch Size (train_batch_size)</b>：指定在每次模型更新時所使用的樣本數，即每個迭代中的 timestep 的數量。較大的 Batch Size 能夠提供更穩定的梯度估計，但同時需要更多的記憶體資源；較小的 Batch Size 則可能使模型更新更頻繁，從而捕捉到更多的樣本變化。建議值在 <strong>1000 ~ 10000</strong> 之間，視硬體資源和具體應用需求進行調整。
  </p>
  
  <p>
    <b>Entropy Coefficient (entropy_coeff)</b>：這個參數用來鼓勵模型在策略探索過程中採取更多隨機性，避免過早陷入局部最優解。較高的 entropy coefficient 值能促使模型嘗試更多新策略，但也可能導致策略過於隨機；較低的值則會讓模型過早收斂於固定策略。建議值在 <strong>0.001 ~ 0.1</strong> 之間。
  </p>
  
  <p>
    <b>Entropy Coefficient Schedule</b>：透過依序填入兩組 [Timestep, Value] 數據對（例如：[0, 0.01] 與 [1000000, 0.0]），來動態調整探索的強度。這樣在訓練初期可以鼓勵較強的探索行為，而隨著訓練進展逐步降低探索性。此為默認選項，默認值為 <strong>[0, 0.0001] , [1000000, 0.0]</strong>。
  </p>
  
  <p>
    <b>Max Seq Len (max_seq_len)</b>：代表 LSTM 模型在處理序列數據時，每次參與計算的最大時間步數。過長的序列可能會導致梯度消失或梯度爆炸問題，而過短則可能無法捕捉序列中的長期依賴關係。預設為 <strong>10</strong>，可根據具體應用場景進行調整。
  </p>
  
  <p>
    <b>FC Net Hiddens (fcnet_hiddens)</b>：定義全連接神經網路中各隱藏層的單元數量，數值之間以逗號分隔。該參數決定了網路的容量與表達能力，對於處理複雜任務尤為重要。預設為 <strong>256,256</strong>，根據問題複雜度可適當增減層數或每層單元數量。
  </p>
  
  <p>
    <b>Gamma</b>：也稱為折扣因子，用於決定未來獎勵在當前決策中的重要性。較高的 Gamma 值表示模型更重視長期回報，而較低的值則使模型更加關注近期獎勵。預設為 <strong>0.99</strong>，在大部分強化學習問題中是一個常用且穩定的選擇。
  </p>
  
  <p>
    <b>grad_clip</b>：梯度裁剪的數值上限，用來防止梯度爆炸問題。在反向傳播過程中，當梯度值超過此設定時會被裁剪到該值。此參數可填入具體數字，或留空（代表 None）。若留空則會使用預設的裁剪方式，即 grad_clip_by 為 <em>global_norm</em>。
  </p>
  
  <p>
    <b>grad_clip_by</b>：定義梯度裁剪的方法。可選項包括 <em>value</em>（直接依據梯度數值進行裁剪）、<em>norm</em>（依據梯度範數進行裁剪）以及預設選項 <em>global_norm</em>（全局範數裁剪），以確保在不同層間梯度分佈均衡。
  </p>
  
  <p>
    <b>Lambda</b>：在廣義優勢估計（GAE）中用來平衡偏差與方差的一個參數。該值控制了未來獎勵折衷的程度，預設為 <strong>1.0</strong>，意味著對未來獎勵給予全權重，對於不同任務可以根據需求進行調整。
  </p>
  
  <p>
    <b>Minibatch Size</b>：指定在每次模型更新中使用的小批次樣本數量。這個參數影響了模型參數更新的精細程度與訓練速度，預設為 <strong>128</strong>，根據問題的複雜度及硬體條件可以做適當的調整。
  </p>
  
  <p>
    <b>Clip Param</b>：用於 PPO（近端策略優化）演算法中限制策略更新幅度的參數，以防止策略在單次更新中變化過大，從而導致訓練不穩定。預設值為 <strong>0.3</strong>。
  </p>
  
  <p>
    <b>VF Clip Param</b>：則是針對 Value Function 的更新所設定的裁剪參數，預設值為 <strong>10.0</strong>，兩者均有助於提升訓練過程的穩定性。
  </p>
</div>
  
  <!-- 表單區 -->
  <div class="form-section">
    <!-- 基本參數 -->
    <div style="flex-basis: 100%;">
      <label for="modelNameInput">模型名稱：</label>
      <input type="text" id="modelNameInput" placeholder="my_multiagent_ai_${timestamp}" />
    </div>
    <div style="flex-basis: 100%;">
      <label for="iterationInput">Iteration 次數：</label>
      <input type="number" id="iterationInput" placeholder="5" min="1" />
    </div>
    <div style="flex-basis: 100%;">
      <label for="batchInput">Batch Size：</label>
      <input type="number" id="batchInput" placeholder="4000" min="1" />
    </div>
    
    <!-- Learning Rate 與 LR Schedule -->
    <fieldset style="flex-basis: 100%; border: 1px solid #ccc; padding: 10px; border-radius: 3px;">
      <legend><b>Learning Rate 設定</b></legend>
      <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <div style="flex: 1 1 300px;">
          <label for="lrInput">Learning Rate：</label>
          <input type="number" step="0.0000001" id="lrInput" placeholder="未填入則使用 Schedule" />
        </div>
        <div style="flex: 1 1 300px;">
          <label>LR Schedule:</label>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
            <input type="number" id="lrSchedule1Timestep" placeholder="Timestep 1" />
            <input type="number" step="0.0000001" id="lrSchedule1Value" placeholder="Value 1" />
            <input type="number" id="lrSchedule2Timestep" placeholder="Timestep 2" />
            <input type="number" step="0.0000001" id="lrSchedule2Value" placeholder="Value 2" />
          </div>
        </div>
      </div>
    </fieldset>
    
    <!-- Entropy Coefficient 與其 Schedule -->
    <fieldset style="flex-basis: 100%; border: 1px solid #ccc; padding: 10px; border-radius: 3px;">
      <legend><b>Entropy Coefficient 設定</b></legend>
      <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        <div style="flex: 1 1 300px;">
          <label for="entropyInput">Entropy Coefficient：</label>
          <input type="number" step="0.0001" id="entropyInput" placeholder="未填入則使用 Schedule" />
        </div>
        <div style="flex: 1 1 300px;">
          <label>Entropy Schedule:</label>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
            <input type="number" id="entropySchedule1Timestep" placeholder="Timestep 1" />
            <input type="number" step="0.0001" id="entropySchedule1Value" placeholder="Value 1" />
            <input type="number" id="entropySchedule2Timestep" placeholder="Timestep 2" />
            <input type="number" step="0.0001" id="entropySchedule2Value" placeholder="Value 2" />
          </div>
        </div>
      </div>
    </fieldset>
    
    <!-- 其他模型參數 -->
    <div style="display: flex; flex-wrap: wrap; gap: 10px; flex-basis: 100%;">
      <div style="flex: 1 1 300px;">
        <label for="fcnetHiddensInput">FC Net Hiddens：</label>
        <input type="text" id="fcnetHiddensInput" placeholder="256,256" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="maxSeqLenInput">Max Seq Len：</label>
        <input type="number" id="maxSeqLenInput" placeholder="10" min="1" />
      </div>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; flex-basis: 100%;">
      <div style="flex: 1 1 300px;">
        <label for="gammaInput">Gamma：</label>
        <input type="number" step="0.0001" id="gammaInput" placeholder="0.99" value="0.99" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="gradClipInput">grad_clip：</label>
        <input type="number" step="0.0001" id="gradClipInput" placeholder="留空表示 None" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="gradClipBySelect">grad_clip_by：</label>
        <select id="gradClipBySelect">
          <option value="value">value</option>
          <option value="norm">norm</option>
          <option value="global_norm" selected>global_norm</option>
        </select>
      </div>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; flex-basis: 100%;">
      <div style="flex: 1 1 300px;">
        <label for="lambdaInput">Lambda：</label>
        <input type="number" step="0.0001" id="lambdaInput" placeholder="1.0" value="1.0" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="minibatchSizeInput">Minibatch Size：</label>
        <input type="number" id="minibatchSizeInput" placeholder="128" value="128" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="clipParamInput">Clip Param：</label>
        <input type="number" step="0.0001" id="clipParamInput" placeholder="0.3" value="0.3" />
      </div>
      <div style="flex: 1 1 300px;">
        <label for="vfClipParamInput">VF Clip Param：</label>
        <input type="number" step="0.0001" id="vfClipParamInput" placeholder="10.0" value="10.0" />
      </div>
    </div>
    <div style="flex-basis: 100%; text-align: center; margin-top: 20px;">
      <button id="startTrainBtn">開始訓練</button>
      <button id="stopTrainBtn" disabled>終止訓練</button>
    </div>
  </div>
  
  <!-- 訓練初始化中狀態 -->
  <div id="initializing-status" style="display:none; align-items: center;">
    <span style="color: black;">訓練初始化中，請稍候...</span>
    <div class="spinner" style="margin-left:10px;"></div>
  </div>
  
  <!-- 初始化與訓練中資訊 -->
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
    modelNameInput.value = `my_multiagent_ai_${timestamp}`;
  });

  function startTraining() {
    // 模型名稱與 iteration
    const modelName =
      document.getElementById("modelNameInput").value.trim() ||
      `my_multiagent_ai_${getCurrentTimestamp()}`;
    const iteration =
      parseInt(document.getElementById("iterationInput").value) || 5;
    const batchSize =
      parseInt(document.getElementById("batchInput").value) || 4000;

    // Learning Rate 與 LR Schedule：若 lrInput 有值則使用該值，否則使用 lr_schedule
    const lrRaw = document.getElementById("lrInput").value.trim();
    const lr = lrRaw !== "" ? parseFloat(lrRaw) : null;
    const lrSchedule1TimestepRaw = document
      .getElementById("lrSchedule1Timestep")
      .value.trim();
    const lrSchedule1ValueRaw = document
      .getElementById("lrSchedule1Value")
      .value.trim();
    const lrSchedule2TimestepRaw = document
      .getElementById("lrSchedule2Timestep")
      .value.trim();
    const lrSchedule2ValueRaw = document
      .getElementById("lrSchedule2Value")
      .value.trim();
    const lrSchedule = [
      [
        lrSchedule1TimestepRaw !== "" ? parseInt(lrSchedule1TimestepRaw) : 0,
        lrSchedule1ValueRaw !== "" ? parseFloat(lrSchedule1ValueRaw) : 0.0001,
      ],
      [
        lrSchedule2TimestepRaw !== ""
          ? parseInt(lrSchedule2TimestepRaw)
          : 1000000,
        lrSchedule2ValueRaw !== "" ? parseFloat(lrSchedule2ValueRaw) : 0.0,
      ],
    ];

    // Entropy Coefficient 與 Schedule：若 entropyInput 有值則使用該值，否則使用 entropy_schedule
    const entropyRaw = document.getElementById("entropyInput").value.trim();
    const entropy = entropyRaw !== "" ? parseFloat(entropyRaw) : null;
    const entropySchedule1TimestepRaw = document
      .getElementById("entropySchedule1Timestep")
      .value.trim();
    const entropySchedule1ValueRaw = document
      .getElementById("entropySchedule1Value")
      .value.trim();
    const entropySchedule2TimestepRaw = document
      .getElementById("entropySchedule2Timestep")
      .value.trim();
    const entropySchedule2ValueRaw = document
      .getElementById("entropySchedule2Value")
      .value.trim();
    const entropySchedule = [
      [
        entropySchedule1TimestepRaw !== ""
          ? parseInt(entropySchedule1TimestepRaw)
          : 0,
        entropySchedule1ValueRaw !== ""
          ? parseFloat(entropySchedule1ValueRaw)
          : 0.01,
      ],
      [
        entropySchedule2TimestepRaw !== ""
          ? parseInt(entropySchedule2TimestepRaw)
          : 1000000,
        entropySchedule2ValueRaw !== ""
          ? parseFloat(entropySchedule2ValueRaw)
          : 0.0,
      ],
    ];

    // 其他參數
    const maxSeqLen =
      parseInt(document.getElementById("maxSeqLenInput").value) || 10;
    const fcnetHiddensRaw = document
      .getElementById("fcnetHiddensInput")
      .value.trim();
    let fcnetHiddens;
    if (fcnetHiddensRaw) {
      fcnetHiddens = fcnetHiddensRaw
        .split(",")
        .map((item) => parseInt(item))
        .filter((num) => !isNaN(num));
    } else {
      fcnetHiddens = [256, 256];
    }
    const gammaRaw = document.getElementById("gammaInput").value.trim();
    const gamma = gammaRaw !== "" ? parseFloat(gammaRaw) : 0.99;
    const gradClipRaw = document.getElementById("gradClipInput").value.trim();
    const gradClip = gradClipRaw !== "" ? parseFloat(gradClipRaw) : null;
    let gradClipBy = "global_norm";
    if (gradClip !== null) {
      gradClipBy = document.getElementById("gradClipBySelect").value;
    }
    const lambdaRaw = document.getElementById("lambdaInput").value.trim();
    const lambdaVal = lambdaRaw !== "" ? parseFloat(lambdaRaw) : 1.0;
    const minibatchSizeRaw = document
      .getElementById("minibatchSizeInput")
      .value.trim();
    const minibatchSize =
      minibatchSizeRaw !== "" ? parseInt(minibatchSizeRaw) : 128;
    const clipParamRaw = document.getElementById("clipParamInput").value.trim();
    const clipParam = clipParamRaw !== "" ? parseFloat(clipParamRaw) : 0.3;
    const vfClipParamRaw = document
      .getElementById("vfClipParamInput")
      .value.trim();
    const vfClipParam =
      vfClipParamRaw !== "" ? parseFloat(vfClipParamRaw) : 10.0;

    // 輸入值檢查
    if (iteration < 1) {
      alert("Iteration 次數必須大於 0");
      return;
    }
    if (lr !== null && lr <= 0) {
      alert("Learning Rate 必須大於 0");
      return;
    }
    if (batchSize < 1) {
      alert("Batch Size 必須大於 0");
      return;
    }
    if (entropy !== null && entropy < 0) {
      alert("Entropy Coefficient 不能為負數");
      return;
    }

    const hyperparams = {
      learning_rate: lr,
      lr_schedule: lr !== null ? null : lrSchedule,
      train_batch_size: batchSize,
      entropy_coeff: entropy,
      entropy_coeff_schedule: entropy !== null ? null : entropySchedule,
      fcnet_hiddens: fcnetHiddens,
      max_seq_len: maxSeqLen,
      gamma: gamma,
      grad_clip: gradClip,
      grad_clip_by: gradClipBy,
      lambda: lambdaVal,
      minibatch_size: minibatchSize,
      clip_param: clipParam,
      vf_clip_param: vfClipParam,
    };

    const resultsContainer = document.getElementById("results-container");
    resultsContainer.innerHTML = "";
    const progressBar = document.getElementById("trainProgressBar");
    progressBar.style.width = "0%";

    const initializingStatus = document.getElementById("initializing-status");
    const initializedInfo = document.getElementById("initialized-info");
    const startTrainBtn = document.getElementById("startTrainBtn");
    const stopTrainBtn = document.getElementById("stopTrainBtn");

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
          initializedInfo.style.color = "green";
          if (initializedInfo.style.display === "none") {
            initializedInfo.style.display = "block";
          }
          initializedInfo.innerHTML =
            data.message || "algo = config.build() 完成";
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
      fetch("/api/stop_train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "終止訓練請求" }),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("終止訓練回應:", data);
          showModal(data.message || "訓練終止請求已發送。");
        })
        .catch((error) => {
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
    return `${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, "0")}_${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}_${String(now.getMinutes()).padStart(2, "0")}_${String(now.getSeconds()).padStart(2, "0")}`;
  }
});
