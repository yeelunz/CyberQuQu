// static/js/manage_models.js

document.addEventListener("DOMContentLoaded", () => {
  const menuManageModels = document.getElementById("menu-manage-models");
  const contentArea = document.getElementById("content-area");

  // 當點擊「管理模型」選單時
  menuManageModels.addEventListener("click", () => {
    // 1. 請求後端，取得所有模型列表及其 metadata
    fetch("/api/list_models")
      .then((res) => res.json())
      .then((models) => {
        // 2. 生成管理模型的頁面
        renderManageModelsPage(models);
      })
      .catch((err) => {
        console.error("取得模型列表失敗:", err);
        contentArea.innerHTML = `<p style="color:red;">取得模型列表失敗！</p>`;
      });
  });

  function renderManageModelsPage(models) {
    // 建立主要容器
    contentArea.innerHTML = `
        <div id="manage-models-page">
          <h1>模型管理</h1>
          <div id="models-container"></div>
        </div>
      `;

    const modelsContainer = document.getElementById("models-container");

    if (models.length === 0) {
      modelsContainer.innerHTML = `<p>目前沒有任何已訓練的模型。</p>`;
      return;
    }

    // 逐一建立模型卡片
    models.forEach((modelObj) => {
      const folderName = modelObj.folder_name;
      const meta = modelObj.meta; // training_meta.json 內容

      // 主容器
      const modelBlock = document.createElement("div");
      modelBlock.classList.add("mm-model-block");

      // 標題(可點擊展開/收合)
      const modelHeader = document.createElement("div");
      modelHeader.classList.add("mm-model-header");
      modelHeader.textContent = folderName; // 或 meta.model_name

      const modelDetails = document.createElement("div");
      modelDetails.classList.add("mm-model-details");
      modelDetails.style.display = "none";

      // 只有當 hidden_info 為 false 才顯示完整 meta 資訊
      if (!meta.hidden_info) {
        // 顯示所有 meta 資訊（排除 elo_result）
        const detailsTable = createMetaTable(meta);
        modelDetails.appendChild(detailsTable);
      }

      // 根據是否有 ELO 結果，顯示不同的按鈕和資訊
      if (!meta.elo_result) { // 避免 `null` 或 `undefined`
        // 顯示「無 ELO 資訊」和「計算 ELO」按鈕
        const noInfoDiv = document.createElement("div");
        noInfoDiv.classList.add("mm-no-elo-info");
        noInfoDiv.textContent = "無 ELO 資訊。";

        const computeELOBtn = document.createElement("button");
        computeELOBtn.textContent = "計算 ELO";
        computeELOBtn.classList.add("mm-compute-elo-btn");

        // 綁定點擊事件
        computeELOBtn.addEventListener("click", () => {
          // 按下後 => 開始 ELO 計算
          startComputeELO(folderName, computeELOBtn, modelDetails);
        });

        modelDetails.appendChild(noInfoDiv);
        modelDetails.appendChild(computeELOBtn);
      } else {
        // 顯示 ELO 結果和「重新計算 ELO」按鈕
        const eloDiv = document.createElement("div");
        eloDiv.classList.add("mm-elo-result");

        // 生成職業詳細 ELO 卡片
        const eloDetailsHTML = generateEloDetailsHTML(meta.elo_result["詳細"]);

        // 生成總和 ELO 卡片
        const eloTotalHTML = generateEloTotalHTML(meta.elo_result);

        eloDiv.innerHTML = `
          <h3>ELO 計算結果</h3>
          <div class="mm-elo-details">
            ${eloDetailsHTML}
            ${eloTotalHTML}
          </div>
        `;
        modelDetails.appendChild(eloDiv);

        const recomputeELOBtn = document.createElement("button");
        recomputeELOBtn.textContent = "重新計算 ELO";
        recomputeELOBtn.classList.add("mm-compute-elo-btn");

        // 綁定點擊事件
        recomputeELOBtn.addEventListener("click", () => {
          // 按下後 => 開始 ELO 計算
          startComputeELO(folderName, recomputeELOBtn, modelDetails);
        });

        modelDetails.appendChild(recomputeELOBtn);
      }

      // 點擊標題 -> 展開/收合
      modelHeader.addEventListener("click", () => {
        if (modelDetails.style.display === "none") {
          modelDetails.style.display = "block";
          modelHeader.classList.add("active");
        } else {
          modelDetails.style.display = "none";
          modelHeader.classList.remove("active");
        }
      });

      // 組合
      modelBlock.appendChild(modelHeader);
      modelBlock.appendChild(modelDetails);
      modelsContainer.appendChild(modelBlock);
    });
  }

  /**
   * 建立 meta 的表格（排除 elo_result 與 hidden_info）
   */
  function createMetaTable(meta) {
    const table = document.createElement("table");
    table.classList.add("mm-meta-table");
  
    // 若 hyperparams 存在且包含 mask_model，額外新增一列顯示 mask_model
    if (
      meta.hyperparams &&
      typeof meta.hyperparams === "object" &&
      meta.hyperparams.hasOwnProperty("mask_model")
    ) {
      const row = document.createElement("tr");
      const tdKey = document.createElement("td");
      tdKey.textContent = "mask_model";
      const tdVal = document.createElement("td");
      tdVal.textContent = meta.hyperparams.mask_model;
      row.appendChild(tdKey);
      row.appendChild(tdVal);
      table.appendChild(row);
    }
  
    // 依序建立其他欄位（排除不想顯示的 key，如 elo_result 與 hidden_info）
    for (const key in meta) {
      if (["elo_result", "hidden_info"].includes(key)) {
        continue;
      }
      const row = document.createElement("tr");
      const tdKey = document.createElement("td");
      tdKey.textContent = key;
      const tdVal = document.createElement("td");
  
      // 如果是物件，使用 JSON 格式呈現
      if (typeof meta[key] === "object" && meta[key] !== null) {
        tdVal.textContent = JSON.stringify(meta[key], null, 2);
      } else {
        tdVal.textContent = meta[key];
      }
      row.appendChild(tdKey);
      row.appendChild(tdVal);
      table.appendChild(row);
    }
    return table;
  }

  /**
   * 開始計算 ELO 的流程
   * - 顯示一個進度條和轉圈圈
   * - 呼叫後端的 `/api/compute_elo_sse` 端點 (使用 SSE)
   * - 計算完後，更新介面(顯示 ELO 結果)
   */
  function startComputeELO(folderName, btn, detailsDiv) {
    // 1) 先把按鈕 disable
    btn.disabled = true;

    // 2) 在下方顯示進度指示
    const progressContainer = document.createElement("div");
    progressContainer.classList.add("mm-elo-progress");

    const progressLine = document.createElement("div");
    progressLine.classList.add("mm-progress-line");

    const spinner = document.createElement("div");
    spinner.classList.add("mm-spinner");

    const progressText = document.createElement("span");
    progressText.classList.add("mm-progress-text");
    progressText.textContent = "環境初始化中，請稍後...";

    progressLine.appendChild(spinner);
    progressLine.appendChild(progressText);

    const progressBarContainer = document.createElement("div");
    progressBarContainer.classList.add("mm-progress-bar-container");

    const progressBar = document.createElement("div");
    progressBar.classList.add("mm-progress-bar");

    progressBarContainer.appendChild(progressBar);

    progressContainer.appendChild(progressLine);
    progressContainer.appendChild(progressBarContainer);

    detailsDiv.appendChild(progressContainer);

    // 3) 建立 SSE 連線
    let completedNormally = false; // 用來判斷是否真的「正常完成」
    const source = new EventSource(
      `/api/compute_elo_sse?folder_name=${encodeURIComponent(folderName)}`
    );

    // 4) 監聽 SSE 事件
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("SSE 收到資料:", data);

        switch (data.type) {
          case "progress":
            progressBar.style.width = `${data.progress}%`;
            progressText.textContent = data.message;
            break;

          case "done":
            // 標記為正常結束
            completedNormally = true;

            // 關閉 SSE
            source.close();

            // 移除進度指示
            progressContainer.remove();

            // 顯示新的 ELO 結果
            if (data.new_elo_result) {
              const eloDiv = document.createElement("div");
              eloDiv.classList.add("mm-elo-result");

              // 生成職業詳細 ELO 卡片
              const eloDetailsHTML = generateEloDetailsHTML(
                data.new_elo_result["詳細"]
              );

              // 生成總和 ELO 卡片
              const eloTotalHTML = generateEloTotalHTML(data.new_elo_result);

              eloDiv.innerHTML = `
                <h3>ELO 計算結果</h3>
                <div class="mm-elo-details">
                  ${eloDetailsHTML}
                  ${eloTotalHTML}
                </div>
              `;
              detailsDiv.appendChild(eloDiv);
            }

            // 啟用按鈕
            btn.disabled = false;
            break;

          case "stopped":
            // 中斷
            source.close();
            progressContainer.remove();
            const stoppedMsg = document.createElement("div");
            stoppedMsg.style.color = "red";
            stoppedMsg.textContent =
              data.message || "ELO 計算已被終止。";
            detailsDiv.appendChild(stoppedMsg);
            btn.disabled = false;
            break;

          case "error":
            // 真的後端報錯
            source.close();
            progressContainer.remove();
            const errorMsg = document.createElement("div");
            errorMsg.style.color = "red";
            errorMsg.textContent =
              data.message || "ELO 計算過程中發生錯誤。";
            detailsDiv.appendChild(errorMsg);
            btn.disabled = false;
            break;

          default:
            console.warn("未知的 SSE 資料類型:", data);
        }
      } catch (err) {
        console.error("解析 SSE 資料時發生錯誤:", err);
        progressContainer.remove();
        const errorMsg = document.createElement("div");
        errorMsg.style.color = "red";
        errorMsg.textContent = "接收到無效的 ELO 計算資料。";
        detailsDiv.appendChild(errorMsg);
        btn.disabled = false;
      }
    };

    // 如果 SSE 被動或意外關閉，可能會觸發 onerror
    source.onerror = (err) => {
      console.error("SSE 發生錯誤或連線中斷:", err);
      source.close();

      // 如果不是正常結束，才顯示錯誤
      if (!completedNormally) {
        progressContainer.remove();
        const errorMsg = document.createElement("div");
        errorMsg.style.color = "red";
        errorMsg.textContent =
          "ELO 計算過程中連線中斷或發生錯誤。";
        detailsDiv.appendChild(errorMsg);
        btn.disabled = false;
      }
    };
  }

  /**
   * 生成 ELO 詳細結果的 HTML，包含職業圖像
   */
  function generateEloDetailsHTML(details) {
    let html = "";

    for (const prof in details) {
      const eloData = details[prof];
      const imagePath = `/static/images/${encodeURIComponent(prof)}.png`; // 確保職業名稱編碼正確

      html += `
          <div class="mm-elo-profession">
            <img src="${imagePath}" alt="${prof}" class="mm-profession-image" />
            <div class="mm-profession-info">
              <h4>${prof}</h4>
              <p><strong>先攻方 ELO:</strong> ${Math.round(
                eloData["先攻方 ELO"]
              )}</p>
              <p><strong>後攻方 ELO:</strong> ${Math.round(
                eloData["後攻方 ELO"]
              )}</p>
              <p><strong>總和 ELO:</strong> ${Math.round(
                eloData["總和 ELO"]
              )}</p>
            </div>
          </div>
        `;
    }

    return html;
  }

  /**
   * 生成總和 ELO 的 HTML，不包含圖片
   */
  function generateEloTotalHTML(totalElo) {
    // totalElo 包含 "總和先攻方ELO", "總和後攻方ELO", "總和ELO"
    return `
      <div class="mm-elo-total">
        <div class="mm-elo-total-title">總和</div>
        <p><strong>總和先攻方ELO:</strong> ${Math.round(
          totalElo["總和先攻方ELO"]
        )}</p>
        <p><strong>總和後攻方ELO:</strong> ${Math.round(
          totalElo["總和後攻方ELO"]
        )}</p>
        <p><strong>總和ELO:</strong> ${Math.round(totalElo["總和ELO"])}</p>
      </div>
    `;
  }
});
