// backend/static/js/cross_battle_generate_pc.js

document.addEventListener("DOMContentLoaded", () => {
  const menuGenerateCrossBattle = document.getElementById("menu-cross-battle-generate-pc");
  if (!menuGenerateCrossBattle) return; // 如果沒有該選單，就略過

  menuGenerateCrossBattle.addEventListener("click", (e) => {
    e.preventDefault();
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = ""; // 清空

    // 建立一個容器
    const generateContainer = document.createElement("div");
    generateContainer.classList.add("cv-generate-container");

    const title = document.createElement("h2");
    title.textContent = "產生交叉對戰數據";
    generateContainer.appendChild(title);

    // 選擇對戰模式
    const modeLabel = document.createElement("label");
    modeLabel.textContent = "對戰模式：";
    const modeSelect = document.createElement("select");
    modeSelect.innerHTML = `
      <option value="pc">PC vs PC</option>
      <option value="ai">AI vs AI</option>
    `;
    generateContainer.appendChild(modeLabel);
    generateContainer.appendChild(modeSelect);

    // 模型選單(只有 AI vs AI 需要顯示)
    const modelLabel = document.createElement("label");
    modelLabel.textContent = "選擇模型：";
    const modelSelect = document.createElement("select");
    // 先給個 placeholder
    modelSelect.innerHTML = `<option value="">(載入中...)</option>`;

    modelLabel.style.display = "none";
    modelSelect.style.display = "none";
    generateContainer.appendChild(modelLabel);
    generateContainer.appendChild(modelSelect);

    // 場次
    const battleLabel = document.createElement("label");
    battleLabel.textContent = "對戰場次(num_battles)：";
    const battleInput = document.createElement("input");
    battleInput.type = "number";
    battleInput.value = "10";
    battleInput.min = "1";
    generateContainer.appendChild(battleLabel);
    generateContainer.appendChild(battleInput);

    // 按鈕
    const generateBtn = document.createElement("button");
    generateBtn.textContent = "開始產生";
    generateBtn.id = "generate-btn"; // 給按鈕一個ID以便於CSS選取
    generateContainer.appendChild(generateBtn);

    // 進度條容器
    const progressArea = document.createElement("div");
    progressArea.classList.add("cv-generate-progress-area");
    progressArea.style.display = "none";

    const spinner = document.createElement("div");
    spinner.classList.add("cv-generate-spinner");

    const progressText = document.createElement("div");
    progressText.classList.add("cv-generate-progress-text");
    progressText.textContent = "尚未開始";

    const progressBarContainer = document.createElement("div");
    progressBarContainer.classList.add("cv-generate-progress-bar-container");
    const progressBar = document.createElement("div");
    progressBar.classList.add("cv-generate-progress-bar");
    progressBar.style.width = "0%";
    progressBarContainer.appendChild(progressBar);

    progressArea.appendChild(spinner);
    progressArea.appendChild(progressText);
    progressArea.appendChild(progressBarContainer);
    generateContainer.appendChild(progressArea);

    contentArea.appendChild(generateContainer);

    // --- 監聽模式切換，AI vs AI 時才顯示「選擇模型」 ---
    modeSelect.addEventListener("change", () => {
      if (modeSelect.value === "ai") {
        modelLabel.style.display = "inline-block";
        modelSelect.style.display = "inline-block";
        fetchModels(); // 立即載入模型
      } else {
        modelLabel.style.display = "none";
        modelSelect.style.display = "none";
      }
    });

    // --- 從後端取得所有模型檔案，填入 modelSelect ---
    function fetchModels() {
      modelSelect.innerHTML = `<option value="">(載入中...)</option>`;
      fetch("/api/list_saved_models_simple")
        .then((res) => res.json())
        .then((data) => {
          modelSelect.innerHTML = ""; // 先清空

          if (!data.models || data.models.length === 0) {
            // 找不到任何模型
            modelSelect.innerHTML = `<option value="">找不到已訓練的模型</option>`;
          } else {
            // 填入每個模型名稱
            data.models.forEach((modelName) => {
              const opt = document.createElement("option");
              opt.value = modelName;     // 傳給後端的 model_path
              opt.textContent = modelName;
              modelSelect.appendChild(opt);
            });
          }
        })
        .catch((err) => {
          console.error("取得模型清單失敗:", err);
          modelSelect.innerHTML = `<option value="">模型清單載入失敗</option>`;
        });
    }

    // 初始載入模型清單（若預設選擇 AI）
    if (modeSelect.value === "ai") {
      fetchModels();
    }

    // --- 按下「開始產生」按鈕，使用 SSE ---
    generateBtn.addEventListener("click", () => {
      // 禁用按鈕並添加灰色樣式
      generateBtn.disabled = true;
      generateBtn.classList.add("btn-disabled");

      // 顯示進度區
      progressArea.style.display = "block";
      spinner.style.display = "inline-block";
      progressText.textContent = "環境建置中，請稍候...";
      progressBar.style.width = "0%";

      const modeVal = modeSelect.value;
      const modelVal = modelSelect.value;
      const numBattlesVal = parseInt(battleInput.value, 10) || 100;

      // 如果是 AI，但又沒有可用的模型，需提示
      if (modeVal === "ai") {
        if (!modelVal || modelVal === "" || modelVal === "找不到已訓練的模型" || modelVal === "模型清單載入失敗") {
          progressText.textContent = "找不到可用的模型，無法進行 AI VS AI。";
          spinner.style.display = "none";
          // 重新啟用按鈕並移除灰色樣式
          generateBtn.disabled = false;
          generateBtn.classList.remove("btn-disabled");
          return;
        }
      }

      // 建立 SSE 連線 (GET + Query Param)
      const queryParams = new URLSearchParams({
        mode: modeVal,
        model_path: modelVal,
        num_battles: numBattlesVal
      });

      const sseUrl = `/api/version_test_generate?${queryParams.toString()}`;
      console.log("SSE URL =", sseUrl);

      const evtSource = new EventSource(sseUrl);

      // 監聽訊息
      evtSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // data => { type, progress, message, ... }

          if (data.type === "progress") {
            // 進度更新
            progressText.textContent = data.message || "產生中...";
            if (typeof data.progress === "number") {
              progressBar.style.width = data.progress.toFixed(2) + "%";
            }

          } else if (data.type === "final_result") {
            // 最終完成
            progressText.textContent = data.message || "產生完成！";
            progressBar.style.width = "100%";
            spinner.style.display = "none";

            // 可在此顯示結果或提示
            alert("交叉對戰數據產生完成！");

            // 若需要，可以在此刷新顯示交叉對戰數據的頁面

            // 重新啟用按鈕並移除灰色樣式
            generateBtn.disabled = false;
            generateBtn.classList.remove("btn-disabled");

            evtSource.close();

          } else if (data.type === "error") {
            // 若後端回傳找不到模型等錯誤
            progressText.textContent = data.message || "產生過程發生錯誤";
            spinner.style.display = "none";

            // 重新啟用按鈕並移除灰色樣式
            generateBtn.disabled = false;
            generateBtn.classList.remove("btn-disabled");

            evtSource.close();
          }

        } catch (err) {
          console.error("SSE JSON parse error:", err);
          // 重新啟用按鈕並移除灰色樣式
          generateBtn.disabled = false;
          generateBtn.classList.remove("btn-disabled");
        }
      };

      // 監聽錯誤
      evtSource.onerror = (err) => {
        console.error("SSE 連線發生錯誤", err);
        progressText.textContent = "產生過程中斷或發生錯誤 (onerror)";
        spinner.style.display = "none";

        // 重新啟用按鈕並移除灰色樣式
        generateBtn.disabled = false;
        generateBtn.classList.remove("btn-disabled");

        evtSource.close();
      };
    });
  });
});