// 檔案：static/js/model_vs_model_generate.js
// 前綴：mvmg_ (model vs model generate)

function mvmg_buildPage() {
    // 建立外層容器
    const container = document.createElement("div");
    container.className = "mvmg-container";
  
    // 標題
    const h2 = document.createElement("h2");
    h2.textContent = "模型間對戰產生";
    container.appendChild(h2);
  
    // 模型 A 選單區
    const groupA = document.createElement("div");
    groupA.className = "mvmg-form-group";
    const labelA = document.createElement("label");
    labelA.className = "mvmg-label";
    labelA.setAttribute("for", "mvmg-modelA");
    labelA.textContent = "模型 A";
    const selectA = document.createElement("select");
    selectA.className = "mvmg-input";
    selectA.id = "mvmg-modelA";
    groupA.appendChild(labelA);
    groupA.appendChild(selectA);
    container.appendChild(groupA);
  
    // 模型 B 選單區
    const groupB = document.createElement("div");
    groupB.className = "mvmg-form-group";
    const labelB = document.createElement("label");
    labelB.className = "mvmg-label";
    labelB.setAttribute("for", "mvmg-modelB");
    labelB.textContent = "模型 B";
    const selectB = document.createElement("select");
    selectB.className = "mvmg-input";
    selectB.id = "mvmg-modelB";
    groupB.appendChild(labelB);
    groupB.appendChild(selectB);
    container.appendChild(groupB);
  
    // 產生場數輸入區
    const groupBattles = document.createElement("div");
    groupBattles.className = "mvmg-form-group";
    const labelBattles = document.createElement("label");
    labelBattles.className = "mvmg-label";
    labelBattles.setAttribute("for", "mvmg-num-battles");
    labelBattles.textContent = "產生場數";
    const inputBattles = document.createElement("input");
    inputBattles.className = "mvmg-input";
    inputBattles.type = "number";
    inputBattles.id = "mvmg-num-battles";
    inputBattles.value = "10";
    inputBattles.min = "1";
    groupBattles.appendChild(labelBattles);
    groupBattles.appendChild(inputBattles);
    container.appendChild(groupBattles);
  
    // 開始按鈕
    const startBtn = document.createElement("button");
    startBtn.id = "mvmg-start-btn";
    startBtn.className = "mvmg-btn";
    startBtn.textContent = "開始產生對戰資料";
    container.appendChild(startBtn);
  
    // 進度條區塊 (參考 cross_battle_generate_pc.js 的 UI)
    const progressArea = document.createElement("div");
    progressArea.classList.add("cv-generate-progress-area");
    progressArea.style.display = "none"; // 初始隱藏
  
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
  
    container.appendChild(progressArea);
  
    return container;
  }
  
  function mvmg_init() {
    // 將建立好的頁面加入 #content-area
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = "";
    contentArea.appendChild(mvmg_buildPage());
  
    // 取得模型列表，並填入下拉選單（後端從 data/saved_models 讀取）
    fetch("/api/list_models")
      .then((res) => res.json())
      .then((models) => {
        const selectA = document.getElementById("mvmg-modelA");
        const selectB = document.getElementById("mvmg-modelB");
  
        if (models.length === 0) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "無可用模型";
          selectA.appendChild(option);
          selectB.appendChild(option.cloneNode(true));
        } else {
          models.forEach((modelObj) => {
            const folderName = modelObj.folder_name;
            const optionA = document.createElement("option");
            optionA.value = folderName;
            optionA.textContent = folderName;
            selectA.appendChild(optionA);
  
            const optionB = document.createElement("option");
            optionB.value = folderName;
            optionB.textContent = folderName;
            selectB.appendChild(optionB);
          });
        }
      })
      .catch((err) => {
        console.error("取得模型列表失敗:", err);
      });
  
    // 綁定開始按鈕事件
    const startBtn = document.getElementById("mvmg-start-btn");
    // 進度區塊內的元素（注意：這邊使用 cross_battle_generate_pc 的 class 名稱）
    const progressArea = document.querySelector(".cv-generate-progress-area");
    const spinner = progressArea.querySelector(".cv-generate-spinner");
    const progressText = progressArea.querySelector(".cv-generate-progress-text");
    const progressBar = progressArea.querySelector(".cv-generate-progress-bar");
  
    startBtn.addEventListener("click", () => {
      const modelA = document.getElementById("mvmg-modelA").value;
      const modelB = document.getElementById("mvmg-modelB").value;
      const numBattles = document.getElementById("mvmg-num-battles").value;
  
      if (!modelA || !modelB) {
        alert("請選擇模型A與模型B");
        return;
      }
      // 開始產生時不再顯示對戰結果區塊，僅顯示進度更新
      // 禁用按鈕並加入灰色樣式
      startBtn.disabled = true;
      startBtn.classList.add("btn-disabled");
  
      // 顯示進度區塊
      progressArea.style.display = "block";
      spinner.style.display = "inline-block";
      progressText.textContent = "環境建置中，請稍候...";
      progressBar.style.width = "0%";
  
      // 組成 SSE URL (請確認後端 API 路由正確)
      const sseUrl = `/api/version_test_generate_model_vs_model?modelA=${encodeURIComponent(
        modelA
      )}&modelB=${encodeURIComponent(modelB)}&num_battles=${encodeURIComponent(
        numBattles
      )}`;
      const evtSource = new EventSource(sseUrl);
  
      evtSource.onmessage = function (e) {
        try {
          const info = JSON.parse(e.data);
          if (info.type === "progress") {
            progressBar.style.width = info.progress + "%";
            progressText.textContent = info.message || "環境建置中...";
          } else if (info.type === "final_result") {
            progressBar.style.width = "100%";
            progressText.textContent = info.message || "環境建置完成！";
            spinner.style.display = "none";
            // 不顯示最終對戰結果
            // 重新啟用按鈕並移除灰色樣式
            startBtn.disabled = false;
            startBtn.classList.remove("btn-disabled");
            evtSource.close();
          } else if (info.type === "error") {
            progressText.textContent = info.message || "產生過程發生錯誤";
            spinner.style.display = "none";
            // 重新啟用按鈕並移除灰色樣式
            startBtn.disabled = false;
            startBtn.classList.remove("btn-disabled");
            evtSource.close();
          }
        } catch (err) {
          console.error("解析 SSE 資料失敗:", err);
        }
      };
  
      evtSource.onerror = function (err) {
        console.error("SSE 錯誤:", err);
        progressText.textContent = "產生過程中斷或發生錯誤";
        spinner.style.display = "none";
        // 重新啟用按鈕並移除灰色樣式
        startBtn.disabled = false;
        startBtn.classList.remove("btn-disabled");
        evtSource.close();
      };
    });
  }
  
  // 將 mvmg_init 暴露至全域（方便其他頁面或選單呼叫）
  window.mvmg_init = mvmg_init;
  