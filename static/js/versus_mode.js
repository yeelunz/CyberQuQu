// /static/js/versus_mode.js

/**
 * 初始化左側選單事件
 * (AI vs AI) vs (PC vs PC)
 */
function initVersusMenuEvents() {
  const menuAiVsAi = document.getElementById("menu-ai-vs-ai");
  if (menuAiVsAi) {
    menuAiVsAi.addEventListener("click", (e) => {
      e.preventDefault();
      showAiVsAiForm();
    });
  }

  const menuPcVsPc = document.getElementById("menu-pc-vs-pc");
  if (menuPcVsPc) {
    menuPcVsPc.addEventListener("click", (e) => {
      e.preventDefault();
      showPcVsPcForm();
    });
  }
}

/* ===================
     PC vs PC
  =================== */
async function showPcVsPcForm() {
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = "";

  const wrapper = document.createElement("div");
  wrapper.classList.add("versus-mode");

  // 取得職業清單 (後端回傳 "聖騎士","狂戰士" 等)
  const professions = await fetchProfessions();
  // 前端手動加一個 "Random"
  professions.push("Random");

  // (1) 左Panel
  const leftPanel = document.createElement("div");
  leftPanel.classList.add("form-panel");
  leftPanel.innerHTML = `
      <h3>左方 PC</h3>
      <label>職業：
        <select id="pc-left-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
    `;

  // (2) 右Panel
  const rightPanel = document.createElement("div");
  rightPanel.classList.add("form-panel");
  rightPanel.innerHTML = `
      <h3>右方 PC</h3>
      <label>職業：
        <select id="pc-right-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
    `;

  // 置於一個 row
  const row = document.createElement("div");
  row.classList.add("versus-form-row");
  row.appendChild(leftPanel);
  row.appendChild(rightPanel);

  // 按鈕
  const buttonRow = document.createElement("div");
  buttonRow.classList.add("button-row");

  const startBtn = document.createElement("button");
  startBtn.textContent = "開始對戰";
  startBtn.addEventListener("click", () => {
    startPcVsPcBattle();
  });
  buttonRow.appendChild(startBtn);

  wrapper.appendChild(row);
  wrapper.appendChild(buttonRow);

  contentArea.appendChild(wrapper);
}
async function startPcVsPcBattle() {
  const leftProfession = document.getElementById("pc-left-profession").value;
  const rightProfession = document.getElementById("pc-right-profession").value;

  // 存下目前的對戰模式與配置 (用於「再來一場」)
  window.currentBattleType = "pc_vs_pc";
  window.currentBattleConfig = {
    leftProfession,
    rightProfession,
  };

  showLoadingSpinner();
  try {
    const query = new URLSearchParams({
      pr1: leftProfession,
      pr2: rightProfession,
    });
    const res = await fetch(`/api/computer_vs_computer?${query.toString()}`);
    if (!res.ok) {
      hideLoadingSpinner();
      const text = await res.text();
      alert("對戰發生錯誤: " + text);
      return;
    }
    const data = await res.json();
    hideLoadingSpinner();

    const battleLog = data.battle_log;
    initBattleView();
    startBattle(battleLog);
  } catch (error) {
    hideLoadingSpinner();
    alert("PC vs PC 對戰錯誤: " + error);
    console.error(error);
  }
}

/* ===================
     AI vs AI
  =================== */
async function showAiVsAiForm() {
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = "";

  const wrapper = document.createElement("div");
  wrapper.classList.add("versus-mode");

  // 取得職業
  const professions = await fetchProfessions();
  professions.push("Random");

  // 取得模型列表
  let modelList = [];
  try {
    const res = await fetch("/api/list_saved_models_simple");
    const data = await res.json();
    modelList = data.models || [];
  } catch (e) {
    console.warn("取得模型列表失敗:", e);
    modelList = [];
  }

  // 左方
  const leftPanel = document.createElement("div");
  leftPanel.classList.add("form-panel");
  leftPanel.innerHTML = `
      <h3>左方 AI</h3>
      <label>職業：
        <select id="ai-left-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
      <label>模型：
        ${
          modelList.length > 0
            ? `
            <select id="ai-left-model">
              <option value="">請選擇模型</option>
              ${modelList
                .map((m) => `<option value="${m}">${m}</option>`)
                .join("")}
            </select>
          `
            : `<span style="color:red;">(尚無已訓練模型)</span>`
        }
      </label>
    `;

  // 右方
  const rightPanel = document.createElement("div");
  rightPanel.classList.add("form-panel");
  rightPanel.innerHTML = `
      <h3>右方 AI</h3>
      <label>職業：
        <select id="ai-right-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
      <label>模型：
        ${
          modelList.length > 0
            ? `
            <select id="ai-right-model">
              <option value="">請選擇模型</option>
              ${modelList
                .map((m) => `<option value="${m}">${m}</option>`)
                .join("")}
            </select>
          `
            : `<span style="color:red;">(尚無已訓練模型)</span>`
        }
      </label>
    `;

  // row
  const row = document.createElement("div");
  row.classList.add("versus-form-row");
  row.appendChild(leftPanel);
  row.appendChild(rightPanel);

  // 按鈕
  const buttonRow = document.createElement("div");
  buttonRow.classList.add("button-row");

  const startBtn = document.createElement("button");
  startBtn.textContent = "開始對戰";
  startBtn.addEventListener("click", () => {
    startAiVsAiBattle();
  });
  buttonRow.appendChild(startBtn);

  wrapper.appendChild(row);
  wrapper.appendChild(buttonRow);

  contentArea.appendChild(wrapper);
}

async function startAiVsAiBattle() {
  const leftProfession = document.getElementById("ai-left-profession").value;
  const rightProfession = document.getElementById("ai-right-profession").value;

  const leftModelSelect = document.getElementById("ai-left-model");
  const rightModelSelect = document.getElementById("ai-right-model");

  const leftModel = leftModelSelect ? leftModelSelect.value : "";
  const rightModel = rightModelSelect ? rightModelSelect.value : "";

  // 存下目前的對戰模式與配置 (用於「再來一場」)
  window.currentBattleType = "ai_vs_ai";
  window.currentBattleConfig = {
    leftProfession,
    rightProfession,
    leftModel,
    rightModel,
  };

  showLoadingSpinner();
  try {
    const query = new URLSearchParams({
      pr1: leftProfession,
      pr2: rightProfession,
      model1: leftModel,
      model2: rightModel,
    });
    const res = await fetch(`/api/ai_vs_ai?${query.toString()}`);

    if (!res.ok) {
      hideLoadingSpinner();
      const data = await res.json().catch(() => ({}));
      const errMsg = data.error || "HTTP狀態碼：" + res.status;
      alert("AI vs AI 發生錯誤: " + errMsg);
      return;
    }

    const data = await res.json();
    hideLoadingSpinner();

    const battleLog = data.battle_log;
    initBattleView();
    startBattle(battleLog);
  } catch (error) {
    hideLoadingSpinner();
    alert("AI vs AI 對戰錯誤: " + error);
    console.error(error);
  }
}

/* 取得職業清單 */
async function fetchProfessions() {
  try {
    const res = await fetch("/api/list_professions");
    const data = await res.json();
    return data.professions || [];
  } catch (e) {
    console.warn("無法取得職業清單", e);
    return [];
  }
}

/* Loading Spinner */
function showLoadingSpinner() {
  const spinner = document.getElementById("model-loading-spinner");
  if (spinner) {
    spinner.style.display = "flex";
  }
}
function hideLoadingSpinner() {
  const spinner = document.getElementById("model-loading-spinner");
  if (spinner) {
    spinner.style.display = "none";
  }
}

window.initVersusMenuEvents = initVersusMenuEvents;
document.addEventListener("DOMContentLoaded", function () {
  initVersusMenuEvents();
});
