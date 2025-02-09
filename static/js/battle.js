// backend/static/js/battle.js

/* 
  ================
  全域/設定
  ================
*/

// battleLog 資料、當前索引、timerID、播放速度
let battleLogGlobal = [];
let battleIndexGlobal = 0;
let battleTimerGlobal = null;
let battleSpeedGlobal = 1000; // 預設1秒一個事件
let battlePausedGlobal = false; // 是否暫停

const EFFECT_DATA = {
  1: { name: "攻擊力變更", type: "buff" },
  2: { name: "防禦力變更", type: "buff" },
  3: { name: "治癒力變更", type: "buff" },
  4: { name: "燃燒", type: "burning" },
  5: { name: "中毒", type: "poison" },
  6: { name: "凍結", type: "frozen" },
  7: { name: "免疫傷害", type: "special" },
  8: { name: "免疫控制", type: "special" },
  9: { name: "流血", type: "bleeding" },
  10: { name: "麻痺", type: "paralyzed" },
  11: { name: "生命值持續變更", type: "buff" },
  12: { name: "最大生命值變更", type: "buff" },
  13: { name: "追蹤", type: "track" },
};

/* 事件類型 -> 文字樣式 */
const EVENT_TEXT_CLASS_MAP = {
  damage: "log-damage",
  heal: "log-heal",
  skill: "log-skill",
  self_mutilation: "log-damage",
  status_apply: "log-status",
  status_apply_fail: "log-status-fail",
  status_remove: "log-status",
  status_duration_update: "log-status",
  status_stack_update: "log-status",
  status_set: "log-status",
  skip_turn: "log-skip",
  status_tick: "log-status",
  turn_start: "log-turn",
  turn_end: "log-turn",
  other: "log-other",
  text: "log-text",
};

/* 
  ================
  初始化戰鬥UI
  ================
*/
function initBattleView() {
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = "";

  const battleContainer = document.createElement("div");
  battleContainer.id = "battle-container";

  // ❶ 滿版寬度、使用 80vh 高度(可自行調整)、並設個 min-height
  battleContainer.style.width = "100%";
  battleContainer.style.height = "80vh";
  battleContainer.style.minHeight = "600px";

  // (1) 最上方: 回合 / 全域傷害倍率 / 進度條 / 速度控制
  const topControls = document.createElement("div");
  topControls.id = "battle-top-controls";
  topControls.innerHTML = `
    <div id="global-info">
      <span id="round-indicator">回合 0/0</span>
      <span id="global-damage-coeff">全域傷害倍率: 1.00</span>
    </div>
    <div id="battle-progress-container">
      <input type="range" id="battle-progress-bar" min="0" max="0" value="0">
    </div>
    <div id="speed-control-panel">
      <button id="pause-btn">繼續/暫停</button>
      <button class="speed-btn" data-speed="2000">0.5x</button>
      <button class="speed-btn" data-speed="1000">1x</button>
      <button class="speed-btn" data-speed="500">2x</button>
      <button class="speed-btn" data-speed="333">3x</button>
      <button class="speed-btn" data-speed="200">5x</button>
      <button class="speed-btn" data-speed="100">10x</button>
      <button class="speed-btn" data-speed="50">20x</button>
      <button id="skip-all-btn">再來一場</button>
      <button id="replay-btn" disabled>Replay</button>
    </div>
  `;
  battleContainer.appendChild(topControls);

  // (2) 中間UI: 左角色 / 中間(本回合事件) / 右角色
  const battleMain = document.createElement("div");
  battleMain.id = "battle-main";

  const leftPanel = document.createElement("div");
  leftPanel.id = "left-panel";
  leftPanel.classList.add("character-panel");
  leftPanel.innerHTML = `
    <div class="avatar-container">
      <img id="left-avatar" class="avatar-image" src="" alt="Left Avatar"/>
      <div id="left-profession-name" class="profession-name">Left</div>
    </div>
    <div class="hp-bar-container">
      <div class="hp-text">
        <span id="left-hp">0</span>/<span id="left-max-hp">0</span>
      </div>
      <div class="hp-bar">
        <div class="hp-fill" id="left-hp-fill"></div>
      </div>
    </div>
    <div class="stats-line" id="left-stats">
      <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="left-atk-val">x1.00</span></div>
      <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="left-def-val">x1.00</span></div>
      <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="left-heal-val">x1.00</span></div>
    </div>
    <div class="effects-list" id="left-effects"></div>
    <div class="skills-list" id="left-skills"></div>

    <!-- 狀態特效容器 (燃燒、冰凍...等粒子會加在這) -->
    <div class="status-effects-layer" id="left-status-effects-layer"></div>
  `;

  const turnBroadcast = document.createElement("div");
  turnBroadcast.id = "turn-broadcast";
  turnBroadcast.innerHTML = `
    <h3>本回合事件</h3>
    <div id="turn-broadcast-log"></div>
  `;

  const rightPanel = document.createElement("div");
  rightPanel.id = "right-panel";
  rightPanel.classList.add("character-panel");
  rightPanel.innerHTML = `
    <div class="avatar-container">
      <img id="right-avatar" class="avatar-image" src="" alt="Right Avatar"/>
      <div id="right-profession-name" class="profession-name">Right</div>
    </div>
    <div class="hp-bar-container">
      <div class="hp-text">
        <span id="right-hp">0</span>/<span id="right-max-hp">0</span>
      </div>
      <div class="hp-bar">
        <div class="hp-fill" id="right-hp-fill"></div>
      </div>
    </div>
    <div class="stats-line" id="right-stats">
      <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="right-atk-val">x1.00</span></div>
      <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="right-def-val">x1.00</span></div>
      <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="right-heal-val">x1.00</span></div>
    </div>
    <div class="effects-list" id="right-effects"></div>
    <div class="skills-list" id="right-skills"></div>

    <div class="status-effects-layer" id="right-status-effects-layer"></div>
  `;

  battleMain.appendChild(leftPanel);
  battleMain.appendChild(turnBroadcast);
  battleMain.appendChild(rightPanel);
  battleContainer.appendChild(battleMain);

  // (3) 下方: 全回合事件(文字播報)
  const bottomArea = document.createElement("div");
  bottomArea.id = "battle-bottom";
  bottomArea.innerHTML = `<div id="text-log" class="text-log"></div>`;
  battleContainer.appendChild(bottomArea);

  contentArea.appendChild(battleContainer);

  // 綁定 速度/暫停/重播
  const speedBtns = topControls.querySelectorAll(".speed-btn");
  speedBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const newSpeed = parseInt(btn.getAttribute("data-speed"), 10);
      changeBattleSpeed(newSpeed);
    });
  });

  const pauseBtn = document.getElementById("pause-btn");
  pauseBtn.addEventListener("click", () => {
    togglePause();
  });

  const replayBtn = document.getElementById("replay-btn");
  replayBtn.addEventListener("click", () => {
    replayBattle();
  });
  // 將原本的「跳過全部」按鈕改為「再來一場」
  const newBattleBtn = document.getElementById("skip-all-btn");
  newBattleBtn.textContent = "再來一場";
  newBattleBtn.addEventListener("click", () => {
    startNewBattle();
  });

  // 新增：進度條拖動控制 (這裡使用 change 事件，若想即時反映可改用 input 事件)
  const progressBar = document.getElementById("battle-progress-bar");
  progressBar.addEventListener("change", (e) => {
    const newIndex = parseInt(e.target.value, 10);
    seekBattle(newIndex);
  });

  // 注意：原本 skipAllEvents() 函式已不再使用
}

/* 
  ================
  播放控制
  ================
*/

function startBattle(battleLog) {
  battleLogGlobal = battleLog || [];
  battleIndexGlobal = 0;
  battleSpeedGlobal = 1000;
  battlePausedGlobal = false;
  clearTimeout(battleTimerGlobal);

  document.getElementById("text-log").innerHTML = "";
  document.getElementById("turn-broadcast-log").innerHTML = "";

  const replayBtn = document.getElementById("replay-btn");
  if (replayBtn) replayBtn.disabled = true;

  // 初始化進度條最大值與目前進度
  updateProgressBar();

  advanceBattle();
}

function advanceBattle() {
  if (battlePausedGlobal) return;
  if (battleIndexGlobal >= battleLogGlobal.length) {
    showBattleEnd();
    return;
  }
  const event = battleLogGlobal[battleIndexGlobal];
  handleBattleEvent(event);
  battleIndexGlobal++;
  updateProgressBar(); // 每次處理後更新進度條

  battleTimerGlobal = setTimeout(() => {
    advanceBattle();
  }, battleSpeedGlobal);
}

function changeBattleSpeed(newSpeed) {
  battleSpeedGlobal = newSpeed;
  if (!battlePausedGlobal) {
    clearTimeout(battleTimerGlobal);
    advanceBattle();
  }
}

function togglePause() {
  battlePausedGlobal = !battlePausedGlobal;
  const pauseBtn = document.getElementById("pause-btn");
  if (battlePausedGlobal) {
    pauseBtn.textContent = "繼續/暫停";
    clearTimeout(battleTimerGlobal);
  } else {
    pauseBtn.textContent = "繼續/暫停";
    advanceBattle();
  }
}

function replayBattle() {
  battleIndexGlobal = 0;
  document.getElementById("text-log").innerHTML = "";
  document.getElementById("turn-broadcast-log").innerHTML = "";
  startBattle(battleLogGlobal);
}

// 當戰鬥結束時，只顯示結束文字（不再顯示底部再來一場/返回選單的按鈕）
function showBattleEnd() {
  addTextLog("【戰鬥結束】", "log-end");
  const replayBtn = document.getElementById("replay-btn");
  if (replayBtn) replayBtn.disabled = true;
}

// 使用目前的配置發起一場新戰鬥
function startNewBattle() {
  if (window.currentBattleType === "pc_vs_pc") {
    const config = window.currentBattleConfig;
    if (window.showLoadingSpinner) window.showLoadingSpinner();
    const query = new URLSearchParams({ pr1: config.leftProfession, pr2: config.rightProfession });
    fetch(`/api/computer_vs_computer?${query.toString()}`)
      .then((res) => {
        if (!res.ok) {
          return res.text().then(text => { throw new Error(text); });
        }
        return res.json();
      })
      .then((data) => {
        if (window.hideLoadingSpinner) window.hideLoadingSpinner();
        const battleLog = data.battle_log;
        initBattleView();
        startBattle(battleLog);
      })
      .catch((error) => {
        if (window.hideLoadingSpinner) window.hideLoadingSpinner();
        alert("再來一場發生錯誤: " + error);
      });
  } else if (window.currentBattleType === "ai_vs_ai") {
    const config = window.currentBattleConfig;
    if (window.showLoadingSpinner) window.showLoadingSpinner();
    const query = new URLSearchParams({
      pr1: config.leftProfession,
      pr2: config.rightProfession,
      model1: config.leftModel,
      model2: config.rightModel
    });
    fetch(`/api/ai_vs_ai?${query.toString()}`)
      .then((res) => {
        if (!res.ok) {
          return res.text().then(text => { throw new Error(text); });
        }
        return res.json();
      })
      .then((data) => {
        if (window.hideLoadingSpinner) window.hideLoadingSpinner();
        const battleLog = data.battle_log;
        initBattleView();
        startBattle(battleLog);
      })
      .catch((error) => {
        if (window.hideLoadingSpinner) window.hideLoadingSpinner();
        alert("再來一場發生錯誤: " + error);
      });
  } else {
    alert("無法辨識目前的對戰模式。");
  }
}

// 返回選單 (這裡以重新載入頁面為例)
function returnToMainMenu() {
  location.reload();
  // 或者，如果你有自訂的主選單渲染函式，則可直接呼叫該函式：
  // showMainMenu();
}

/* 
  ================
  進度條與拖動尋找功能
  ================
*/

// 更新進度條狀態
function updateProgressBar() {
  const progressBar = document.getElementById("battle-progress-bar");
  if (progressBar && battleLogGlobal.length > 0) {
    progressBar.max = battleLogGlobal.length;
    progressBar.value = battleIndexGlobal;
  }
}

// 當進度條拖動時，從頭依序重播所有事件到指定位置
function seekBattle(newIndex) {
  // 暫停戰鬥
  battlePausedGlobal = true;
  clearTimeout(battleTimerGlobal);

  // 清空文字日誌與回合廣播（其他區塊可依需求重置）
  document.getElementById("text-log").innerHTML = "";
  document.getElementById("turn-broadcast-log").innerHTML = "";

  // 依序執行從 0 到 newIndex 的所有事件（不使用延遲）
  for (let i = 0; i < newIndex; i++) {
    handleBattleEvent(battleLogGlobal[i]);
  }
  battleIndexGlobal = newIndex;
  updateProgressBar();
}

/* 
  ================
  核心: 處理單一事件
  ================
*/
function handleBattleEvent(event) {
  // 1) 全域播報
  if (event.text) {
    const cls = EVENT_TEXT_CLASS_MAP[event.type] || "log-other";
    addTextLog(event.text, cls);
  }

  // 2) 本回合區
  if (event.type !== "turn_start" && event.text) {
    addTurnBroadcastLine(event.text, EVENT_TEXT_CLASS_MAP[event.type]);
  }

  switch (event.type) {
    case "turn_start":
      clearTurnBroadcast();
      addTurnBroadcastLine("【回合開始】", "log-turn");
      break;
    case "turn_end":
      addTurnBroadcastLine("【回合結束】", "log-turn");
      break;

    // (A) 自傷
    case "self_mutilation":
      if (event.appendix?.amount) {
        showFloatingNumber(
          `${event.user}-panel`,
          `-${event.appendix.amount}`,
          "float-damage"
        );
        redGlowAndShake(`${event.user}-panel`);
      }
      break;

    // (B) 傷害
    case "damage":
      if (event.appendix?.amount) {
        showFloatingNumber(
          `${event.target}-panel`,
          `-${event.appendix.amount}`,
          "float-damage"
        );
        redGlowAndShake(`${event.target}-panel`);
      }
      break;

    // (C) 治癒
    case "heal":
      if (event.appendix?.amount) {
        showFloatingNumber(
          `${event.target}-panel`,
          `+${event.appendix.amount}`,
          "float-heal"
        );
        greenGlow(`${event.target}-panel`);
      }
      break;

    // (D) skill
    case "skill":
      if (event.appendix?.relatively_skill_id !== undefined) {
        animateSkillIcon(event.user, event.appendix.relatively_skill_id);
      }
      break;

    // 狀態: apply / tick
    case "status_apply":
    case "status_tick":
      if (event.appendix?.effect_name) {
        handleEffectAddOrTick(event);
      }
      break;

    // 狀態: remove
    case "status_remove":
      if (event.appendix?.effect_name) {
        handleEffectRemove(event);
      }
      break;
  }

  // (E) 刷新面板
  if (event.type === "refresh_status" && event.appendix) {
    updateStatusBars(event.appendix);
  }
}

/* 使用技能 -> skill icon 動畫 */
function animateSkillIcon(side, skillId) {
  const container = document.getElementById(`${side}-skills`);
  if (!container) return;
  const icon = container.querySelector(`[data-skill-index="${skillId}"]`);
  if (!icon) return;

  icon.classList.add("skill-activated");
  setTimeout(() => {
    icon.classList.remove("skill-activated");
  }, 800);
}

/* =========================
   中間區(本回合)顯示
   ========================= */
function addTurnBroadcastLine(msg, className = "") {
  const tb = document.getElementById("turn-broadcast-log");
  if (!tb) return;
  const p = document.createElement("p");
  if (className) p.classList.add(className);
  p.textContent = msg;
  tb.appendChild(p);
  tb.scrollTop = tb.scrollHeight;
}

function clearTurnBroadcast() {
  const tb = document.getElementById("turn-broadcast-log");
  if (tb) tb.innerHTML = "";
}

/* 
  ================
  更新面板 (refresh_status)
  ================
*/
function updateStatusBars(appendix) {
  // 1) global info
  if (appendix.global) {
    const roundIndicator = document.getElementById("round-indicator");
    if (roundIndicator) {
      const r = appendix.global.round || 0;
      const maxR = appendix.global.max_rounds || 0;
      roundIndicator.textContent = `回合 ${r}/${maxR}`;
    }
    const gdc = document.getElementById("global-damage-coeff");
    if (gdc) {
      const val = parseFloat(appendix.global.damage_coefficient || 1).toFixed(2);
      gdc.textContent = `全域傷害倍率: ${val}`;
    }
  }

  // 2) 左方
  if (appendix.left) {
    const hp = parseInt(appendix.left.hp, 10);
    const maxHp = parseInt(appendix.left.max_hp, 10);
    // HP
    const leftHpElem = document.getElementById("left-hp");
    const leftMaxHpElem = document.getElementById("left-max-hp");
    const leftHpFill = document.getElementById("left-hp-fill");
    if (leftHpElem) leftHpElem.textContent = hp;
    if (leftMaxHpElem) leftMaxHpElem.textContent = maxHp;
    if (leftHpFill) {
      let pct = maxHp > 0 ? (hp / maxHp) * 100 : 0;
      leftHpFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
    }

    // 職業圖
    const leftAvatar = document.getElementById("left-avatar");
    const leftNameElem = document.getElementById("left-profession-name");
    if (appendix.global.left_profession && leftAvatar) {
      const professionName = appendix.global.left_profession;
      leftNameElem.textContent = professionName;
      const avatarUrl = `/static/images/${professionName}.png`;
      leftAvatar.src = avatarUrl;
      leftAvatar.onerror = () => {
        console.warn("角色圖載入失敗:", avatarUrl);
        leftAvatar.src = "/static/images/default_avatar.png";
      };
    }

    // 攻/防/治
    if (appendix.left.multiplier) {
      const atkVal = document.getElementById("left-atk-val");
      const defVal = document.getElementById("left-def-val");
      const healVal = document.getElementById("left-heal-val");
      if (atkVal)
        atkVal.textContent =
          "x" + parseFloat(appendix.left.multiplier.damage).toFixed(2);
      if (defVal)
        defVal.textContent =
          "x" + parseFloat(appendix.left.multiplier.defend).toFixed(2);
      if (healVal)
        healVal.textContent =
          "x" + parseFloat(appendix.left.multiplier.heal).toFixed(2);
    }

    // 效果列表
    const leftEffects = document.getElementById("left-effects");
    if (leftEffects) {
      leftEffects.innerHTML = parseEffects(appendix.left.effects);
    }

    // 技能列表 + 冷卻 (注意：此處已調整為 4 個技能)
    const leftSkills = document.getElementById("left-skills");
    if (leftSkills && appendix.left.cooldowns) {
      const prof = appendix.global.left_profession;
      leftSkills.innerHTML = buildSkillsHTML(prof, appendix.left.cooldowns);
    }

    // buff-glow
    const lPanel = document.getElementById("left-panel");
    toggleBuffGlow(lPanel, appendix.left.effects);
  }

  // 3) 右方
  if (appendix.right) {
    const hp = parseInt(appendix.right.hp, 10);
    const maxHp = parseInt(appendix.right.max_hp, 10);
    // HP
    const rightHpElem = document.getElementById("right-hp");
    const rightMaxHpElem = document.getElementById("right-max-hp");
    const rightHpFill = document.getElementById("right-hp-fill");
    if (rightHpElem) rightHpElem.textContent = hp;
    if (rightMaxHpElem) rightMaxHpElem.textContent = maxHp;
    if (rightHpFill) {
      let pct = maxHp > 0 ? (hp / maxHp) * 100 : 0;
      rightHpFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
    }

    // 職業圖
    const rightAvatar = document.getElementById("right-avatar");
    const rightNameElem = document.getElementById("right-profession-name");
    if (appendix.global.right_profession && rightAvatar) {
      const professionName = appendix.global.right_profession;
      rightNameElem.textContent = professionName;
      const avatarUrl = `/static/images/${professionName}.png`;
      rightAvatar.src = avatarUrl;
      rightAvatar.onerror = () => {
        console.warn("角色圖載入失敗:", avatarUrl);
        rightAvatar.src = "/static/images/default_avatar.png";
      };
    }

    // 攻/防/治
    if (appendix.right.multiplier) {
      const atkVal = document.getElementById("right-atk-val");
      const defVal = document.getElementById("right-def-val");
      const healVal = document.getElementById("right-heal-val");
      if (atkVal)
        atkVal.textContent =
          "x" + parseFloat(appendix.right.multiplier.damage).toFixed(2);
      if (defVal)
        defVal.textContent =
          "x" + parseFloat(appendix.right.multiplier.defend).toFixed(2);
      if (healVal)
        healVal.textContent =
          "x" + parseFloat(appendix.right.multiplier.heal).toFixed(2);
    }

    // 效果列表
    const rightEffects = document.getElementById("right-effects");
    if (rightEffects) {
      rightEffects.innerHTML = parseEffects(appendix.right.effects);
    }

    // 技能列表 + 冷卻 (此處同樣調整為 4 個技能)
    const rightSkills = document.getElementById("right-skills");
    if (rightSkills && appendix.right.cooldowns) {
      const prof = appendix.global.right_profession;
      rightSkills.innerHTML = buildSkillsHTML(prof, appendix.right.cooldowns);
    }

    // buff-glow
    const rPanel = document.getElementById("right-panel");
    toggleBuffGlow(rPanel, appendix.right.effects);
  }
}

/* 效果列表 -> HTML */
function parseEffects(effectVector) {
  if (!effectVector || effectVector.length === 0) {
    return `<div class="no-effects">無狀態</div>`;
  }
  let htmlStr = "";
  // 每五個數值代表一組效果：[effect id, stacks, max stacks, duration, eff_special]
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const stacks = effectVector[i + 1];
    const maxStacks = effectVector[i + 2];
    const duration = effectVector[i + 3];
    const effSpecial = effectVector[i + 4]; // 當 buff 類型時為 multiplier；當 track 類型時為 track 的真實名稱

    // 取出效果的基本資料，如果找不到則顯示效果ID
    const effData = EFFECT_DATA[effId] || { name: `效果ID:${effId}`, type: "other" };
    // 預設名稱為 EFFECT_DATA 裡的名稱
    let effName = effData.name;
    // 若效果類型為 track，則使用 effSpecial（track 的真實名稱）
    if (effData.type === "track" && typeof effSpecial === "string") {
      effName = effSpecial;
    }

    // 根據效果類型決定 badge 顏色
    let baseClass = getEffectColorClass(effData.type);

    // 如果是 buff 類型（攻/防/治），依 multiplier 判斷顏色變化
    if (effId === 1) {
      baseClass = effSpecial < 1 ? "badge-buff-attack-lower" : "badge-buff-attack";
    } else if (effId === 2) {
      baseClass = effSpecial < 1 ? "badge-buff-defense-lower" : "badge-buff-defense";
    } else if (effId === 3) {
      baseClass = effSpecial < 1 ? "badge-buff-heal-lower" : "badge-buff-heal";
    }

    htmlStr += `
      <div class="effect-badge ${baseClass}">
        <span class="eff-name">${effName}</span>
        <span class="eff-stack">(${stacks}/${maxStacks})</span>
        <span class="eff-duration">${duration}T</span>
      </div>
    `;
  }
  return htmlStr;
}

// 根據效果類型 -> 對應顏色class
function getEffectColorClass(effType) {
  switch (effType) {
    case "buff":
      return "badge-buff";
    case "dot":
      return "badge-dot";
    case "control":
      return "badge-control";
    case "special":
      return "badge-special";
    case "track":
      return "badge-track";
    default:
      return "badge-other";
  }
}

/* 生成技能列表HTML(帶冷卻)
   調整說明：
   由於每個職業的技能數量由 3 個變為 4 個，
   因此這裡改成固定從索引 0 迭代至 3，若某個技能在 cooldowns 中沒有值則預設為 0
*/
function buildSkillsHTML(professionName, cooldowns) {
  let htmlStr = "";
  for (let i = 0; i < 4; i++) {
    const cdVal = cooldowns[i] !== undefined ? cooldowns[i] : 0;
    const skillUrl = `/static/images/${professionName}_skill_${i}.png`;
    htmlStr += createSkillIcon(skillUrl, cdVal, false, i);
  }
  return htmlStr;
}

function createSkillIcon(imgUrl, cooldown, isPassive = false, skillIndex = 0) {
  const disableStyle =
    cooldown > 0 ? 'style="pointer-events: none; opacity:0.5;"' : "";

  return `
      <div class="skill-icon-container" ${disableStyle}>
        <img class="skill-icon"
             data-skill-index="${skillIndex}"
             src="${imgUrl}"
             alt="skill_${skillIndex}"
             onerror="this.src='/static/images/skill_default.png'; console.warn('技能圖載入失敗:', '${imgUrl}');" />
        ${cooldown > 0 ? `<div class="skill-cd-overlay">${cooldown}</div>` : ``}
        ${isPassive ? `<div class="skill-passive-label">被動</div>` : ``}
      </div>
    `;
}

/* 同時存在多個 buff 時，輪流顯示外框Glow */
function toggleBuffGlow(panel, effectVector) {
  // 1) 先清掉舊的 Glow
  panel.classList.remove(
    "buff-glow-attack",
    "buff-glow-attack-lower",
    "buff-glow-defense",
    "buff-glow-defense-lower",
    "buff-glow-heal",
    "buff-glow-heal-lower"
  );
  // 2) 清除舊 interval
  if (panel.buffGlowInterval) {
    clearInterval(panel.buffGlowInterval);
    delete panel.buffGlowInterval;
  }

  // 3) 收集所有需要輪流顯示的 glow class
  const glowClasses = [];
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const multiplier = effectVector[i + 4];
    switch (effId) {
      case 1:
        glowClasses.push(
          multiplier >= 1 ? "buff-glow-attack" : "buff-glow-attack-lower"
        );
        break;
      case 2:
        glowClasses.push(
          multiplier >= 1 ? "buff-glow-defense" : "buff-glow-defense-lower"
        );
        break;
      case 3:
        glowClasses.push(
          multiplier >= 1 ? "buff-glow-heal" : "buff-glow-heal-lower"
        );
        break;
    }
  }
  if (glowClasses.length === 0) return;

  if (glowClasses.length === 1) {
    // 只有1種光環
    panel.classList.add(glowClasses[0]);
  } else {
    // 多種 => 輪播
    let idx = 0;
    panel.classList.add(glowClasses[idx]);
    panel.buffGlowInterval = setInterval(() => {
      panel.classList.remove(glowClasses[idx]);
      idx = (idx + 1) % glowClasses.length;
      panel.classList.add(glowClasses[idx]);
    }, 500);
  }
}

// 新增文字播報
function addTextLog(msg, className = "") {
  const textLog = document.getElementById("text-log");
  if (!textLog) return;
  const p = document.createElement("p");
  p.innerHTML = msg;
  if (className) p.classList.add(className);
  textLog.appendChild(p);
  textLog.scrollTop = textLog.scrollHeight;
}

// 浮動數字
function showFloatingNumber(panelId, text, floatClass) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  const floatDiv = document.createElement("div");
  floatDiv.className = `floating-number ${floatClass}`;
  floatDiv.textContent = text;
  panel.appendChild(floatDiv);

  setTimeout(() => {
    floatDiv.remove();
  }, 1200);
}

/* 紅光 + 震動 */
function redGlowAndShake(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.style.zIndex = "999";
  panel.classList.add("red-glow", "shake");
  setTimeout(() => {
    panel.classList.remove("red-glow", "shake");
    panel.style.zIndex = "";
  }, 500);
}

/* 綠光 + 閃 */
function greenGlow(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.style.zIndex = "999";
  panel.classList.add("heal-effect");
  setTimeout(() => {
    panel.classList.remove("heal-effect");
    panel.style.zIndex = "";
  }, 800);
}

/* 取得effectName對應的type */
function getEffectTypeByName(effectName) {
  for (let key in EFFECT_DATA) {
    if (EFFECT_DATA[key].name === effectName) {
      return EFFECT_DATA[key].type;
    }
  }
  return null;
}

/* 移除特效 */
function handleEffectRemove(event) {
  const effectType = getEffectTypeByName(event.appendix.effect_name);
  if (!effectType) return;
  const layer = document.getElementById(`${event.user}-status-effects-layer`);
  if (!layer) return;

  const effDiv = layer.querySelector(`.${effectType}-effect`);
  if (effDiv) {
    effDiv.remove();
  }
}

/* 新增 or tick 特效 -> 這裡我們加了多粒子! */
function handleEffectAddOrTick(event) {
  const effectType = getEffectTypeByName(event.appendix.effect_name);
  if (!effectType) return;

  const layer = document.getElementById(`${event.user}-status-effects-layer`);
  if (!layer) return;

  // 檢查是否已經有該特效
  let effDiv = layer.querySelector(`.${effectType}-effect`);
  if (!effDiv) {
    // 沒有 -> 建立
    effDiv = document.createElement("div");
    effDiv.className = `status-effect-layer ${effectType}-effect`;
    layer.appendChild(effDiv);

    // === 多粒子示範 ===
    const particleCount = 12; // 粒子數量可自行調整
    // 根據不同效果，產生不同 class
    if (effectType === "burning") {
      // 清除之前的燃燒效果（如果有的話）
      effDiv.innerHTML = '';

      // 添加燃燒火焰容器
      const burningEffect = document.createElement("div");
      burningEffect.className = "burning-effect";
      effDiv.appendChild(burningEffect);

      // 添加火星粒子
      const sparkCount = 30; // 增加火星數量
      for (let i = 0; i < sparkCount; i++) {
        const spark = document.createElement("span");
        spark.className = "spark";
        // 隨機水平位置
        spark.style.left = Math.random() * 100 + "%";
        // 起始位置在火焰上方
        spark.style.bottom = Math.random() * -10 + "%";
        // 隨機延遲，讓火星不會同步飛出
        spark.style.animationDelay = Math.random() * 2 + "s";
        // 隨機大小
        spark.style.width = spark.style.height = (2 + Math.random() * 3) + "px";
        // 隨機顏色
        const colors = ['#FFD700', '#FFA500', '#FF4500', '#FF6347'];
        spark.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        // 隨機水平偏移
        spark.style.setProperty('--rand-x', Math.random() * 2 - 1);
        burningEffect.appendChild(spark);
      }
    } else if (effectType === "poison") {
      for (let i = 0; i < particleCount; i++) {
        const bubble = document.createElement("span");
        bubble.className = "poison-bubble";
        bubble.style.top = Math.random() * 100 + "%";
        bubble.style.left = Math.random() * 100 + "%";
        effDiv.appendChild(bubble);
      }
    } else if (effectType === "bleeding") {
      for (let i = 0; i < particleCount; i++) {
        const drop = document.createElement("span");
        drop.className = "bleed-drop";
        drop.style.top = Math.random() * 100 + "%";
        drop.style.left = Math.random() * 100 + "%";
        effDiv.appendChild(drop);
      }
    } else if (effectType === "frozen") {
      for (let i = 0; i < particleCount; i++) {
        const shard = document.createElement("span");
        shard.className = "ice-shard";
        shard.style.top = Math.random() * 100 + "%";
        shard.style.left = Math.random() * 100 + "%";
        effDiv.appendChild(shard);
      }
    }
    // 其他效果可依需求擴充
  }

  // 若是 tick 時，可在此額外添加閃動或其他效果
}

/* 
  ================
  導出
  ================
*/
window.initBattleView = initBattleView;
window.startBattle = startBattle;
