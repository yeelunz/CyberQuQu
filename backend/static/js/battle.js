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
  9: { name: "流血", type: "dot" },
  10: { name: "麻痺", type: "stunned" },
  11: { name: "回血", type: "buff" },
  12: { name: "最大生命值變更", type: "buff" },
  13: { name: "追蹤", type: "track" },
  19: { name: "自定義傷害效果", type: "dot" },
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

  // (1) 最上方: 回合 / 全域傷害倍率 / 速度控制
  const topControls = document.createElement("div");
  topControls.id = "battle-top-controls";
  topControls.innerHTML = `
    <div id="global-info">
      <span id="round-indicator">回合 0/0</span>
      <span id="global-damage-coeff">全域傷害倍率: 1.00</span>
    </div>
    <div id="speed-control-panel">
      <button class="speed-btn" data-speed="1000">1x</button>
      <button class="speed-btn" data-speed="500">2x</button>
      <button class="speed-btn" data-speed="333">3x</button>
      <button id="pause-btn">暫停</button>
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
  // 注意: 我們加一個 .status-effects-layer 來放狀態子元素
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

    <!-- 此容器專門放"多個"狀態特效層 (燃燒,冰凍...都可同存) -->
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

  contentArea.appendChild(battleContainer);
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
    pauseBtn.textContent = "繼續";
    clearTimeout(battleTimerGlobal);
  } else {
    pauseBtn.textContent = "暫停";
    advanceBattle();
  }
}

function replayBattle() {
  battleIndexGlobal = 0;
  document.getElementById("text-log").innerHTML = "";
  document.getElementById("turn-broadcast-log").innerHTML = "";
  startBattle(battleLogGlobal);
}

function showBattleEnd() {
  addTextLog("【戰鬥結束】", "log-end");
  const replayBtn = document.getElementById("replay-btn");
  if (replayBtn) replayBtn.disabled = false;
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

  // 2) 本回合區 (不顯示 turn_start, turn_end 也可以；視需求)
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

    // (A) 自傷 -> event.user 面板: 紅光+震動 + 浮動傷害
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

    // (B) 傷害 -> event.target 面板: 紅光+震動 + 浮動傷害
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

    // (C) 治癒 -> event.target 面板: 綠光+閃 + 浮動治癒
    case "heal":
      if (event.appendix?.amount) {
        showFloatingNumber(
          `${event.user}-panel`,
          `+${event.appendix.amount}`,
          "float-heal"
        );
        greenGlow(`${event.target}-panel`);
      }
      break;

    // (D) skill -> 在 user 對應的 skill icon 做動畫
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

  // 若是 refresh_status -> 更新面板
  if (event.type === "refresh_status" && event.appendix) {
    updateStatusBars(event.appendix);
  }
}

/* 使用技能 -> 對應 skill icon 做動畫(放大/閃爍) */
function animateSkillIcon(side, skillId) {
  // skillId 可能是 0,1,2
  const container = document.getElementById(`${side}-skills`);
  if (!container) return;
  // 在 buildSkillHTML 時，可以給每個 skill icon 加 data-skill-index="0"
  // 這裡就能 select
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
  p.textContent = msg; // or innerHTML
  tb.appendChild(p);
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
      const val = parseFloat(appendix.global.damage_coefficient || 1).toFixed(
        2
      );
      gdc.textContent = `全域傷害倍率: ${val}`;
    }
  }

  // 2) 更新 左方
  const lPanel = document.getElementById("left-panel");
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
      let pct = 0;
      if (maxHp > 0) {
        pct = (hp / maxHp) * 100;
      }
      leftHpFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
    }

    // 職業圖
    const leftAvatar = document.getElementById("left-avatar");
    const leftNameElem = document.getElementById("left-profession-name");
    if (appendix.global.left_profession && leftAvatar) {
      const professionName = appendix.global.left_profession; // 後端給
      leftNameElem.textContent = professionName;
      const avatarUrl = `/static/images/${professionName}.png`;
      leftAvatar.src = avatarUrl;

      // 如果載入失敗 -> 預設圖 + Debug
      leftAvatar.onerror = () => {
        console.warn("角色圖載入失敗:", avatarUrl);
        leftAvatar.src = "/static/images/default_avatar.png";
      };
    }

    // 攻/防/治 multiplier
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

    // 技能列表 + 冷卻
    const leftSkills = document.getElementById("left-skills");
    if (leftSkills && appendix.left.cooldowns) {
      const prof = appendix.global.left_profession;
      leftSkills.innerHTML = buildSkillsHTML(prof, appendix.left.cooldowns);
    }

    // 檢查是否有 buff -> 加 "buff-glow"
    toggleBuffGlow(lPanel, appendix.left.effects);
  }

  // 3) 更新 右方
  const rPanel = document.getElementById("right-panel");
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
      let pct = 0;
      if (maxHp > 0) {
        pct = (hp / maxHp) * 100;
      }
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

    // 技能列表 + 冷卻
    const rightSkills = document.getElementById("right-skills");
    if (rightSkills && appendix.right.cooldowns) {
      const prof = appendix.global.right_profession;
      rightSkills.innerHTML = buildSkillsHTML(prof, appendix.right.cooldowns);
    }

    // buff glow
    toggleBuffGlow(rPanel, appendix.right.effects);
  }
}

/* 
  ================
  UI / 動畫 / 效果
  ================
*/

// 效果列表 -> 產生HTML
function parseEffects(effectVector) {
  if (!effectVector || effectVector.length === 0) {
    return `<div class="no-effects">無狀態</div>`;
  }
  // 每 5 個: [effectID, stack, maxStack, duration, multiplier]
  let htmlStr = "";
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const stacks = effectVector[i + 1];
    const maxStacks = effectVector[i + 2];
    const duration = effectVector[i + 3];
    // const mult = effectVector[i+4]; // 可能用不到

    // 取得效果名稱 + 類型(決定顏色)
    const effData = EFFECT_DATA[effId] || {
      name: `效果ID:${effId}`,
      type: "other",
    };
    const effName = effData.name;
    const effType = effData.type;
    const colorClass = getEffectColorClass(effType);

    htmlStr += `
      <div class="effect-badge ${colorClass}">
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

// 生成技能列表HTML(帶冷卻)
function buildSkillsHTML(professionName, cooldowns) {
  // cooldowns 範例: { "0": 2, "1": 0, "2": 5 }
  // 也假設有被動 skillIndex = "passive"？
  let htmlStr = "";
    // debug 輸出cooldowns
    console.log("professionName: ", professionName);
    console.log(cooldowns);

  
  // 再處理一般技能(0~N)
  for (let i in cooldowns) {
    if (i === "passive") continue; // 跳過被動
    const skillIndex = parseInt(i, 10);
    const skillUrl = `/static/images/${professionName}_skill_${skillIndex}.png`;
    const cdVal = cooldowns[i];
    htmlStr += createSkillIcon(skillUrl, cdVal, false, skillIndex);
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

// 若有任一 buff 效果 -> 加 "buff-glow"
function toggleBuffGlow(panel, effectVector) {
  if (!panel || !effectVector) return;
  // 檢查是否有 buff
  let hasBuff = false;
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const effData = EFFECT_DATA[effId];
    if (effData && effData.type === "buff") {
      hasBuff = true;
      break;
    }
  }
  if (hasBuff) {
    panel.classList.add("buff-glow");
  } else {
    panel.classList.remove("buff-glow");
  }
}

// 中間大字提示
function showCenterNotice(msg, extraClass = "") {
  const animationArea = document.getElementById("animation-area");
  if (!animationArea) return;
  const div = document.createElement("div");
  div.className = `center-notice ${extraClass}`;
  div.textContent = msg;
  animationArea.appendChild(div);
  setTimeout(() => {
    if (div.parentNode) div.parentNode.removeChild(div);
  }, 1500);
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

// 顯示浮動數字 (ex: -100, +50)
// 浮動數字(大字+粗體)
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
  
    // 檢查原本是否有 buff-glow
    const hadBuffGlow = panel.classList.contains("buff-glow");
    if (hadBuffGlow) {
      panel.classList.remove("buff-glow");
    }
  
    // 播動畫
    panel.style.zIndex = "999"; // 讓它在最上層
    panel.classList.add("red-glow", "shake");
    setTimeout(() => {
      panel.classList.remove("red-glow", "shake");
      panel.style.zIndex = "";
  
      // 播完再加回 buff-glow
      if (hadBuffGlow) {
        panel.classList.add("buff-glow");
      }
    }, 500);
  }
  
  /* 綠光 + 閃 */
  function greenGlow(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
  
    const hadBuffGlow = panel.classList.contains("buff-glow");
    if (hadBuffGlow) {
      panel.classList.remove("buff-glow");
    }
  
    panel.style.zIndex = "999";
    panel.classList.add("heal-effect");
    setTimeout(() => {
      panel.classList.remove("heal-effect");
      panel.style.zIndex = "";
      if (hadBuffGlow) {
        panel.classList.add("buff-glow");
      }
    }, 800);
  }



// 技能動畫 (示例：微晃動 + 閃光)
function skillAnimation(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  // 加個class
  panel.style.zIndex = "999";
  panel.classList.add("skill-cast");
  setTimeout(() => {
    panel.classList.remove("skill-cast");
  }, 800);
}

/* 狀態: 燃燒/中毒/冰凍/暈眩 => Apply/Remove */
function handleEffectApplyOrTick(event) {
  // event.user = "left" or "right"
  const panel = document.getElementById(`${event.user}-panel`);
  if (!panel) return;

  const effName = event.appendix.effect_name; // "燃燒", "中毒"...
  // 用簡易 mapping
  let cssClass = "";
  switch (effName) {
    case "燃燒":
      cssClass = "effect-burning";
      break;
    case "中毒":
      cssClass = "effect-poison";
      break;
    case "凍結":
      cssClass = "effect-frozen";
      break;
    case "麻痺":
      cssClass = "effect-stunned";
      break;
  }
  if (!cssClass) return;

  // (1) Apply / Tick 都加 class (若已加過也不重複)
  // 讓它持續到 remove
  if (!panel.classList.contains(cssClass)) {
    panel.classList.add(cssClass);
  }
}

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

// 根據狀態中文名 => type
function getEffectTypeByName(effectName) {
  // 在 EFFECT_DATA 裡反查
  for (let key in EFFECT_DATA) {
    if (EFFECT_DATA[key].name === effectName) {
      return EFFECT_DATA[key].type;
    }
  }
  return null;
}

function handleEffectAddOrTick(event) {
  // ex: effect_name="燃燒" => type="burning"
  const effectType = getEffectTypeByName(event.appendix.effect_name);
  if (!effectType) return;

  const layer = document.getElementById(`${event.user}-status-effects-layer`);
  if (!layer) return;

  // 檢查是否已經有該特效的子元素
  let effDiv = layer.querySelector(`.${effectType}-effect`);
  if (!effDiv) {
    // 沒有 -> 建立一個
    effDiv = document.createElement("div");
    effDiv.className = `status-effect-layer ${effectType}-effect`;
    layer.appendChild(effDiv);
  }

  // 如果是 tick，保持不動就好
  // (若你想在 tick 時加一次閃動，可再加)
}

/* 
  ================
  導出
  ================
*/
window.initBattleView = initBattleView;
window.startBattle = startBattle;
