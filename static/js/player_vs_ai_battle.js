// player_vs_ai_battle.js

/* 
  =====================
   全域變數
  =====================
*/
let pva_sse = null;          // SSE連線用
let pva_session_id = null;   // 從後端 /api/player_vs_ai_init 拿到
let pva_currentLogQueue = []; // 每回合回傳的 battle log
let pva_animationPlaying = false; // 動畫是否在播放中 (播放中則技能按鈕暫時禁用)

// 以下與 old battle.js 類似
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

// 事件類型 -> 文字樣式
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
   =====================
   初始化UI
   =====================
*/
function initPlayerVsAiView() {
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = "";

  const container = document.createElement("div");
  container.id = "player-vs-ai-container";

  // 上方區域: 顯示回合數、Hint 按鈕
  const topArea = document.createElement("div");
  topArea.id = "pva-top-area";
  topArea.innerHTML = `
    <div id="pva-round-indicator">回合: 0/0</div>
    <div class="pva-controls">
      <button id="pva-hint-btn">Hint</button>
    </div>
  `;
  container.appendChild(topArea);

  // 中間UI: 與 battle.js 類似 (左角色 / 中間回合資訊 / 右角色)
  const battleMain = document.createElement("div");
  battleMain.id = "pva-battle-main";

  // 左
  const leftPanel = document.createElement("div");
  leftPanel.id = "pva-left-panel";
  leftPanel.classList.add("character-panel");
  leftPanel.innerHTML = `
    <div class="avatar-container">
      <img id="pva-left-avatar" class="avatar-image" src="" alt="Left Avatar"/>
      <div id="pva-left-profession-name" class="profession-name">Player</div>
    </div>
    <div class="hp-bar-container">
      <div class="hp-text">
        <span id="pva-left-hp">0</span>/<span id="pva-left-max-hp">0</span>
      </div>
      <div class="hp-bar">
        <div class="hp-fill" id="pva-left-hp-fill"></div>
      </div>
    </div>
    <div class="stats-line" id="pva-left-stats">
      <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="pva-left-atk-val">x1.00</span></div>
      <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="pva-left-def-val">x1.00</span></div>
      <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="pva-left-heal-val">x1.00</span></div>
    </div>
    <div class="effects-list" id="pva-left-effects"></div>
    <div class="skills-list" id="pva-left-skills"></div>
    <div class="status-effects-layer" id="pva-left-status-effects-layer"></div>
  `;

  // 中間 => 本回合事件
  const turnBroadcast = document.createElement("div");
  turnBroadcast.id = "pva-turn-broadcast";
  turnBroadcast.innerHTML = `
    <h3>本回合事件</h3>
    <div id="pva-turn-broadcast-log"></div>
  `;

  // 右
  const rightPanel = document.createElement("div");
  rightPanel.id = "pva-right-panel";
  rightPanel.classList.add("character-panel");
  rightPanel.innerHTML = `
    <div class="avatar-container">
      <img id="pva-right-avatar" class="avatar-image" src="" alt="Right Avatar"/>
      <div id="pva-right-profession-name" class="profession-name">AI</div>
    </div>
    <div class="hp-bar-container">
      <div class="hp-text">
        <span id="pva-right-hp">0</span>/<span id="pva-right-max-hp">0</span>
      </div>
      <div class="hp-bar">
        <div class="hp-fill" id="pva-right-hp-fill"></div>
      </div>
    </div>
    <div class="stats-line" id="pva-right-stats">
      <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="pva-right-atk-val">x1.00</span></div>
      <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="pva-right-def-val">x1.00</span></div>
      <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="pva-right-heal-val">x1.00</span></div>
    </div>
    <div class="effects-list" id="pva-right-effects"></div>
    <div class="skills-list" id="pva-right-skills"></div>
    <div class="status-effects-layer" id="pva-right-status-effects-layer"></div>
  `;

  battleMain.appendChild(leftPanel);
  battleMain.appendChild(turnBroadcast);
  battleMain.appendChild(rightPanel);

  container.appendChild(battleMain);

  // 下方: 全回合事件
  const bottomArea = document.createElement("div");
  bottomArea.id = "pva-bottom-area";
  bottomArea.innerHTML = `
    <div id="pva-text-log" class="text-log"></div>
  `;
  container.appendChild(bottomArea);

  // 放到 content-area
  contentArea.appendChild(container);

  // 綁定事件
  document.getElementById("pva-hint-btn").addEventListener("click", handleHint);

}

/*
  =====================
  SSE 事件監聽 (stream)
  =====================
*/
function connectSSE(sessionId) {
  if (pva_sse) {
    pva_sse.close();
  }
  const url = `/api/player_vs_ai_stream/${sessionId}`;
  pva_sse = new EventSource(url);

  pva_sse.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      if (data.type === "round_result") {
        // 本回合 battle log => 播放動畫
        const log = data.round_battle_log || [];
        playRoundAnimation(log);
      } else if (data.type === "battle_end") {
        // 戰鬥結束
        addTextLog(`【戰鬥結束】${data.winner_text}`, "log-end");
        pva_sse.close();
        // 技能按鈕全部 disable
        disableAllSkills();
      }
    } catch (err) {
      console.error("SSE data parse error:", err);
    }
  };

  pva_sse.onerror = (err) => {
    console.error("SSE error:", err);
  };
}

function playRoundAnimation(battleLog) {
  // 播放本回合的事件
  pva_animationPlaying = true;
  clearTurnBroadcast();

  // 逐條處理
  let index = 0;
  const intervalId = setInterval(() => {
    if (index >= battleLog.length) {
      clearInterval(intervalId);
      pva_animationPlaying = false;
      // 回合事件播完後 => 可以再讓玩家選擇技能
      // (技能冷卻有沒有到，需要後端在 refresh_status 事件後再更新 UI)
      return;
    }
    const event = battleLog[index];
    handleBattleEvent(event);
    index++;
  }, 700); // 每0.7秒播一條，可自行調整
}

/*
  =====================
  前端操作: 點技能 => call /api/player_vs_ai_step
  =====================
*/
function onSkillClick(skillIdx) {
  // 若動畫在播 or skill disabled 則return
  if (pva_animationPlaying) return;

  // 呼叫後端 step
  fetch(`/api/player_vs_ai_step/${pva_session_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ skill_idx: skillIdx })
  })
  .then(res => res.json())
  .then(data => {
    if (data.done) {
      // 戰鬥結束
      console.log("battle done");
    } else {
      console.log("回合進行中...");
    }
  })
  .catch(err => console.error("player_vs_ai_step error:", err));
}

function disableAllSkills() {
  const leftSkills = document.getElementById("pva-left-skills");
  if (!leftSkills) return;
  const icons = leftSkills.querySelectorAll(".skill-icon-container");
  icons.forEach(icon => {
    icon.style.pointerEvents = "none";
    icon.style.opacity = "0.5";
  });
}

/*
  =====================
  Hint: call /api/player_vs_ai_hint
  =====================
*/
function handleHint() {
  if (!pva_session_id) return;
  fetch(`/api/player_vs_ai_hint/${pva_session_id}`)
    .then(res => res.json())
    .then(data => {
      if (data.action !== undefined) {
        // action => 0~3
        alert(`AI建議使用技能: ${data.action}\n${data.skill_name}\n${data.skill_desc}`);
      }
    })
    .catch(err => console.error("hint error:", err));
}

/*
  =====================
  初始化對戰
  =====================
*/
function startPlayerVsAiBattle(playerProf, enemyProf, model) {
  initPlayerVsAiView();

  // POST /api/player_vs_ai_init
  fetch("/api/player_vs_ai_init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      player_profession: playerProf,
      enemy_profession: enemyProf,
      model: model,
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.session_id) {
      pva_session_id = data.session_id;
      connectSSE(pva_session_id); // 連線 SSE
    } else {
      alert(data.error || "初始化失敗");
    }
  })
  .catch(err => console.error("init error:", err));
}

/*
  =====================
  下面是與 battle.js 類似的事件處理函式
  =====================
*/

function handleBattleEvent(event) {
  if (event.text) {
    const cls = EVENT_TEXT_CLASS_MAP[event.type] || "log-other";
    addTextLog(event.text, cls);
    if (event.type !== "turn_start") {
      addTurnBroadcastLine(event.text, cls);
    } else {
      clearTurnBroadcast();
      addTurnBroadcastLine("【回合開始】", "log-turn");
    }
  }

  switch (event.type) {
    case "turn_end":
      addTurnBroadcastLine("【回合結束】", "log-turn");
      break;
    case "damage":
    case "self_mutilation":
      if (event.appendix?.amount) {
        showFloatingNumber(`pva-${event.target || event.user}-panel`, `-${event.appendix.amount}`, "float-damage");
        redGlowAndShake(`pva-${event.target || event.user}-panel`);
      }
      break;
    case "heal":
      if (event.appendix?.amount) {
        showFloatingNumber(`pva-${event.target}-panel`, `+${event.appendix.amount}`, "float-heal");
        greenGlow(`pva-${event.target}-panel`);
      }
      break;
    case "skill":
      if (event.appendix?.relatively_skill_id !== undefined) {
        animateSkillIcon(event.user, event.appendix.relatively_skill_id);
      }
      break;
    case "status_apply":
    case "status_tick":
      if (event.appendix?.effect_name) {
        handleEffectAddOrTick(event);
      }
      break;
    case "status_remove":
      if (event.appendix?.effect_name) {
        handleEffectRemove(event);
      }
      break;
    case "refresh_status":
      if (event.appendix) {
        updateStatusBars(event.appendix);
      }
      break;
  }
}

function updateStatusBars(appendix) {
  // global
  if (appendix.global) {
    const roundElem = document.getElementById("pva-round-indicator");
    if (roundElem) {
      const r = appendix.global.round || 0;
      const mr = appendix.global.max_rounds || 0;
      roundElem.textContent = `回合: ${r}/${mr}`;
    }
  }
  // 左方
  if (appendix.left) {
    const hp = parseInt(appendix.left.hp) || 0;
    const maxHp = parseInt(appendix.left.max_hp) || 0;
    document.getElementById("pva-left-hp").textContent = hp;
    document.getElementById("pva-left-max-hp").textContent = maxHp;
    const fillL = document.getElementById("pva-left-hp-fill");
    let pct = (maxHp > 0) ? (hp / maxHp) * 100 : 0;
    fillL.style.width = Math.max(0, Math.min(100, pct)) + "%";

    if (appendix.global.left_profession) {
      document.getElementById("pva-left-profession-name").textContent = appendix.global.left_profession;
      const avatarUrl = `/static/images/${appendix.global.left_profession}.png`;
      const leftAvatar = document.getElementById("pva-left-avatar");
      leftAvatar.src = avatarUrl;
      leftAvatar.onerror = () => {
        leftAvatar.src = "/static/images/default_avatar.png";
      };
    }

    if (appendix.left.multiplier) {
      document.getElementById("pva-left-atk-val").textContent = "x" + parseFloat(appendix.left.multiplier.damage).toFixed(2);
      document.getElementById("pva-left-def-val").textContent = "x" + parseFloat(appendix.left.multiplier.defend).toFixed(2);
      document.getElementById("pva-left-heal-val").textContent = "x" + parseFloat(appendix.left.multiplier.heal).toFixed(2);
    }

    const leftEff = document.getElementById("pva-left-effects");
    if (leftEff) {
      leftEff.innerHTML = parseEffects(appendix.left.effects);
    }

    // 技能
    const leftSkills = document.getElementById("pva-left-skills");
    if (leftSkills && appendix.left.cooldowns) {
      const prof = appendix.global.left_profession;
      leftSkills.innerHTML = buildSkillsHTML(prof, appendix.left.cooldowns, true);
    }

    // buff glow
    const lPanel = document.getElementById("pva-left-panel");
    toggleBuffGlow(lPanel, appendix.left.effects);
  }

  // 右方
  if (appendix.right) {
    const hp = parseInt(appendix.right.hp) || 0;
    const maxHp = parseInt(appendix.right.max_hp) || 0;
    document.getElementById("pva-right-hp").textContent = hp;
    document.getElementById("pva-right-max-hp").textContent = maxHp;
    const fillR = document.getElementById("pva-right-hp-fill");
    let pct = (maxHp > 0) ? (hp / maxHp) * 100 : 0;
    fillR.style.width = Math.max(0, Math.min(100, pct)) + "%";

    if (appendix.global.right_profession) {
      document.getElementById("pva-right-profession-name").textContent = appendix.global.right_profession;
      const avatarUrl = `/static/images/${appendix.global.right_profession}.png`;
      const rightAvatar = document.getElementById("pva-right-avatar");
      rightAvatar.src = avatarUrl;
      rightAvatar.onerror = () => {
        rightAvatar.src = "/static/images/default_avatar.png";
      };
    }

    if (appendix.right.multiplier) {
      document.getElementById("pva-right-atk-val").textContent = "x" + parseFloat(appendix.right.multiplier.damage).toFixed(2);
      document.getElementById("pva-right-def-val").textContent = "x" + parseFloat(appendix.right.multiplier.defend).toFixed(2);
      document.getElementById("pva-right-heal-val").textContent = "x" + parseFloat(appendix.right.multiplier.heal).toFixed(2);
    }

    const rightEff = document.getElementById("pva-right-effects");
    if (rightEff) {
      rightEff.innerHTML = parseEffects(appendix.right.effects);
    }

    // 技能
    const rightSkills = document.getElementById("pva-right-skills");
    if (rightSkills && appendix.right.cooldowns) {
      const prof = appendix.global.right_profession;
      rightSkills.innerHTML = buildSkillsHTML(prof, appendix.right.cooldowns, false);
    }

    const rPanel = document.getElementById("pva-right-panel");
    toggleBuffGlow(rPanel, appendix.right.effects);
  }
}

function parseEffects(effectVector) {
  if (!effectVector || effectVector.length === 0) {
    return `<div class="no-effects">無狀態</div>`;
  }
  let htmlStr = "";
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const stacks = effectVector[i + 1];
    const maxStacks = effectVector[i + 2];
    const duration = effectVector[i + 3];
    const effSpecial = effectVector[i + 4];

    const effData = EFFECT_DATA[effId] || { name: `效果ID:${effId}`, type: "other" };
    let effName = effData.name;
    if (effData.type === "track" && typeof effSpecial === "string") {
      effName = effSpecial;
    }

    let baseClass = getEffectColorClass(effData.type);

    // 特殊判斷 (攻/防/治)
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

function getEffectColorClass(effType) {
  switch (effType) {
    case "buff": return "badge-buff";
    case "burning": return "badge-burning";
    case "poison": return "badge-poison";
    case "frozen": return "badge-frozen";
    case "bleeding": return "badge-bleeding";
    case "special": return "badge-special";
    case "track": return "badge-track";
    default: return "badge-other";
  }
}

function buildSkillsHTML(professionName, cooldowns, clickable) {
  let htmlStr = "";
  for (let i = 0; i < 4; i++) {
    const cd = cooldowns[i] !== undefined ? cooldowns[i] : 0;
    const url = `/static/images/${professionName}_skill_${i}.png`;
    const disabled = (cd > 0 || !clickable) ? true : false;
    htmlStr += createSkillIcon(url, cd, disabled, i);
  }
  return htmlStr;
}

function createSkillIcon(imgUrl, cooldown, disabled, skillIdx) {
  const disableStyle = disabled ? 'style="pointer-events: none; opacity:0.5;"' : '';
  return `
    <div class="skill-icon-container" ${disableStyle}
         onclick="onSkillClick(${skillIdx})">
      <img class="skill-icon"
           src="${imgUrl}"
           alt="skill_${skillIdx}"
           onerror="this.src='/static/images/skill_default.png';" />
      ${cooldown > 0 ? `<div class="skill-cd-overlay">${cooldown}</div>` : ``}
    </div>
  `;
}

function toggleBuffGlow(panel, effectVector) {
  panel.classList.remove(
    "buff-glow-attack", "buff-glow-attack-lower",
    "buff-glow-defense", "buff-glow-defense-lower",
    "buff-glow-heal", "buff-glow-heal-lower"
  );
  if (panel.buffGlowInterval) {
    clearInterval(panel.buffGlowInterval);
    delete panel.buffGlowInterval;
  }
  const glowClasses = [];
  for (let i = 0; i < effectVector.length; i += 5) {
    const effId = effectVector[i];
    const multiplier = effectVector[i + 4];
    switch (effId) {
      case 1:
        glowClasses.push(multiplier >= 1 ? "buff-glow-attack" : "buff-glow-attack-lower");
        break;
      case 2:
        glowClasses.push(multiplier >= 1 ? "buff-glow-defense" : "buff-glow-defense-lower");
        break;
      case 3:
        glowClasses.push(multiplier >= 1 ? "buff-glow-heal" : "buff-glow-heal-lower");
        break;
    }
  }
  if (glowClasses.length === 0) return;
  if (glowClasses.length === 1) {
    panel.classList.add(glowClasses[0]);
  } else {
    let idx = 0;
    panel.classList.add(glowClasses[idx]);
    panel.buffGlowInterval = setInterval(() => {
      panel.classList.remove(glowClasses[idx]);
      idx = (idx + 1) % glowClasses.length;
      panel.classList.add(glowClasses[idx]);
    }, 500);
  }
}

function addTextLog(msg, className = "") {
  const textLog = document.getElementById("pva-text-log");
  if (!textLog) return;
  const p = document.createElement("p");
  if (className) p.classList.add(className);
  p.textContent = msg;
  textLog.appendChild(p);
  textLog.scrollTop = textLog.scrollHeight;
}

function addTurnBroadcastLine(msg, className = "") {
  const tb = document.getElementById("pva-turn-broadcast-log");
  if (!tb) return;
  const p = document.createElement("p");
  if (className) p.classList.add(className);
  p.textContent = msg;
  tb.appendChild(p);
  tb.scrollTop = tb.scrollHeight;
}

function clearTurnBroadcast() {
  const tb = document.getElementById("pva-turn-broadcast-log");
  if (tb) tb.innerHTML = "";
}

function showFloatingNumber(panelId, text, floatClass) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  const div = document.createElement("div");
  div.className = `floating-number ${floatClass}`;
  div.textContent = text;
  panel.appendChild(div);
  setTimeout(() => {
    div.remove();
  }, 1200);
}

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

function animateSkillIcon(side, skillId) {
  const container = document.getElementById(`pva-${side}-skills`);
  if (!container) return;
  const icon = container.querySelectorAll(".skill-icon-container")[skillId];
  if (!icon) return;

  icon.classList.add("skill-activated");
  setTimeout(() => {
    icon.classList.remove("skill-activated");
  }, 800);
}

function handleEffectRemove(event) {
  const effType = getEffectTypeByName(event.appendix.effect_name);
  if (!effType) return;
  const layer = document.getElementById(`pva-${event.user}-status-effects-layer`);
  if (!layer) return;
  const div = layer.querySelector(`.${effType}-effect`);
  if (div) div.remove();
}

function handleEffectAddOrTick(event) {
  const effType = getEffectTypeByName(event.appendix.effect_name);
  if (!effType) return;
  const layer = document.getElementById(`pva-${event.user}-status-effects-layer`);
  if (!layer) return;
  let effDiv = layer.querySelector(`.${effType}-effect`);
  if (!effDiv) {
    effDiv = document.createElement("div");
    effDiv.className = `status-effect-layer ${effType}-effect`;
    layer.appendChild(effDiv);
    // 可依效果類型加粒子，略...
  }
}

function getEffectTypeByName(effectName) {
  for (let key in EFFECT_DATA) {
    if (EFFECT_DATA[key].name === effectName) {
      return EFFECT_DATA[key].type;
    }
  }
  return null;
}

// 導出到 window
window.startPlayerVsAiBattle = startPlayerVsAiBattle;
