// player_vs_ai.js
(function () {
  // 全域變數
  let pva_session_id = null;
  let pva_sse = null;
  let pva_animationPlaying = false; // 是否正在播放回合事件動畫
  let pva_animationInterval = 300; // 預設 300ms
  let latestAppendix = null; // 儲存最新的 refresh 資料

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

  // 建立一個格式化描述的函式
  function formatSkillDescription(rawText) {
    let text = rawText;

    // 傷害數值（紅色）
    text = text.replace(/對.*?造成 (\d+) 點傷害/g, (match, p1) => {
      return `對敵方造成 <span class="damage-text">${p1}</span> 點傷害`;
    });

    text = text.replace(/額外造成 (\d+) 點傷害/g, (match, p1) => {
      return `額外造成 <span class="damage-text">${p1}</span> 點傷害`;
    });

    text = text.replace(/反嗜 (\d+) 的攻擊傷害/g, (match, p1) => {
      return `反嗜 <span class="damage-text">${p1}</span> 的攻擊傷害`;
    });

    text = text.replace(/消耗 (\d+) 點生命值/g, (match, p1) => {
      return `消耗 <span class="damage-text">${p1}</span> 點生命值`;
    });

    text = text.replace(
      /對攻擊者立即造成其本次攻擊傷害的 (\d+) /g,
      (match, p1) => {
        return `對攻擊者立即造成其本次攻擊傷害的 <span class="damage-text">${p1}</span> `;
      }
    );

    // 治癒數值（綠色）
    text = text.replace(/恢復 (\d+) 點生命值/g, (match, p1) => {
      return `恢復 <span class="heal-text">${p1}</span> 點生命值`;
    });

    text = text.replace(/恢復造成傷害次數 (\d+) 的血量/g, (match, p1) => {
      return `恢復造成傷害次數 <span class="heal-text">${p1}</span> 的血量`;
    });

    // 機率（紫色）
    text = text.replace(/(\d+%) 機率/g, (match, p1) => {
      return `<span class="probability-text">${p1}</span> 機率`;
    });

    // 持續回合（黃色）
    text = text.replace(/(持續|接下來) (\d+) 回合/g, (match, p1, p2) => {
      return `${p1} <span class="duration-text">${p2}</span> 回合`;
    });

    // 效果百分比（黃色）
    text = text.replace(
      /(提升|降低|降低其|增加|提升自身|增加自身|降低自身) (\d+)%/g,
      (match, p1, p2) => {
        return `${p1} <span class="effect-text">${p2}%</span>`;
      }
    );

    // 扣除生命值（紅色）
    text = text.replace(/扣除 (\d+) 點生命值/g, (match, p1) => {
      return `扣除 <span class="deduct-health-text">${p1}</span> 點生命值`;
    });

    // 暈眩回合數（黃色）
    text = text.replace(/暈眩 (\d+)~(\d+) 回合/g, (match, p1, p2) => {
      return `暈眩 <span class="stun-duration-text">${p1}~${p2}</span> 回合`;
    });

    // 傷害倍數（深紅色）
    text = text.replace(/(\d+%) 的傷害/g, (match, p1) => {
      return `<span class="multiplier-text">${p1}</span> 的傷害`;
    });

    // 異常狀態（藍色粗體）
    text = text.replace(/(冰凍|中毒|流血|燃燒|麻痺)/g, (match) => {
      let colorClass = "";
      switch (match) {
        case "冰凍":
          colorClass = "frozen-text";
          break;
        case "中毒":
          colorClass = "poisoned-text";
          break;
        case "流血":
          colorClass = "bleeding-text";
          break;
        case "燃燒":
          colorClass = "burning-text";
          break;
        case "麻痺":
          colorClass = "paralyzed-text";
          break;
        default:
          colorClass = "status-text";
      }
      return `<span class="${colorClass}">${match}</span>`;
    });

    return text;
  }

  // 以下程式碼大致與原先相同，重點修改在 buildSkillCardsHTML 內
  // 取得職業清單、showLoadingSpinner、hideLoadingSpinner、showPlayerVsAiForm、startPlayerVsAiBattle、initPlayerVsAiBattleView、connectPvaSSE 等保持不變
  async function fetchProfessions() {
    try {
      const res = await fetch("/api/list_professions");
      const data = await res.json();
      return data.professions || [];
    } catch (e) {
      console.warn("[PVA] 無法取得職業清單", e);
      return [];
    }
  }

  function showLoadingSpinner() {
    const spinner = document.getElementById("model-loading-spinner");
    if (spinner) spinner.style.display = "flex";
  }
  function hideLoadingSpinner() {
    const spinner = document.getElementById("model-loading-spinner");
    if (spinner) spinner.style.display = "none";
  }

  async function showPlayerVsAiForm() {
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = "";
    const wrapper = document.createElement("div");
    wrapper.classList.add("versus-mode");
    const professions = await fetchProfessions();
    professions.push("Random");
    let modelList = [];
    try {
      const res = await fetch("/api/list_saved_models_simple");
      const data = await res.json();
      modelList = data.models || [];
    } catch (err) {
      console.warn("[PVA] 無法取得模型列表", err);
    }
    const leftPanel = document.createElement("div");
    leftPanel.classList.add("form-panel");
    leftPanel.innerHTML = `
      <h3>我方 (玩家)</h3>
      <label>職業：
        <select id="pva-player-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
      <label>AI 模型：
        ${
          modelList.length > 0
            ? `<select id="pva-player-ai-model">
                 <option value="">請選擇模型</option>
                 ${modelList
                   .map((m) => `<option value="${m}">${m}</option>`)
                   .join("")}
               </select>`
            : `<span style="color:red;">(尚無已訓練模型)</span>`
        }
      </label>
    `;
    const rightPanel = document.createElement("div");
    rightPanel.classList.add("form-panel");
    rightPanel.innerHTML = `
      <h3>敵方 AI</h3>
      <label>職業：
        <select id="pva-enemy-profession">
          ${professions
            .map((pr) => `<option value="${pr}">${pr}</option>`)
            .join("")}
        </select>
      </label>
      <label>AI 模型：
        ${
          modelList.length > 0
            ? `<select id="pva-enemy-ai-model">
                 <option value="">請選擇模型</option>
                 ${modelList
                   .map((m) => `<option value="${m}">${m}</option>`)
                   .join("")}
               </select>`
            : `<span style="color:red;">(尚無已訓練模型)</span>`
        }
      </label>
    `;
    const row = document.createElement("div");
    row.classList.add("versus-form-row");
    row.appendChild(leftPanel);
    row.appendChild(rightPanel);
    const buttonRow = document.createElement("div");
    buttonRow.classList.add("button-row");
    const startBtn = document.createElement("button");
    startBtn.textContent = "開始對戰";
    startBtn.addEventListener("click", startPlayerVsAiBattle);
    buttonRow.appendChild(startBtn);
    wrapper.appendChild(row);
    wrapper.appendChild(buttonRow);
    contentArea.appendChild(wrapper);
  }

  async function startPlayerVsAiBattle() {
    const playerProfession =
      document.getElementById("pva-player-profession")?.value || "";
    const enemyProfession =
      document.getElementById("pva-enemy-profession")?.value || "";
    const playerModel =
      document.getElementById("pva-player-ai-model")?.value || "";
    const enemyModel =
      document.getElementById("pva-enemy-ai-model")?.value || "";
    console.log("[PVA] StartBattle param:", {
      playerProfession,
      enemyProfession,
      playerModel,
      enemyModel,
    });
    if (!playerModel || !enemyModel) {
      alert("請選擇模型 (檢查是否有已訓練模型)");
      return;
    }
    showLoadingSpinner();
    try {
      const res = await fetch("/api/player_vs_ai_init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_profession: playerProfession,
          enemy_profession: enemyProfession,
          model1: playerModel,
          model2: enemyModel,
        }),
      });
      if (!res.ok) {
        hideLoadingSpinner();
        const errData = await res.json().catch(() => {});
        alert("初始化錯誤: " + (errData.error || res.status));
        return;
      }
      const data = await res.json();
      hideLoadingSpinner();
      if (data.session_id) {
        pva_session_id = data.session_id;
        console.log("[PVA] init ok, session_id:", pva_session_id);
        initPlayerVsAiBattleView();
        connectPvaSSE(pva_session_id);
      } else {
        alert("初始化失敗: 未取得 session_id");
      }
    } catch (err) {
      hideLoadingSpinner();
      alert("初始化對戰錯誤: " + err);
      console.error(err);
    }
  }

  function initPlayerVsAiBattleView() {
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = "";
    const battleContainer = document.createElement("div");
    battleContainer.id = "battle-container";
    battleContainer.classList.add("pva-battle-container");
    const topControls = document.createElement("div");
    topControls.id = "battle-top-controls";
    topControls.innerHTML = `
      <div id="global-info">
        <span id="round-indicator">回合: 0/0</span>
        <span id="global-damage-coeff">全域傷害倍率: 1.00</span>
      </div>
      <div id="speed-control-panel">
        <button class="speed-btn" data-speed="1000">1x</button>
        <button class="speed-btn" data-speed="500">2x</button>
        <button class="speed-btn" data-speed="300">3x</button>
        <button class="speed-btn" data-speed="200">5x</button>
        <button class="speed-btn" data-speed="100">10x</button>
      </div>
    `;
    battleContainer.appendChild(topControls);
    const battleMain = document.createElement("div");
    battleMain.id = "battle-main";
    battleMain.innerHTML = `
      <div id="left-panel" class="character-panel">
        <div class="avatar-container">
          <img id="left-avatar" class="avatar-image" src="" alt="Left Avatar"/>
          <div id="left-profession-name" class="profession-name">Player</div>
        </div>
        <div class="hp-bar-container">
          <div class="hp-text">
            <span id="left-hp">0</span>/<span id="left-max-hp">0</span>
          </div>
          <div class="hp-bar">
            <div class="hp-fill" id="left-hp-fill"></div>
          </div>
          <div class="stats-line" id="left-stats">
            <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="left-atk-val">x1.00</span></div>
            <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="left-def-val">x1.00</span></div>
            <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="left-heal-val">x1.00</span></div>
          </div>
          <div class="effects-list" id="left-effects"></div>
          <div class="skills-list" id="left-skills"></div>
        </div>
      </div>
      <div id="turn-broadcast" class="turn-broadcast">
        <h3>本回合事件</h3>
        <div id="turn-broadcast-log"></div>
      </div>
      <div id="right-panel" class="character-panel">
        <div class="avatar-container">
          <img id="right-avatar" class="avatar-image" src="" alt="Right Avatar"/>
          <div id="right-profession-name" class="profession-name">AI</div>
        </div>
        <div class="hp-bar-container">
          <div class="hp-text">
            <span id="right-hp">0</span>/<span id="right-max-hp">0</span>
          </div>
          <div class="hp-bar">
            <div class="hp-fill" id="right-hp-fill"></div>
          </div>
          <div class="stats-line" id="right-stats">
            <div class="stat-item stat-atk"><span class="stat-icon">⚔</span> <span id="right-atk-val">x1.00</span></div>
            <div class="stat-item stat-def"><span class="stat-icon">🛡</span> <span id="right-def-val">x1.00</span></div>
            <div class="stat-item stat-heal"><span class="stat-icon">✚</span> <span id="right-heal-val">x1.00</span></div>
          </div>
          <div class="effects-list" id="right-effects"></div>
          <div class="skills-list" id="right-skills"></div>
        </div>
      </div>
    `;
    battleContainer.appendChild(battleMain);
    contentArea.appendChild(battleContainer);
    const skillContainer = document.createElement("div");
    skillContainer.id = "battle-skill-container";
    skillContainer.classList.add("pva-battle-skill-container");
    const bottomArea = document.createElement("div");
    bottomArea.id = "battle-skill-cards";
    bottomArea.classList.add("pva-battle-skill-cards");
    skillContainer.appendChild(bottomArea);
    contentArea.appendChild(skillContainer);
    const speedBtns = topControls.querySelectorAll(".speed-btn");
    speedBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const newSpeed = parseInt(btn.getAttribute("data-speed"), 10);
        pva_animationInterval = newSpeed;
        console.log("[PVA] 改變播放速度:", newSpeed);
      });
    });
  }

  function connectPvaSSE(sessionId) {
    if (pva_sse) pva_sse.close();
    const url = `/api/player_vs_ai_stream/${sessionId}`;
    pva_sse = new EventSource(url);
    pva_sse.onmessage = function (evt) {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "refresh_status") {
          updateStatusBars(data.appendix);
        } else if (data.type === "round_result") {
          const log = data.round_battle_log || [];
          playRoundAnimation(log);
        } else if (data.type === "battle_end") {
          addTurnBroadcastLine("【戰鬥結束】 " + data.winner_text, "log-end");
          pva_sse.close();
          disableAllSkills();
        }
      } catch (err) {
        console.error("[PVA] SSE 錯誤:", err);
      }
    };
    pva_sse.onerror = function (err) {
      console.error("[PVA] SSE onerror:", err);
    };
  }

  function playRoundAnimation(battleLog) {
    pva_animationPlaying = true;
    clearTurnBroadcast();
    let idx = 0;
    const intervalId = setInterval(() => {
      if (idx >= battleLog.length) {
        clearInterval(intervalId);
        setTimeout(() => {
          if (latestAppendix) {
            updateStatusBars(latestAppendix, true);
          }
          pva_animationPlaying = false;
        }, pva_animationInterval);
        return;
      }
      const event = battleLog[idx];
      handleBattleEvent(event);
      idx++;
    }, pva_animationInterval);
  }

  async function onSkillClick(skillIdx) {
    if (!pva_session_id) return;
    if (pva_animationPlaying) return;
    disableAllSkills();
    try {
      const res = await fetch(`/api/player_vs_ai_step/${pva_session_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ skill_idx: skillIdx }),
      });
      const data = await res.json();
      if (data.done) {
        console.log("[PVA] 戰鬥結束");
      }
    } catch (err) {
      console.error("[PVA] onSkillClick error:", err);
    }
  }

  function disableAllSkills() {
    const skillCardsDiv = document.getElementById("battle-skill-cards");
    if (!skillCardsDiv) return;
    const cards = skillCardsDiv.querySelectorAll(".pva-skill-card");
    cards.forEach((card) => {
      card.style.pointerEvents = "none";
      card.style.opacity = "0.5";
    });
  }

  // 更新技能卡狀態（forceUpdate 為 true 時強制更新，不在動畫期間時更新 disable 狀態）
  function updateSkillCardsState(appendix, forceUpdate = false) {
    if (appendix.left.cooldowns) {
      const skillCards = document.getElementById("battle-skill-cards");
      for (let i = 0; i < 4; i++) {
        const cdVal =
          appendix.left.cooldowns[i] !== undefined
            ? appendix.left.cooldowns[i]
            : 0;
        const card = skillCards.querySelector(
          `.pva-skill-card[data-skill-index="${i}"]`
        );
        if (card) {
          if (cdVal > 0) {
            card.style.pointerEvents = "none";
            card.style.opacity = "0.5";
            card.classList.add("disabled");
          } else {
            if (forceUpdate) {
              card.style.pointerEvents = "auto";
              card.style.opacity = "1";
              card.classList.remove("disabled");
            }
          }
        }
      }
    }
  }

  function enableAllSkills() {
    const skillCardsDiv = document.getElementById("battle-skill-cards");
    if (!skillCardsDiv) return;
    const cards = skillCardsDiv.querySelectorAll(".pva-skill-card");
    cards.forEach((card) => {
      if (!card.classList.contains("disabled")) {
        card.style.pointerEvents = "auto";
        card.style.opacity = "1";
      }
    });
  }

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

  function handleBattleEvent(event) {
    if (event.text) {
      const cls = EVENT_TEXT_CLASS_MAP[event.type] || "log-other";
      addTextLog(event.text, cls);
    }
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
    }
    if (event.type === "refresh_status" && event.appendix) {
      updateStatusBars(event.appendix);
    }
  }

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
  function redGlowAndShake(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.classList.add("red-glow", "shake");
    setTimeout(() => {
      panel.classList.remove("red-glow", "shake");
    }, 500);
  }
  function greenGlow(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.classList.add("heal-effect");
    setTimeout(() => {
      panel.classList.remove("heal-effect");
    }, 800);
  }

  function updateStatusBars(appendix, forceUpdate = false) {
    latestAppendix = appendix;
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
    if (appendix.left) {
      const hp = parseInt(appendix.left.hp, 10);
      const maxHp = parseInt(appendix.left.max_hp, 10);
      const leftHpElem = document.getElementById("left-hp");
      const leftMaxHpElem = document.getElementById("left-max-hp");
      const leftHpFill = document.getElementById("left-hp-fill");
      if (leftHpElem) leftHpElem.textContent = hp;
      if (leftMaxHpElem) leftMaxHpElem.textContent = maxHp;
      if (leftHpFill) {
        let pct = maxHp > 0 ? (hp / maxHp) * 100 : 0;
        leftHpFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      }
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
      console.log(
        "[Debug] before enter build skill, appendix =",
        JSON.stringify(appendix)
      );
      if (appendix.left.skill_info && appendix.left.cooldowns) {
        console.log(
          "[Debug] updateStatusBars, appendix =",
          JSON.stringify(appendix)
        );
        const skillCards = document.getElementById("battle-skill-cards");
        skillCards.innerHTML = buildSkillCardsHTML(
          appendix.global.left_profession,
          appendix.left.skill_info,
          appendix.left.cooldowns
        );
      }
      if (appendix.left.cooldowns) {
        updateSkillCardsState(appendix, forceUpdate);
      }
      const leftEffects = document.getElementById("left-effects");
      if (leftEffects) {
        leftEffects.innerHTML = parseEffects(appendix.left.effects);
      }
      const leftSkills = document.getElementById("left-skills");
      if (leftSkills && appendix.left.cooldowns) {
        const prof = appendix.global.left_profession;
        leftSkills.innerHTML = buildSkillsHTMLMin(
          prof,
          appendix.left.cooldowns
        );
      }
      const lPanel = document.getElementById("left-panel");
      toggleBuffGlow(lPanel, appendix.left.effects);
    }
    if (appendix.right) {
      const hp = parseInt(appendix.right.hp, 10);
      const maxHp = parseInt(appendix.right.max_hp, 10);
      const rightHpElem = document.getElementById("right-hp");
      const rightMaxHpElem = document.getElementById("right-max-hp");
      const rightHpFill = document.getElementById("right-hp-fill");
      if (rightHpElem) rightHpElem.textContent = hp;
      if (rightMaxHpElem) rightMaxHpElem.textContent = maxHp;
      if (rightHpFill) {
        let pct = maxHp > 0 ? (hp / maxHp) * 100 : 0;
        rightHpFill.style.width = Math.max(0, Math.min(100, pct)) + "%";
      }
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
      const rightEffects = document.getElementById("right-effects");
      if (rightEffects) {
        rightEffects.innerHTML = parseEffects(appendix.right.effects);
      }
      const rightSkills = document.getElementById("right-skills");
      if (rightSkills && appendix.right.cooldowns) {
        const prof = appendix.global.right_profession;
        rightSkills.innerHTML = buildSkillsHTMLMin(
          prof,
          appendix.right.cooldowns
        );
      }
      const rPanel = document.getElementById("right-panel");
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
      const effData = EFFECT_DATA[effId] || {
        name: `效果ID:${effId}`,
        type: "other",
      };
      let effName = effData.name;
      if (effData.type === "track" && typeof effSpecial === "string") {
        effName = effSpecial;
      }
      let baseClass = getEffectColorClass(effData.type);
      if (effId === 1) {
        baseClass =
          effSpecial < 1 ? "badge-buff-attack-lower" : "badge-buff-attack";
      } else if (effId === 2) {
        baseClass =
          effSpecial < 1 ? "badge-buff-defense-lower" : "badge-buff-defense";
      } else if (effId === 3) {
        baseClass =
          effSpecial < 1 ? "badge-buff-heal-lower" : "badge-buff-heal";
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
  function buildSkillsHTMLMin(professionName, cooldowns) {
    let htmlStr = "";
    for (let i = 0; i < 4; i++) {
      const cdVal = cooldowns[i] !== undefined ? cooldowns[i] : 0;
      const skillUrl = `/static/images/${professionName}_skill_${i}.png`;
      htmlStr += createSkillIcon(skillUrl, cdVal, false, i);
    }
    return htmlStr;
  }

  // 建立技能卡（重點修改處）
  function buildSkillCardsHTML(profName, skillInfoArray, cooldownDict) {
    let html = "";
    for (let i = 0; i < skillInfoArray.length; i++) {
      const info = skillInfoArray[i];
      const cdVal = cooldownDict[i] !== undefined ? cooldownDict[i] : 0;
      const isDisabled = cdVal > 0;
      let cdSpan = "";
      // 只有當 info.cooldown 有設定且大於 0 且冷卻值大於 0 時，才顯示冷卻數值
      if (info.cooldown >0) {
        cdSpan = `<span class="pva-skill-cooldown"><b>${info.cooldown}</b></span>`;
      }
      // 如果 cdSpan 有值，則組合冷卻文字；否則設定為空字串
      const cdText = cdSpan ? `冷卻：${cdSpan} 回合` : "";

      const iconUrl = `/static/images/${profName}_skill_${i}.png`;
      const disabledClass = isDisabled ? "disabled" : "";
      // 根據 info.type 決定技能類型膠囊文字及顏色
      let typeText = "";
      let pillClass = "";
      if (info.type.toLowerCase() === "damage") {
        typeText = "傷害";
        pillClass = "pva-pill-damage";
      } else if (info.type.toLowerCase() === "heal") {
        typeText = "治癒";
        pillClass = "pva-pill-heal";
      } else {
        typeText = "效果";
        pillClass = "pva-pill-effect";
      }
      // 技能敘述經過格式化處理
      const formattedDescription = formatSkillDescription(info.description);
      html += `
      <div class="pva-skill-card ${disabledClass}" onclick="onSkillClick(${i})" data-skill-index="${i}">
        <div class="skill-card-top">
          <div class="skill-name">${info.name}</div>
          <div class="skill-meta">
            <span class="skill-type-pill ${pillClass}">${typeText}</span>
            ${cdText}
          </div>
        </div>
        <div class="skill-card-bottom">
          <div class="skill-image">
            <img src="${iconUrl}" alt="${info.name}" onerror="this.src='/static/images/skill_default.png';"/>
          </div>
          <div class="skill-description">
            ${formattedDescription}
          </div>
        </div>
      </div>
    `;
    }
    return html;
  }

  window.showPlayerVsAiForm = showPlayerVsAiForm;
  window.onSkillClick = onSkillClick;

  document.addEventListener("DOMContentLoaded", () => {
    const menuPva = document.getElementById("menu-player-vs-ai");
    if (menuPva) {
      menuPva.addEventListener("click", (e) => {
        e.preventDefault();
        showPlayerVsAiForm();
      });
    }
  });
})();
