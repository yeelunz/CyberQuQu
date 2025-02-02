// 檔案：static/js/model_vs_model_result.js
// 前綴：mvmr_ (model vs model result)

/**
 * 初始化頁面：建立UI(下拉)並綁定事件
 */
function mvmr_init() {
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = "";
  
    const container = document.createElement("div");
    container.className = "mvmr-container";
    contentArea.appendChild(container);
  
    // 標題
    const h2 = document.createElement("h2");
    h2.textContent = "模型間對戰資料檢視";
    container.appendChild(h2);
  
    // 下拉 & 按鈕
    const formGroup = document.createElement("div");
    formGroup.className = "mvmr-form-group";
    container.appendChild(formGroup);
  
    const label = document.createElement("label");
    label.className = "mvmr-label";
    label.setAttribute("for", "mvmr-result-select");
    label.textContent = "選擇要查看的對戰資料：";
    formGroup.appendChild(label);
  
    const select = document.createElement("select");
    select.id = "mvmr-result-select";
    select.className = "mvmr-input";
    formGroup.appendChild(select);
  
    const loadBtn = document.createElement("button");
    loadBtn.id = "mvmr-load-btn";
    loadBtn.className = "mvmr-btn";
    loadBtn.textContent = "載入結果";
    container.appendChild(loadBtn);
  
    // 結果區
    const resultArea = document.createElement("div");
    resultArea.id = "mvmr-result-area";
    container.appendChild(resultArea);
  
    // 取得清單
    fetch("/api/list_model_vs_results")
      .then(res => res.json())
      .then(data => {
        if (data.results && data.results.length > 0) {
          data.results.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r.id;
            opt.textContent = `${r.model_A} vs ${r.model_B} (${r.timestamp})`;
            select.appendChild(opt);
          });
        } else {
          select.innerHTML = "<option>沒有可用的對戰資料</option>";
        }
      })
      .catch(err => {
        console.error("取得對戰資料清單失敗:", err);
        select.innerHTML = "<option>取得資料清單失敗</option>";
      });
  
    // 綁定載入
    loadBtn.addEventListener("click", () => {
      const val = select.value;
      if (!val) {
        alert("請選擇要載入的對戰資料");
        return;
      }
      fetch(`/api/model_vs_model_result_json?result_id=${encodeURIComponent(val)}`)
        .then(res => res.json())
        .then(data => {
          displayResult(data);
        })
        .catch(err => {
          console.error("取得資料錯誤:", err);
          resultArea.innerHTML = "<p class='mvmr-error'>取得資料失敗</p>";
        });
    });
  }
  
  /**
   * 主顯示函式
   */
  function displayResult(data) {
    const resultArea = document.getElementById("mvmr-result-area");
    resultArea.innerHTML = "";
  
    // 先建一個反轉的 cross 給 B視角使用
    const reversedCross = buildReversedCrossResults(data.cross_results);
  
    // 1. 基本摘要 + 整體勝率
    const summary = document.createElement("div");
    summary.className = "mvmr-summary";
    summary.innerHTML = `
      <h3>基本摘要</h3>
      <p><strong>模型 A:</strong> ${data.model_A}</p>
      <p><strong>模型 B:</strong> ${data.model_B}</p>
      <p><strong>對戰時間:</strong> ${data.timestamp}</p>
      <p><strong>總場次:</strong> ${data.total_battles}</p>
    `;
    resultArea.appendChild(summary);
  
    // 計算「A對B勝率」(用原始 cross_results 來累計)
    const overallA = calcOverallWinRateForA(data.cross_results);
    // 計算「B對A勝率」(用 reversedCross 來累計 => B當left => B的win_modelA)
    const overallB = calcOverallWinRateForA(reversedCross);
  
    const overallDiv = document.createElement("div");
    overallDiv.className = "mvmr-analysis-block";
    overallDiv.innerHTML = `
      <p>整體 ${data.model_A} 對 ${data.model_B} 勝率：約 <strong>${overallA.toFixed(1)}%</strong></p>
      <p>整體 ${data.model_B} 對 ${data.model_A} 勝率：約 <strong>${overallB.toFixed(1)}%</strong></p>
    `;
    resultArea.appendChild(overallDiv);
  
    // 2. 同職業分析 (intra_results)
    const intraBlock = document.createElement("div");
    intraBlock.className = "mvmr-block";
    intraBlock.innerHTML = `<h3>同職業對戰 (Intra) 分析</h3>`;
    resultArea.appendChild(intraBlock);
  
    // 同職業表
    const tableIntra = buildIntraTable(data.intra_results, data.model_A, data.model_B);
    intraBlock.appendChild(tableIntra);
  
    // 分析文字
    const analysisIntra = analyzeIntraResults(data.intra_results, data.model_A, data.model_B);
    intraBlock.appendChild(analysisIntra);
  
    // 3. 從 A模型 的視角 (跨職業)
    const crossBlockA = document.createElement("div");
    crossBlockA.className = "mvmr-block";
    crossBlockA.innerHTML = `<h3>從 ${data.model_A} 的視角</h3>`;
    resultArea.appendChild(crossBlockA);
  
    // 使用原始 cross (left => A職業, right => B職業)
    const tableA = buildModelPerspectiveTable(data.cross_results, "A");
    crossBlockA.appendChild(tableA);
  
    // 4. 從 B模型 的視角 (跨職業)
    const crossBlockB = document.createElement("div");
    crossBlockB.className = "mvmr-block";
    crossBlockB.innerHTML = `<h3>從 ${data.model_B} 的視角</h3>`;
    resultArea.appendChild(crossBlockB);
  
    // 使用 reversedCross (left => B職業, right => A職業)
    const tableB = buildModelPerspectiveTable(reversedCross, "B");
    crossBlockB.appendChild(tableB);
  
    // 5. 特殊交叉(互克或矛盾對戰)
    const advancedBlock = document.createElement("div");
    advancedBlock.className = "mvmr-block";
    advancedBlock.innerHTML = `<h3>特殊交叉對戰觀察</h3>`;
    resultArea.appendChild(advancedBlock);
  
    const ul = document.createElement("ul");
    advancedBlock.appendChild(ul);
    const interesting = findInterestingCrosses(data.cross_results);
    if (interesting.length === 0) {
      const li = document.createElement("li");
      li.textContent = "暫未發現明顯互克或矛盾現象。";
      ul.appendChild(li);
    } else {
      interesting.forEach(msg => {
        const li = document.createElement("li");
        li.textContent = msg;
        ul.appendChild(li);
      });
    }
  }
  
  /**
   * 建立反轉的 cross_results (讓B當左職業, A當右職業)
   * 如果原始是 "X vs Y": { win_modelA, win_modelB, draw }
   *   => 反轉: "Y vs X": { win_modelA: 原win_modelB, win_modelB: 原win_modelA, draw }
   *   (因為在新視角裡，左方模型=原先的右方模型)
   */
  function buildReversedCrossResults(originalCross) {
    const reversed = {};
    for (const k in originalCross) {
      const [p1, p2] = k.split(" vs ");
      const { win_modelA, win_modelB, draw } = originalCross[k];
      const newKey = `${p2} vs ${p1}`;
      reversed[newKey] = {
        win_modelA: win_modelB, // 交換
        win_modelB: win_modelA,
        draw: draw
      };
    }
    return reversed;
  }
  
  /**
   * 計算「A對B的整體勝率」（全部 cross 的 totalWins / totalMatches）
   * 請注意：這裡只需要把 cross 當成「A的職業 vs B的職業」的資料
   *         win_modelA 就是 A模型的勝場
   */
  function calcOverallWinRateForA(cross) {
    let totalWins = 0;
    let totalMatches = 0;
    for (const k in cross) {
      const rec = cross[k];
      const matches = rec.win_modelA + rec.win_modelB + rec.draw;
      totalWins += rec.win_modelA;
      totalMatches += matches;
    }
    if (totalMatches === 0) return 0;
    return (totalWins / totalMatches) * 100;
  }
  
  /**
   * 建立同職業對戰 (intra_results) 表格
   */
  function buildIntraTable(intra, modelA, modelB) {
    const table = document.createElement("table");
    table.className = "mvmr-table mvmr-table-intra";
  
    const thead = document.createElement("thead");
    thead.innerHTML = `
      <tr>
        <th>職業</th>
        <th>${modelA} 勝</th>
        <th>${modelB} 勝</th>
        <th>平手</th>
        <th>${modelA} 勝率</th>
        <th>差值 (A-B)</th>
      </tr>
    `;
    table.appendChild(thead);
  
    const tbody = document.createElement("tbody");
    table.appendChild(tbody);
  
    for (const profession in intra) {
      const { win_modelA, win_modelB, draw } = intra[profession];
      const matches = win_modelA + win_modelB + draw;
      const aRate = matches === 0 ? 0 : (win_modelA / matches) * 100;
      const diff = win_modelA - win_modelB;
  
      let diffClass = "";
      if (diff > 0) diffClass = "mvmr-pos-diff";
      else if (diff < 0) diffClass = "mvmr-neg-diff";
  
      // 頭像
      const iconPath = `/static/images/${profession}.png`;
  
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mvmr-prof-td">
          <img src="${iconPath}" onerror="this.style.display='none'" class="mvmr-prof-icon" />
          <span>${profession}</span>
        </td>
        <td>${win_modelA}</td>
        <td>${win_modelB}</td>
        <td>${draw}</td>
        <td>${aRate.toFixed(1)}%</td>
        <td class="${diffClass}">${diff}</td>
      `;
      tbody.appendChild(tr);
    }
  
    return table;
  }
  
  /**
   * 分析同職業對戰
   */
  function analyzeIntraResults(intra, modelA, modelB) {
    const div = document.createElement("div");
    div.className = "mvmr-analysis-block";
  
    let sumDiff = 0;
    let bestA = { prof: null, diff: -999999 };
    let bestB = { prof: null, diff: 999999 };
  
    for (const prof in intra) {
      const { win_modelA, win_modelB } = intra[prof];
      const diff = win_modelA - win_modelB;
      sumDiff += diff;
      if (diff > bestA.diff) {
        bestA.diff = diff;
        bestA.prof = prof;
      }
      if (diff < bestB.diff) {
        bestB.diff = diff;
        bestB.prof = prof;
      }
    }
  
    let overallText = "";
    if (sumDiff > 0) {
      overallText = `${modelA} 在同職業對戰中累計優勢 +${sumDiff}`;
    } else if (sumDiff < 0) {
      overallText = `${modelB} 在同職業對戰中累計優勢 ${sumDiff}`;
    } else {
      overallText = `兩模型在同職業對戰中平分秋色`;
    }
  
    div.innerHTML = `
      <p>${overallText}</p>
      <p>對 ${modelA} 而言，最強職業為 <strong>${bestA.prof}</strong> (差值 +${bestA.diff}).</p>
      <p>對 ${modelB} 而言，最強職業為 <strong>${bestB.prof}</strong> (差值 ${bestB.diff}).</p>
    `;
    return div;
  }
  
  /**
   * 建立「模型視角」的職業分析表：
   *   keyPattern = "A" 表示原始 cross (left是A職業, right是B職業)
   *   keyPattern = "B" 表示已反轉 cross (left是B職業, right是A職業)
   * 以 "leftProf" 作為「當前模型」的職業。
   * 逐一列出該模型每個職業的：總對戰、勝場、平均勝率、最佳對手職業、最差對手職業、是否訓練良好
   */
  function buildModelPerspectiveTable(cross, keyPattern) {
    const table = document.createElement("table");
    table.className = "mvmr-table mvmr-table-persp";
  
    const thead = document.createElement("thead");
    thead.innerHTML = `
      <tr>
        <th>職業</th>
        <th>總對戰</th>
        <th>總勝場</th>
        <th>平均勝率</th>
        <th>最佳對戰</th>
        <th>最差對戰</th>
        <th>訓練良好?</th>
      </tr>
    `;
    table.appendChild(thead);
  
    // 先找出所有 "leftProf"
    const leftProfSet = new Set();
    for (const k in cross) {
      const [leftProf] = k.split(" vs ");
      leftProfSet.add(leftProf);
    }
    const leftProfs = Array.from(leftProfSet).sort();
  
    const tbody = document.createElement("tbody");
    table.appendChild(tbody);
  
    leftProfs.forEach(prof => {
      const stat = calcModelProfessionStats(cross, prof);
      const { totalMatches, totalWins, avgWinRate, bestMatch, worstMatch } = stat;
  
      // 頭像
      const iconPath = `/static/images/${prof}.png`;
      // 門檻可自訂 (目前預設 > 50% 算訓練良好)
      const wellTrained = avgWinRate > 50;
  
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mvmr-prof-td">
          <img src="${iconPath}" onerror="this.style.display='none'" class="mvmr-prof-icon" />
          <span>${prof}</span>
        </td>
        <td>${totalMatches}</td>
        <td>${totalWins}</td>
        <td>${avgWinRate.toFixed(1)}%</td>
        <td>${bestMatch ? bestMatch.prof + ` (${bestMatch.rate.toFixed(1)}%)` : "-"}</td>
        <td>${worstMatch ? worstMatch.prof + ` (${worstMatch.rate.toFixed(1)}%)` : "-"}</td>
        <td class="${wellTrained ? "mvmr-good-trained" : "mvmr-bad-trained"}">
          ${wellTrained ? "是" : "否"}
        </td>
      `;
      tbody.appendChild(tr);
    });
  
    return table;
  }
  
  /**
   * 統計某"leftProf"在 cross 裡對上所有"rightProf"的總勝/總場/平均勝率/最佳對手/最差對手
   * 注意：在 cross 中, "leftProf" 對應 win_modelA, "rightProf" 對應 win_modelB
   *       不需要分 "A/B" 參數，因為這個 cross 可能已經被反轉好
   */
  function calcModelProfessionStats(cross, leftProf) {
    let totalMatches = 0;
    let totalWins = 0;  // 對應 cross 中的 win_modelA
    let bestMatch = null;
    let worstMatch = null;
  
    for (const k in cross) {
      const [lp, rp] = k.split(" vs ");
      if (lp === leftProf) {
        const rec = cross[k];
        const matches = rec.win_modelA + rec.win_modelB + rec.draw;
        totalMatches += matches;
        totalWins += rec.win_modelA;
  
        const rate = (matches === 0) ? 0 : (rec.win_modelA / matches) * 100;
        // best
        if (!bestMatch || rate > bestMatch.rate) {
          bestMatch = { prof: rp, rate };
        }
        // worst
        if (!worstMatch || rate < worstMatch.rate) {
          worstMatch = { prof: rp, rate };
        }
      }
    }
  
    const avgWinRate = (totalMatches === 0) ? 0 : (totalWins / totalMatches) * 100;
    return { totalMatches, totalWins, avgWinRate, bestMatch, worstMatch };
  }
  
  /**
   * 尋找互克或特殊對戰: 
   *   當 A.X vs B.Y diff 很大(正) 同時 B.Y vs A.X diff 也很大(正)
   *   (表示A贏很多, B也贏很多，雙方都說自己贏很多 => 可能是AI上有矛盾或資料量少?)
   */
  function findInterestingCrosses(cross) {
    const list = [];
  
    function getDiff(rec) {
      return rec.win_modelA - rec.win_modelB;
    }
  
    // 簡單做法：對 cross 中每組 "p1 vs p2" 找對應 "p2 vs p1"。
    for (const k in cross) {
      const [p1, p2] = k.split(" vs ");
      const recA = cross[k];
      const recB = cross[`${p2} vs ${p1}`];
      if (!recB) continue; // 沒對應反向就跳過
      const diffA = getDiff(recA); // A.X vs B.Y
      const diffB = getDiff(recB); // B.Y vs A.X
      if (diffA >= 3 && diffB >= 3) {
        // 門檻可自訂，這裡 diff>=3 視為「大勝很多」
        list.push(
          `${p1} vs ${p2} 同時互克：` +
          `(${p1}贏${diffA}分, ${p2}也贏${diffB}分)`
        );
      }
    }
  
    return list;
  }
  
  // 將 mvmr_init 暴露全域
  window.mvmr_init = mvmr_init;
  