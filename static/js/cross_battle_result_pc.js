// backend/static/js/cross_battle_result_pc.js

/**
 * 建立並插入交叉對戰數據的整個畫面
 * @param {Array} allCrossData 後端回傳的所有 cross_battle_result_pc 資料
 */
function renderCrossBattleResultPC(allCrossData) {
  // 1. 清空 content-area
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = "";

  // 2. 判斷是否有多筆資料，建立選擇資料的選單
  if (allCrossData.length > 1) {
    const dataSelection = document.createElement("div");
    dataSelection.classList.add("data-selection");

    const selectLabel = document.createElement("label");
    selectLabel.setAttribute("for", "data-select");
    selectLabel.textContent = "選擇要查看的資料：";

    const selectDropdown = document.createElement("select");
    selectDropdown.id = "data-select";

    // 添加選項
    allCrossData.forEach((dataItem, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${dataItem.name} (${dataItem.type})`;
      selectDropdown.appendChild(option);
    });

    dataSelection.appendChild(selectLabel);
    dataSelection.appendChild(selectDropdown);
    contentArea.appendChild(dataSelection);

    // 監聽選單變更事件
    selectDropdown.addEventListener("change", (e) => {
      const selectedIndex = parseInt(e.target.value, 10);
      displayData(selectedIndex);
    });

    // 初始顯示第一筆資料
    displayData(0);
  } else if (allCrossData.length === 1) {
    // 只有一筆資料，直接顯示
    displayData(0);
  } else {
    // 無資料
    const noDataMsg = document.createElement("p");
    noDataMsg.textContent = "無交叉對戰數據可顯示。";
    contentArea.appendChild(noDataMsg);
  }

  /**
   * 顯示特定索引的資料
   * @param {number} index 資料索引
   */
  function displayData(index) {
    const crossData = allCrossData[index];

    // 清除之前的內容（除了選單）
    const existingSections = contentArea.querySelectorAll(
      ".env-evaluation-section, .profession-evaluation-section, .combine-win-rate-section, .skill-usage-section, .data-info"
    );
    existingSections.forEach((section) => section.remove());

    // 在頁面頭部顯示基本資訊
    const infoSection = document.createElement("div");
    infoSection.classList.add("data-info");
    infoSection.innerHTML = `
        <h3>資料基本資訊</h3>
        <p><strong>Name:</strong> ${crossData.name}</p>
        <p><strong>Type:</strong> ${crossData.type}</p>
        <p><strong>ModelName:</strong> ${crossData.model}</p>
        <p><strong>Version:</strong> ${crossData.version}</p>
        <p><strong>Hash:</strong> ${crossData.hash}</p>
        <p><strong>總戰鬥場次:</strong> ${crossData.data.total_battles}</p>
      `;
    contentArea.appendChild(infoSection);

    // 渲染環境評估指標
    renderEnvEvaluation(
      crossData.data.env_evaluation,
      (container = contentArea)
    );

    // 渲染職業評估指標
    renderProfessionEvaluation(
      crossData.data.profession_evaluation,
      (container = contentArea)
    );

    // 渲染勝率對照表（Heat Map）
    renderCombineWinRateTable(
      crossData.data.combine_win_rate_table,
      (container = contentArea)
    );

    // 渲染技能使用頻率分析
    renderSkillUsageFrequency(
      crossData.data.profession_skill_used_freq,
      (container = contentArea)
    );

    // 為職業評估表格添加排序功能
    addTableSort();
  }
}

/**
 * 渲染環境評估指標
 * @param {*} env_evaluation 環境評估指標資料
 * @param {HTMLElement} container 容器元素
 */
function renderEnvEvaluation(env_evaluation, container) {
  const envSection = document.createElement("div");
  envSection.classList.add("env-evaluation-section");

  const envTitle = document.createElement("h2");
  envTitle.textContent = "環境整體評估 (Environment Evaluation)";
  envSection.appendChild(envTitle);

  const envTable = document.createElement("table");
  envTable.classList.add("cross-battle-table");
  envTable.innerHTML = `
      <thead>
        <tr>
          <th>指標</th>
          <th>數值</th>
          <th>更多資訊</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>EHI</td>
          <td>${env_evaluation.ehi.toFixed(3)}</td>
          <td>
            <span class="info-icon" data-info="ehi">ℹ️</span>
            <div class="popover-content" id="popover-ehi">
               <strong>EHI (Environment Health Index)</strong><br>
              評估整體遊戲環境健康度的指標，基於所有職業間勝率的平衡性來計算。值越接近1表示環境越健康平衡，越接近0表示環境越不平衡。
              <br><br>
              <strong>計算方式：</strong><br>
              EHI = 0.4 * 正規化熵值 + 0.4 * (1 - 正規化基尼係數) + 0.2 * 正規化剋制環數.
            </div>
          </td>
        </tr>
        <tr>
          <td>ESI</td>
          <td>${env_evaluation.esi.toFixed(3)}</td>
          <td>
            <span class="info-icon" data-info="esi">ℹ️</span>
            <div class="popover-content" id="popover-esi">

              <strong>ESI (Environment Stability Index)</strong><br>
              衡量環境穩定性的指標，考慮所有職業的 MSI 指標，反映整體遊戲環境的穩定程度。值越高表示環境越穩定。
              <br><br>
              <strong>計算方式：</strong><br>
              ESI = 1 / (1 + 平均指標變化)

            </div>
          </td>
        </tr>
        <tr>
          <td>MPI</td>
          <td>${env_evaluation.mpi.toFixed(3)}</td>
          <td>
            <span class="info-icon" data-info="mpi">ℹ️</span>
            <div class="popover-content" id="popover-mpi">
              <strong>MPI (Meta Polarization Index)</strong><br>
              評估遊戲環境極化程度的指標，基於所有職業的 PI20 指標。值越高表示環境中存在越多極端對戰組合。
              <br><br>
              <strong>計算方式：</strong><br>
              MPI = 平均中層職業的壓制程度 / 最大可能壓制程度
            </div>
          </td>
        </tr>
        <!-- 新增：平均回合數 -->
        <tr>
          <td>平均回合數</td>
          <td>${env_evaluation.avg_rounds.toFixed(2)}</td>
          <td>
            <span class="info-icon" data-info="avgRound">ℹ️</span>
            <div class="popover-content" id="popover-avgRound">
              <strong>平均回合數</strong><br>
              代表此次所有對戰的平均回合長度。
            </div>
          </td>
        </tr>
      </tbody>
    `;
  envSection.appendChild(envTable);
  container.appendChild(envSection);

  // 添加事件監聽器以顯示/隱藏 Popover
  envSection.querySelectorAll(".info-icon").forEach((icon) => {
    icon.addEventListener("mouseenter", (e) => {
      const infoKey = e.target.dataset.info;
      const popover = document.getElementById(`popover-${infoKey}`);
      popover.style.display = "block";
    });

    icon.addEventListener("mouseleave", (e) => {
      const infoKey = e.target.dataset.info;
      const popover = document.getElementById(`popover-${infoKey}`);
      popover.style.display = "none";
    });
  });
}

/**
 * 渲染職業評估指標
 * @param {*} profession_evaluation 職業評估指標資料
 * @param {HTMLElement} container 容器元素
 */
function renderProfessionEvaluation(profession_evaluation, container) {
  const profSection = document.createElement("div");
  profSection.classList.add("profession-evaluation-section");

  const profTitle = document.createElement("h2");
  profTitle.textContent = "職業評估指標 (Profession Evaluation)";
  profSection.appendChild(profTitle);

  // 建立表格
  const profTable = document.createElement("table");
  profTable.classList.add("cross-battle-table", "sortable");
  profTable.id = "profession-evaluation-table";
  profTable.innerHTML = `
    <thead>
      <tr>
        <th>職業</th>
        <th data-sort="EIR">EIR &#x25B2;&#x25BC;</th>
        <th data-sort="Advanced NAS">A-NAS &#x25B2;&#x25BC;</th>
        <th data-sort="MSI">MSI &#x25B2;&#x25BC;</th>
        <th data-sort="PI20">PI20 &#x25B2;&#x25BC;</th>
        <!-- 新增：職業平均勝率 -->
        <th data-sort="AVG_WR">平均勝率 &#x25B2;&#x25BC;</th>
      </tr>
    </thead>
    <tbody>
    </tbody>
  `;

  const profTbody = profTable.querySelector("tbody");

  // profession_evaluation[profName] = { EIR, Advanced NAS, MSI, PI20, AVG_WR }
  for (const [profName, data] of Object.entries(profession_evaluation)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><img src="/static/images/${encodeURIComponent(
        profName
      )}.png" alt="${profName}" class="profession-avatar" /> ${profName}</td>
      <td>${data["EIR"].toFixed(2)}</td>
      <td>${data["Advanced NAS"].toFixed(3)}</td>
      <td>${data["MSI"].toFixed(2)}</td>
      <td>${data["PI20"].toFixed(2)}</td>
      <!-- 新增平均勝率 -->
      <td>${data["AVG_WR"] ? data["AVG_WR"].toFixed(2) : "0.00"}%</td>
    `;
    profTbody.appendChild(tr);
  }

  profSection.appendChild(profTable);

  // 指標說明
  const profExplain = document.createElement("div");
  profExplain.classList.add("indicator-explanation");
  profExplain.innerHTML = `
      <h3>指標說明</h3>
      <ul>
        <li><strong>EIR (Effective Influence Rate):</strong> 衡量職業在整體環境中的實際影響力。<span class="info-icon" data-info="eir">ℹ️</span>
          <div class="popover-content" id="popover-eir">
            <strong>EIR (Effective Influence Rate)</strong><br>
            衡量職業在整體環境中的實際影響力。這是一個綜合指標，結合了職業的平均勝率和對環境的影響力。計算方式為：50% * 平均勝率 + 50% * 總影響力。分數範圍通常在0-100之間，越高代表該職業越強勢。
          </div>
        </li>
        <li><strong>A-NAS (Advanced Normalized Ability Score):</strong> 基於 BTL 模型的相對實力指標。<span class="info-icon" data-info="a_nas">ℹ️</span>
          <div class="popover-content" id="popover-a_nas">
            <strong>A-NAS (Advanced Normalized Ability Score)</strong><br>
            基於 Bradley-Terry-Luce (BTL) 模型計算的標準化能力分數。使用迭代方法計算每個職業的相對強度(theta值)，並將其標準化，使得平均值為0，標準差為1。正值表示強於平均水平，負值表示弱於平均水平，適合用來比較職業間的相對強弱關係。
          </div>
        </li>
        <li><strong>MSI (Match Stability Index):</strong> 衡量職業表現穩定性，數值越高表示越穩定。<span class="info-icon" data-info="msi">ℹ️</span>
          <div class="popover-content" id="popover-msi">
            <strong>MSI (Match Stability Index)</strong><br>
            衡量職業對戰穩定性的指標。計算方式為：1 - (勝率標準差 / 100)。值越接近1表示該職業在各種對戰中表現越穩定，值越接近0表示該職業在不同對戰中表現差異較大。
          </div>
        </li>
        <li><strong>PI20 (Performance Impact 20%):</strong> 與其他職業勝率差異超過 20% 的匹配比例。<span class="info-icon" data-info="pi20">ℹ️</span>
          <div class="popover-content" id="popover-pi20">
            <strong>PI20 (Performance Impact 20%)</strong><br>
            衡量職業對戰極化程度的指標。計算偏離平均勝率超過20%的對戰比例。值越高表示該職業的對戰結果越兩極化，有助於識別具有明顯優劣勢對戰的職業。
          </div>
        </li>
      </ul>
    `;
  profSection.appendChild(profExplain);
  container.appendChild(profSection);
  // 添加事件監聽器以顯示/隱藏 Popover
  profSection.querySelectorAll(".info-icon").forEach((icon) => {
    icon.addEventListener("mouseenter", (e) => {
      const infoKey = e.target.dataset.info;
      const popover = document.getElementById(`popover-${infoKey}`);
      popover.style.display = "block";
    });

    icon.addEventListener("mouseleave", (e) => {
      const infoKey = e.target.dataset.info;
      const popover = document.getElementById(`popover-${infoKey}`);
      popover.style.display = "none";
    });
  });

  container.appendChild(profSection);
}

/**
 * 渲染Combine Win Rate Table（Heat Map）
 * @param {*} combine_win_rate_table Combine Win Rate Table 資料
 * @param {HTMLElement} container 容器元素
 */
function renderCombineWinRateTable(combine_win_rate_table, container) {
  const combineSection = document.createElement("div");
  combineSection.classList.add("combine-win-rate-section");

  const combineTitle = document.createElement("h2");
  combineTitle.textContent = "勝率對照表 (Combine Win Rate Table)";
  combineSection.appendChild(combineTitle);

  // 取得所有職業名稱（假設所有職業都出現在外層鍵中）
  const professionSet = new Set();
  Object.keys(combine_win_rate_table).forEach((profA) => {
    professionSet.add(profA);
    Object.keys(combine_win_rate_table[profA]).forEach((profB) => {
      professionSet.add(profB);
    });
  });
  const professionList = Array.from(professionSet).sort();

  // 建立表格
  const combineTable = document.createElement("table");
  combineTable.classList.add("cross-battle-heatmap-table");

  // 建立表頭
  let combineThead = "<thead><tr><th>職業</th>";
  professionList.forEach((prof) => {
    combineThead += `<th><img src="/static/images/${encodeURIComponent(
      prof
    )}.png" alt="${prof}" class="profession-avatar-small" /><br>${prof}</th>`;
  });
  combineThead += "</tr></thead>";

  // 建立表身
  let combineTbody = "<tbody>";
  professionList.forEach((profA) => {
    combineTbody += `<tr><th><img src="/static/images/${encodeURIComponent(
      profA
    )}.png" alt="${profA}" class="profession-avatar-small" /><br>${profA}</th>`;
    professionList.forEach((profB) => {
      if (profA === profB) {
        combineTbody += `<td>—</td>`;
      } else {
        const matchData =
          combine_win_rate_table[profA] && combine_win_rate_table[profA][profB];
        if (matchData && typeof matchData.win_rate === "number") {
          const { draw, loss, win, win_rate } = matchData;
          const percentage = win_rate.toFixed(2) + "%";
          const bgColor = getHeatMapColor(win_rate / 100);
          combineTbody += `
              <td style="background-color: ${bgColor}; position: relative;">
                ${percentage}
                <div class="tooltip">
                  <strong>${profA} vs ${profB}</strong><br>
                  Win: ${win}<br>
                  Loss: ${loss}<br>
                  Draw: ${draw}<br>
                  Win Rate: ${win_rate.toFixed(2)}%
                </div>
              </td>
            `;
        } else {
          combineTbody += `<td>—</td>`;
        }
      }
    });
    combineTbody += "</tr>";
  });
  combineTbody += "</tbody>";

  combineTable.innerHTML = combineThead + combineTbody;
  combineSection.appendChild(combineTable);

  container.appendChild(combineSection);
}

/**
 * 渲染技能使用頻率分析
 * @param {*} skillusedFreq 技能使用頻率資料
 * @param {HTMLElement} container 容器元素
 */
function renderSkillUsageFrequency(skillusedFreq, container) {
  const skillSection = document.createElement("div");
  skillSection.classList.add("skill-usage-section");

  const skillTitle = document.createElement("h2");
  skillTitle.textContent = "技能使用頻率分析 (Skill Usage Frequency Analysis)";
  skillSection.appendChild(skillTitle);

  // 建立表格
  const skillTable = document.createElement("table");
  skillTable.classList.add("cross-battle-table");

  skillTable.innerHTML = `
      <thead>
        <tr>
          <th>職業</th>
          <th colspan="3">技能</th>
        </tr>
        <tr>
          <th></th>
          <th>技能 0</th>
          <th>技能 1</th>
          <th>技能 2</th>
        </tr>
      </thead>
      <tbody>
      </tbody>
    `;

  const skillTbody = skillTable.querySelector("tbody");

  for (const [profession, skills] of Object.entries(skillusedFreq)) {
    // 計算該職業的總技能使用次數
    const totalSkillUsage = Object.values(skills).reduce(
      (sum, freq) => sum + freq,
      0
    );

    const tr = document.createElement("tr");
    tr.innerHTML = `
        <td><img src="/static/images/${encodeURIComponent(
          profession
        )}.png" alt="${profession}" class="profession-avatar" /> ${profession}</td>
        <td>
          <img src="/static/images/${encodeURIComponent(
            profession
          )}_skill_0.png" alt="Skill 0" class="skill-avatar-small" /><br>
          次數: ${skills[0] || 0} (${(
      ((skills[0] || 0) / totalSkillUsage) *
      100
    ).toFixed(2)}%)
        </td>
        <td>
          <img src="/static/images/${encodeURIComponent(
            profession
          )}_skill_1.png" alt="Skill 1" class="skill-avatar-small" /><br>
          次數: ${skills[1] || 0} (${(
      ((skills[1] || 0) / totalSkillUsage) *
      100
    ).toFixed(2)}%)
        </td>
        <td>
          <img src="/static/images/${encodeURIComponent(
            profession
          )}_skill_2.png" alt="Skill 2" class="skill-avatar-small" /><br>
          次數: ${skills[2] || 0} (${(
      ((skills[2] || 0) / totalSkillUsage) *
      100
    ).toFixed(2)}%)
        </td>
      `;
    skillTbody.appendChild(tr);
  }

  skillSection.appendChild(skillTable);
  container.appendChild(skillSection);
}

/**
 * 根據勝率值回傳對應的 Heat Map 背景色
 * 勝率 > 50%：從深綠到更深綠
 * 勝率 < 50%：從深紅到更深紅
 * 勝率 = 50%：中性灰
 * @param {number} winRate - 勝率值（0到1之間）
 * @returns {string} - RGB 顏色碼
 */
function getHeatMapColor(winRate) {
  if (winRate > 0.5) {
    // 深綠系列
    const intensity = Math.min((winRate - 0.5) * 2, 1); // 0到1
    const green = Math.floor(150 + 105 * intensity); // 150到255
    return `rgb(50, ${green}, 50)`; // 從深綠到更深綠
  } else if (winRate < 0.5) {
    // 深紅系列
    const intensity = Math.min((0.5 - winRate) * 2, 1); // 0到1
    const red = Math.floor(150 + 105 * intensity); // 150到255
    return `rgb(${red}, 50, 50)`; // 從深紅到更深紅
  } else {
    // 中性灰
    return `rgb(200, 200, 200)`;
  }
}

/**
 * 為表格添加排序功能
 */
function addTableSort() {
  const tables = document.querySelectorAll(".sortable");
  tables.forEach((table) => {
    const headers = table.querySelectorAll("th[data-sort]");
    let sortDirection = {}; // 保存每個列的排序方向

    headers.forEach((header) => {
      sortDirection[header.dataset.sort] = true; // true 為升序，false 為降序
      header.style.cursor = "pointer";
      header.addEventListener("click", () => {
        const sortKey = header.dataset.sort;
        const tbody = table.querySelector("tbody");
        const rows = Array.from(tbody.querySelectorAll("tr"));

        const columnIndex = getColumnIndex(table, sortKey);

        if (columnIndex === -1) return;

        rows.sort((a, b) => {
          const aValue = parseFloat(
            a.querySelector(`td:nth-child(${columnIndex})`).textContent
          );
          const bValue = parseFloat(
            b.querySelector(`td:nth-child(${columnIndex})`).textContent
          );
          if (sortDirection[sortKey]) {
            return aValue - bValue;
          } else {
            return bValue - aValue;
          }
        });

        // 重新排列行
        rows.forEach((row) => tbody.appendChild(row));

        // 切換排序方向
        sortDirection[sortKey] = !sortDirection[sortKey];
      });
    });
  });
}

/**
 * 根據排序鍵取得對應的列索引
 * @param {HTMLElement} table - 表格元素
 * @param {string} sortKey - 排序鍵（EIR、Advanced NAS、MSI、PI20）
 * @returns {number} - 列索引（從1開始）
 */
function getColumnIndex(table, sortKey) {
  const headers = table.querySelectorAll("th[data-sort]");
  for (let i = 0; i < headers.length; i++) {
    if (headers[i].dataset.sort === sortKey) {
      return i + 2; // 第一列是職業名稱
    }
  }
  return -1;
}

// ----------------------------------------------------------------------
// 監聽「電腦對戰：顯示交叉對戰數據」的選單點擊事件，從 API 取得資料
// ----------------------------------------------------------------------

// 確保 DOM 讀取後才綁定事件
document.addEventListener("DOMContentLoaded", () => {
  const menuCrossBattlePC = document.getElementById(
    "menu-cross-battle-result-pc"
  );

  menuCrossBattlePC.addEventListener("click", (e) => {
    e.preventDefault();
    const contentArea = document.getElementById("content-area");

    // 顯示載入中的轉動圖標
    const loadingSpinner = document.createElement("div");
    loadingSpinner.classList.add("loading-spinner");
    contentArea.innerHTML = ""; // 清空之前的內容
    contentArea.appendChild(loadingSpinner);

    fetch("/api/version_test")
      .then((res) => res.json())
      .then((data) => {
        // 移除載入中的轉動圖標
        loadingSpinner.remove();

        // 根據您提供的資料結構，應從 data.message 取得所有資料
        if (data.message) {
          let allCrossData = [];

          if (Array.isArray(data.message)) {
            allCrossData = data.message;
          } else if (typeof data.message === "object") {
            allCrossData = [data.message];
          }

          if (allCrossData.length > 0) {
            console.log("[Cross Battle Data]", allCrossData);
            renderCrossBattleResultPC(allCrossData);
          } else {
            console.error("無有效的交叉對戰數據:", data);
            contentArea.innerHTML = "<p>無法取得交叉對戰數據，請稍後再試。</p>";
          }
        } else {
          console.error("資料結構不符合預期:", data);
          contentArea.innerHTML = "<p>無法取得交叉對戰數據，請稍後再試。</p>";
        }
      })
      .catch((err) => {
        // 移除載入中的轉動圖標
        loadingSpinner.remove();

        console.error("取得 cross_battle_result_pc 錯誤:", err);
        contentArea.innerHTML = "<p>發生錯誤，無法載入交叉對戰數據。</p>";
      });
  });
});
