// /static/js/embedding_visualization.js

// 全域變數 (紀錄具有 embedding 的模型清單)
let embeddingModels = [];

/**
 * 初始化「嵌入向量可視化」頁面
 */
function embedding_vis_init() {
  // 更新 content-area 內容
  const contentArea = document.getElementById("content-area");
  contentArea.innerHTML = `
    <div class="prefix-embedding-container">
      <h2 class="prefix-embedding-title">嵌入向量可視化</h2>
      <div class="prefix-embedding-controls">
        <label>選擇模型：</label>
        <select id="embedding-model-select" class="prefix-embedding-select"></select>

        <label class="prefix-embedding-radio-label">
          <input type="radio" name="dimOption" value="2" checked> 2D
        </label>
        <label class="prefix-embedding-radio-label">
          <input type="radio" name="dimOption" value="3"> 3D
        </label>
        <br>
        <!-- 新增分別控制顯示職業、技能、效果的勾選框 -->
        <label>
          <input type="checkbox" id="show-profession" checked> 職業
        </label>
        <label>
          <input type="checkbox" id="show-skill" checked> 技能
        </label>
        <label>
          <input type="checkbox" id="show-effect" checked> 效果
        </label>

        <button id="load-embedding-btn" class="prefix-embedding-button">載入並顯示</button>
      </div>

      <!-- Plotly 圖表容器 -->
      <div id="plotly-embedding-graph" class="prefix-embedding-graph"></div>
    </div>
  `;

  // 取得模型清單
  fetchEmbeddingModels()
    .then(() => {
      const selectEl = document.getElementById("embedding-model-select");
      embeddingModels.forEach(m => {
        // 只加入有 embedding 資料的模型
        if (m.meta.has_embedding) {
          const option = document.createElement("option");
          option.value = m.folder_name;
          option.textContent = m.folder_name;
          selectEl.appendChild(option);
        }
      });
      // 預設選擇第一個模型
      if (selectEl.options.length > 0) {
        selectEl.selectedIndex = 0;
      }
    })
    .catch(err => {
      console.error("取得模型列表發生錯誤:", err);
      alert("無法取得模型列表，請查看主控台錯誤訊息");
    });

  // 綁定「載入並顯示」按鈕事件
  const loadBtn = document.getElementById("load-embedding-btn");
  loadBtn.addEventListener("click", () => {
    const modelName = document.getElementById("embedding-model-select").value;
    const dimValue = document.querySelector('input[name="dimOption"]:checked').value;
    loadAndDisplayEmbedding(modelName, dimValue);
  });
}

/**
 * 取得具有 embedding 的模型清單（呼叫 /api/embedding/list_models）
 */
async function fetchEmbeddingModels() {
  const resp = await fetch("/api/embedding/list_models");
  if (!resp.ok) {
    throw new Error("後端回傳錯誤");
  }
  const models = await resp.json(); // 格式：[{folder_name, meta}, ...]
  embeddingModels = models;
}

/**
 * 請求並顯示指定模型的 embedding
 * @param {string} modelName 模型資料夾名稱
 * @param {string} dimValue "2" 或 "3"
 */
async function loadAndDisplayEmbedding(modelName, dimValue) {
  if (!modelName) {
    alert("請先選擇模型");
    return;
  }

  try {
    const url = `/api/embedding/get?model=${encodeURIComponent(modelName)}&dim=${dimValue}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error("取得 embedding 失敗");
    }
    const data = await resp.json();
    // data.categories 為陣列，每個物件格式：
    // {
    //   name: "profession_p", // 或其它類別名稱
    //   x: [ ... ],
    //   y: [ ... ],
    //   z: [ ... ] // 若 dim==3 則有此屬性
    //   labels: [ ... ] // 每個點的中文標籤
    // }

    // 根據勾選框決定哪些類別要顯示
    const showProfession = document.getElementById("show-profession").checked;
    const showSkill = document.getElementById("show-skill").checked;
    const showEffect = document.getElementById("show-effect").checked;

    const filteredCategories = data.categories.filter(cat => {
      if (cat.name.includes("profession")) {
        return showProfession;
      } else if (cat.name.includes("skill")) {
        return showSkill;
      } else if (cat.name.includes("effect")) {
        return showEffect;
      }
      return true;
    });

    // 依據維度建立 traces（加入 text 標籤）
    const traces = filteredCategories.map(cat => {
      if (dimValue === "2") {
        return {
          x: cat.x,
          y: cat.y,
          mode: "markers+text",
          type: "scatter",
          name: cat.name,
          marker: { size: 6 },
          text: cat.labels,            // 顯示中文標籤
          textposition: "top center"
        };
      } else {
        return {
          x: cat.x,
          y: cat.y,
          z: cat.z,
          mode: "markers+text",
          type: "scatter3d",
          name: cat.name,
          marker: { size: 3 },
          text: cat.labels,
          textposition: "top center"
        };
      }
    });

    const layout = {
      title: `模型：${modelName} (維度：${dimValue}D)`,
      showlegend: true
    };

    Plotly.newPlot("plotly-embedding-graph", traces, layout);
  } catch (err) {
    console.error(err);
    alert("載入 embedding 時發生錯誤：" + err.message);
  }
}
