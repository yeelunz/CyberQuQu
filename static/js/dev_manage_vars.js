// 檔案：static/js/dev_manage_vars.js

// 建立管理技能/職業變數頁面的內容（示範用）
function devManageVars_buildPage() {
    const container = document.createElement("div");
    container.className = "dev-vars-container";
  
    const h2 = document.createElement("h2");
    h2.textContent = "管理技能/職業變數";
    container.appendChild(h2);
  
    // 這裡建立一個示範表格（請依照實際需求調整）
    const table = document.createElement("table");
    table.className = "dev-vars-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>職業</th>
          <th>變數鍵</th>
          <th>值</th>
          <th>更新</th>
          <th>訊息</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Warrior</td>
          <td>hp</td>
          <td><input type="text" id="input-Warrior-hp" value="1000"></td>
          <td><button class="dev-update-btn" data-profession="Warrior" data-varkey="hp">更新</button></td>
          <td><span id="message-Warrior-hp"></span></td>
        </tr>
        <tr>
          <td>Mage</td>
          <td>mana</td>
          <td><input type="text" id="input-Mage-mana" value="500"></td>
          <td><button class="dev-update-btn" data-profession="Mage" data-varkey="mana">更新</button></td>
          <td><span id="message-Mage-mana"></span></td>
        </tr>
      </tbody>
    `;
    container.appendChild(table);
    return container;
  }
  
  // 為所有更新按鈕加上點擊事件監聽
  function initDevManageVars() {
    document.querySelectorAll(".dev-update-btn").forEach(button => {
      button.addEventListener("click", function() {
        const profession = this.dataset.profession;
        const varKey = this.dataset.varkey;
        const inputField = document.getElementById(`input-${profession}-${varKey}`);
        const newValue = inputField.value;
  
        // 發送 AJAX 請求更新後端變數（更新僅影響記憶體中資料）
        fetch("/api/update_var", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            profession: profession,
            var_key: varKey,
            new_value: newValue
          })
        })
        .then(response => response.json())
        .then(data => {
          const messageElem = document.getElementById(`message-${profession}-${varKey}`);
          if (data.status === "success") {
            messageElem.textContent = "更新成功";
            messageElem.style.color = "green";
          } else {
            messageElem.textContent = data.message || "更新失敗";
            messageElem.style.color = "red";
          }
        })
        .catch(err => {
          console.error(err);
          const messageElem = document.getElementById(`message-${profession}-${varKey}`);
          messageElem.textContent = "更新出錯";
          messageElem.style.color = "red";
        });
      });
    });
  }
  
  // 外部呼叫的入口函式，用於載入此頁面
  function loadManageVarsPage() {
    const contentArea = document.getElementById("content-area");
    contentArea.innerHTML = "";
    contentArea.appendChild(devManageVars_buildPage());
    // 綁定更新事件
    initDevManageVars();
  }
  
  // 將 loadManageVarsPage 暴露至全域（讓左側選單可直接呼叫）
  window.loadManageVarsPage = loadManageVarsPage;
  