/* /static/js/dev_manage_vars.js */
function initDevManageVars() {
    // 為所有更新按鈕加上點擊事件監聽
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

document.addEventListener("DOMContentLoaded", () => {
    initDevManageVars();
});
