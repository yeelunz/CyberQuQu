// backend/static/script.js

document.addEventListener("DOMContentLoaded", () => {
    const menuTrain = document.getElementById("menu-train");
    const menuElo = document.getElementById("menu-elo");
    const menuBattle = document.getElementById("menu-battle");
    const menuVersion = document.getElementById("menu-version");
    const menuTestRandom = document.getElementById("menu-test-random");
    const menuInfo = document.getElementById("menu-info");
    

    const contentArea = document.getElementById("content-area");

    // 定義技能類型的中文對應
    const skillTypeMap = {
        'damage': '傷害',
        'heal': '治療',
        'effect': '效果',
        'buff': '增益',
        'debuff': '減益'
        // 如果有更多類型，繼續添加
    };

    // 建立一個格式化描述的函式
    function formatSkillDescription(rawText) {
        let text = rawText;

        // 傷害數值（紅色） - 改為非貪婪匹配
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

        text = text.replace(/對攻擊者立即造成其本次攻擊傷害的 (\d+) /g, (match, p1) => {
            return `對攻擊者立即造成其本次攻擊傷害的 <span class="damage-text">${p1}</span> `;
        });


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
        text = text.replace(/(提升|降低|降低其|增加|提升自身|增加自身|降低自身) (\d+)%/g, (match, p1, p2) => {
            return `${p1} <span class="effect-text">${p2}%</span>`;
        });

        text = text.replace(/(提升|降低|降低其|增加|提升自身|增加自身|降低自身) (\d+)%/g, (match, p1, p2) => {
            return `${p1} <span class="effect-text">${p2}%</span>`;
        });

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
            let colorClass = '';
            switch(match) {
                case '冰凍':
                    colorClass = 'frozen-text';
                    break;
                case '中毒':
                    colorClass = 'poisoned-text';
                    break;
                case '流血':
                    colorClass = 'bleeding-text';
                    break;
                case '燃燒':
                    colorClass = 'burning-text';
                    break;
                case '麻痺':
                    colorClass = 'paralyzed-text';
                    break;
                default:
                    colorClass = 'status-text';
            }
            return `<span class="${colorClass}">${match}</span>`;
        });

        return text;
    }

    // 點選 "各職業介紹" => GET /api/show_professions
    menuInfo.addEventListener("click", (e) => {
        e.preventDefault();
        fetch("/api/show_professions")
            .then((res) => res.json())
            .then((data) => {
                // data.professions_info 是後端回傳的陣列
                const professions = data.professions_info;
                let html = "";

                professions.forEach((p) => {
                    // 生成角色頭像的 URL
                    const avatarExtensions = ['.png', '.jpg', '.jpeg'];
                    let avatarUrl = '';
                    for (let ext of avatarExtensions) {
                        const potentialPath = `/static/images/${p.name.toLowerCase()}${ext}`;
                        // 假設後端有檢查該檔案是否存在
                        avatarUrl = potentialPath;
                        break;
                    }

                    // 生成被動技能圖片的 URL
                    const passiveExtensions = ['.png', '.jpg', '.jpeg'];
                    let passiveImageUrl = '';
                    for (let ext of passiveExtensions) {
                        const potentialPassivePath = `/static/images/${p.name.toLowerCase()}_passive${ext}`;
                        passiveImageUrl = potentialPassivePath;
                        break;
                    }

                    html += 
                        `<div class="profession-card">
                            <img src="${avatarUrl}" alt="${p.name}" class="profession-image" onerror="handleImageError(this, 'avatar')">
                            <div class="profession-details">
                                <h2 class="profession-name">${p.name} <span class="hp">HP: ${p.hp}</span></h2>
                                <p><strong>攻擊係數:</strong> <span class="attack">${p.attack_coeff}</span> 
                                <strong>防禦係數:</strong> <span class="defense">${p.defense_coeff}</span></p>
                                <div class="passive">
                                    <h3>被動技能: ${p.passive.name}</h3>
                                    <div class="passive-container">
                                        <img src="${passiveImageUrl}" alt="${p.passive.name}" class="skill-image" onerror="handleImageError(this, 'skill')">
                                        <p>${formatSkillDescription(p.passive.description)}</p>
                                    </div>
                                </div>
                                <div class="skills">
                                    <h3>技能</h3>
                                    <ul>
                                        ${p.skills.map((skill, index) => {
                                            // 生成技能圖片的 URL
                                            const skillImageExtensions = ['.png', '.jpg', '.jpeg'];
                                            let skillImageUrl = '';
                                            for (let ext of skillImageExtensions) {
                                                const formattedSkillName = skill.name.toLowerCase().replace(/\s+/g, '_');
                                                const potentialSkillPath = `/static/images/${p.name.toLowerCase()}_skill_${index}${ext}`;
                                                skillImageUrl = potentialSkillPath;
                                                break;
                                            }

                                            // 確保 skill.type 存在於 skillTypeMap，否則使用 'unknown'
                                            const skillTypeKey = skill.type ? skill.type.toLowerCase() : 'unknown';
                                            const skillTypeChinese = skillTypeMap[skillTypeKey] || '未知';

                                            // 使用我們的函式把描述裡面的數值染色
                                            const formattedDesc = formatSkillDescription(skill.description);

                                            // 如果有 cooldown，就在描述下方加一行「冷卻時間: X 回合」
                                            const cooldownText = skill.cooldown 
                                                ? `<span class="skill-cooldown">冷卻時間: ${skill.cooldown} 回合</span>`
                                                : "";

                                            return `
                                                <li>
                                                    <img src="${skillImageUrl}" alt="${skill.name}" class="skill-image" onerror="handleImageError(this, 'skill')">
                                                    <div>
                                                        <!-- 技能名稱 & 技能類型 -->
                                                        <span class="skill-name">
                                                            ${skill.name} 
                                                            <span class="skill-type ${skillTypeKey}">${skillTypeChinese}</span>
                                                        </span>
                                                        <!-- 技能描述 (含著色的數值與異常狀態) -->
                                                        <p>${formattedDesc}</p>
                                                        ${skill.cooldown ? `<p>${cooldownText}</p>` : ""}
                                                    </div>
                                                </li>
                                            `;
                                        }).join('')}
                                    </ul>
                                </div>
                            </div>
                        </div>`;
                });

                contentArea.innerHTML = html;
            })
            .catch((err) => {
                console.error(err);
                contentArea.innerHTML = `<p>無法取得職業資訊</p>`;
            });
    });

    // Handle image load errors
    window.handleImageError = function(img, type) {
        // 隱藏破損的圖片
        img.style.display = 'none';
        // 創建一個 '無圖片' 的提示
        const noImageDiv = document.createElement('div');
        noImageDiv.className = type === 'avatar' ? 'no-image' : 'no-image skill';
        noImageDiv.textContent = '無圖片';
        // 插入到圖片後面
        img.parentNode.insertBefore(noImageDiv, img.nextSibling);
    };

    // 其餘功能(如 ELO、Battle、Version) 你可以依相同邏輯去做
    menuElo.addEventListener("click", (e) => {
        e.preventDefault();
        alert("這裡可以做 ELO 相關操作，請自行對應後端 API。");
    });

    menuBattle.addEventListener("click", (e) => {
        e.preventDefault();
        
        alert("這裡可以做戰鬥相關操作(5:電腦vs電腦 / 6:AI vs電腦)，請自行對應後端 API。");
    });

    menuVersion.addEventListener("click", (e) => {
        e.preventDefault();
        alert("版本資訊可自行設計 UI 介面並連動後端。");
    });

    menuTestRandom.addEventListener("click", (e) => {
        e.preventDefault();
        fetch("/api/computer_vs_computer")
        // show it will finish in backend
        alert("在後端 查看。");
        
    });

});
