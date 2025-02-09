document.addEventListener('DOMContentLoaded', function () {
    const skipRoundBtn = document.getElementById('skip-round-btn');
    if (skipRoundBtn) {
      skipRoundBtn.addEventListener('click', function (e) {
        e.preventDefault();
        console.log('跳過當前回合按鈕被點擊');
        fetch('/next_version', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        })
        .then(response => {
          console.log('伺服器回應：', response);
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          console.log('取得資料：', data);
          // 延遲 300 毫秒後刷新頁面
          setTimeout(function() {
            window.location.reload();
          }, 300);
        })
        .catch(error => {
          console.error('發生錯誤:', error);
        });
      });
    }
  });
  