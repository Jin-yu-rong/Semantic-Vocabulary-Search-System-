document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const topKSelect = document.getElementById('topK');
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsCount = document.getElementById('resultsCount');
    const searchTime = document.getElementById('searchTime');
    const loadingOverlay = document.getElementById('loading');
    const exampleTags = document.querySelectorAll('.example-tag');

    // 阈值滑块显示
    thresholdSlider.addEventListener('input', function() {
        thresholdValue.textContent = (this.value / 10).toFixed(1);
    });

    // 搜索触发
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') performSearch();
    });

    // 示例点击
    exampleTags.forEach(tag => {
        tag.addEventListener('click', () => {
            searchInput.value = tag.dataset.query || tag.textContent;
            performSearch();
        });
    });

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            alert('请输入描述内容');
            return;
        }

        loadingOverlay.style.display = 'flex';
        resultsContainer.innerHTML = '';
        resultsCount.textContent = '搜索中...';
        searchTime.textContent = '';

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topKSelect.value),
                    threshold: parseFloat(thresholdSlider.value) / 10
                })
            });

            if (!response.ok) throw new Error('网络错误');

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            renderResults(data);

        } catch (err) {
            resultsContainer.innerHTML = `
                <div class="empty-state" style="color:#e74c3c;text-align:center;padding:60px;">
                    <i class="fas fa-exclamation-triangle fa-3x"></i>
                    <h3>搜索失败</h3>
                    <p>${err.message}</p>
                </div>
            `;
            resultsCount.textContent = '搜索失败';
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    function renderResults(data) {
        resultsCount.textContent = `找到 ${data.count} 个相关单词`;
        searchTime.textContent = `用时 ${data.time_ms.toFixed(1)} ms`;

        if (data.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search fa-3x"></i>
                    <h3>未找到相关单词</h3>
                    <p>试试降低相似度阈值或换个描述方式</p>
                </div>
            `;
            return;
        }

        // 逐个创建美化卡片
        data.results.forEach((result, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';

            // 相似度分类
            let scoreClass = 'score-low';
            if (result.score >= 0.6) scoreClass = 'score-high';
            else if (result.score >= 0.4) scoreClass = 'score-medium';

            // 星星（最高5星）
            const starCount = Math.round(result.score * 10);
            const stars = '★'.repeat(Math.min(starCount, 5)) + '☆'.repeat(5 - Math.min(starCount, 5));

            card.innerHTML = `
                <div class="word-header">
                    <h3 class="word">${result.word}</h3>
                    <div class="score-info ${scoreClass}">
                        <span class="stars">${stars}</span>
                        <span class="score-value">${result.score.toFixed(4)}</span>
                    </div>
                </div>
                <div class="meaning">${result.meaning}</div>
                <div class="score-bar">
                    <div class="fill" style="width: ${result.score * 100}%"></div>
                </div>
            `;

            // 渐入动画
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'all 0.6s ease';
            card.style.animationDelay = `${index * 0.1}s`;

            resultsContainer.appendChild(card);

            // 触发动画
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50);
        });
    }
});