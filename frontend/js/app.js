// Relative path — works on any machine, no need to hardcode localhost
const API_BASE_URL = '/predict';
let currentMode = 'ultrasound'; // 'ultrasound' or 'biopsy'
let selectedFile = null;

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabDesc = document.getElementById('tab-desc');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const dzIdle = document.getElementById('dz-idle');
const dzPreview = document.getElementById('dz-preview');
const imagePreview = document.getElementById('image-preview');
const btnRemove = document.getElementById('btn-remove');
const btnAnalyze = document.getElementById('btn-analyze');

// Results Elements
const resultsPanel = document.getElementById('results-panel');
const loader = document.getElementById('loader');
const report = document.getElementById('report');
const errorBox = document.getElementById('error-box');
const errorText = document.getElementById('error-text');
const verdictMain = document.getElementById('verdict-main');
const verdictConf = document.getElementById('verdict-conf');
const detailsBars = document.getElementById('details-bars');
const mainVerdictBox = document.querySelector('.main-verdict');

// Descriptions for tabs
const descriptions = {
    ultrasound: "Chẩn đoán u vú dựa trên hình ảnh Siêu âm (Ultrasound) sử dụng thuật toán EfficientNet-B3.",
    biopsy: "Phân loại mức độ ác tính dựa trên hình ảnh Vi thể mô bệnh học (Histopathology) sử dụng thuật toán ResNet-50."
};

// --- Tab Switching Logic ---
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Reset ui
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        currentMode = btn.dataset.tab;
        tabDesc.textContent = descriptions[currentMode];
        
        resetUpload();
    });
});

// --- File Handling Logic ---
// Click to open file dialog
dropzone.addEventListener('click', (e) => {
    if (e.target !== btnRemove && e.target.closest('#btn-remove') === null) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// Drag & Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-active');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-active');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-active');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

function handleFile(file) {
    if (!file.type.match('image.*')) {
        showError("Vui lòng chỉ tải lên file hình ảnh (.jpg, .png)");
        return;
    }
    
    selectedFile = file;
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dzIdle.classList.add('hidden');
        dzPreview.classList.remove('hidden');
        btnAnalyze.disabled = false;
        
        // Hide previous results
        resultsPanel.classList.add('hidden');
    }
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    dzIdle.classList.remove('hidden');
    dzPreview.classList.add('hidden');
    dzPreview.classList.remove('has-heatmap');
    
    const existingHeatmap = document.getElementById('heatmap-overlay');
    if (existingHeatmap) existingHeatmap.remove();
    
    btnAnalyze.disabled = true;
    resultsPanel.classList.add('hidden');
}

// --- API Interaction Logic ---
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Prepare UI for loading
    resultsPanel.classList.remove('hidden');
    loader.classList.remove('hidden');
    report.classList.add('hidden');
    errorBox.classList.add('hidden');
    btnAnalyze.disabled = true;
    
    // Prepare FormData
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const endpoint = `${API_BASE_URL}/${currentMode}`;
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Server trả về lỗi không xác định.');
        }
        
        renderResults(data);
        
    } catch (err) {
        // Assume failure connects, fallback generic message
        let errMsg = err.message;
        if(errMsg === "Failed to fetch") errMsg = "Không thể kết nối đến Máy chủ AI. Hãy đảm bảo Backend (Flask) đang chạy.";
        showError(errMsg);
    } finally {
        loader.classList.add('hidden');
        btnAnalyze.disabled = false;
    }
});

function renderResults(data) {
    report.classList.remove('hidden');
    
    // Set Main Verdict
    verdictMain.textContent = data.prediction;
    verdictConf.textContent = `Độ tin cậy: ${data.confidence}%`;
    
    // Set colors
    mainVerdictBox.className = 'data-group main-verdict'; // reset
    mainVerdictBox.classList.add(`val-${data.prediction.toLowerCase()}`);
    
    // Render Grad-CAM Heatmap OVER original image if available
    const existingHeatmap = document.getElementById('heatmap-overlay');
    if (existingHeatmap) existingHeatmap.remove();
    
    if (data.heatmap) {
        const heatmapImg = document.createElement('img');
        heatmapImg.id = 'heatmap-overlay';
        heatmapImg.src = data.heatmap;
        heatmapImg.className = 'heatmap-overlay';
        dzPreview.appendChild(heatmapImg);
        dzPreview.classList.add('has-heatmap');
    }
    
    // Set Details Bars
    detailsBars.innerHTML = '';
    
    // Sort details so highest is top mapping through entries
    const entries = Object.entries(data.details).sort((a,b) => b[1] - a[1]);
    
    entries.forEach(([clsName, percent], index) => {
        const row = document.createElement('div');
        row.className = 'bar-row';
        
        const isHighest = index === 0;
        
        row.innerHTML = `
            <div class="bar-label">${clsName}</div>
            <div class="bar-track">
                <div class="bar-fill ${isHighest ? 'Highest' : ''}" style="width: 0%"></div>
            </div>
            <div class="bar-percent">${percent}%</div>
        `;
        
        detailsBars.appendChild(row);
        
        // Timeout to animate bars
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = `${percent}%`;
        }, 100);
    });
}

function showError(msg) {
    resultsPanel.classList.remove('hidden');
    loader.classList.add('hidden');
    report.classList.add('hidden');
    errorBox.classList.remove('hidden');
    errorText.textContent = msg;
}
