// API Configuration
const API_BASE_URL = '/api/v1';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const queryInput = document.getElementById('queryInput');
const askButton = document.getElementById('askButton');
const resultsSection = document.getElementById('resultsSection');
const answerContent = document.getElementById('answerContent');
const sourcesList = document.getElementById('sourcesList');
const historySection = document.getElementById('historySection');
const historyList = document.getElementById('historyList');

// State
let queryHistory = JSON.parse(localStorage.getItem('queryHistory') || '[]');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('App initialized');
    initializeEventListeners();
    renderHistory();
});

function initializeEventListeners() {
    console.log('Setting up event listeners');
    
    // Upload Area Events
    uploadArea.addEventListener('click', () => {
        console.log('Upload area clicked');
        fileInput.click();
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });
    
    fileInput.addEventListener('change', (e) => {
        console.log('File selected:', e.target.files[0]);
        const file = e.target.files[0];
        if (file) handleFileUpload(file);
    });
    
    // Query Events
    askButton.addEventListener('click', () => {
        console.log('Ask button clicked');
        handleQuery();
    });
    
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleQuery();
        }
    });
}

// File Upload Handler
async function handleFileUpload(file) {
    console.log('Uploading file:', file.name);
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showUploadStatus('Only PDF files are supported', 'error');
        return;
    }
    
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showUploadStatus('File size exceeds 10MB limit', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showUploadStatus('Uploading ' + file.name + '...', 'loading');
    
    try {
        const response = await fetch(API_BASE_URL + '/ingest', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        showUploadStatus(
            ' Successfully uploaded and processed: ' + result.file,
            'success'
        );
        
        // Clear after 5 seconds
        setTimeout(() => {
            uploadStatus.className = 'upload-status';
            uploadStatus.textContent = '';
        }, 5000);
        
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus(' Upload failed: ' + error.message, 'error');
    }
}

function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = 'upload-status ' + type;
}

// Query Handler with Streaming Support
async function handleQuery() {
    const query = queryInput.value.trim();
    
    console.log('Handling query:', query);
    
    if (!query) {
        alert('Please enter a question');
        return;
    }
    
    // Disable button and show loading
    askButton.disabled = true;
    askButton.querySelector('.button-text').style.display = 'none';
    askButton.querySelector('.button-loader').style.display = 'inline-flex';
    
    // Show results section immediately
    resultsSection.style.display = 'block';
    answerContent.innerHTML = '<p class="streaming-status">Processing your question...</p>';
    sourcesList.innerHTML = '';
    
    let fullAnswer = '';
    let sources = [];
    let currentStatus = '';
    
    try {
        const response = await fetch(API_BASE_URL + '/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error('Query failed with status: ' + response.status);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                console.log('Stream complete');
                break;
            }
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.substring(6);
                    try {
                        const data = JSON.parse(jsonStr);
                        console.log('Received:', data);
                        
                        if (data.type === 'status') {
                            // Update status message
                            currentStatus = data.data;
                            answerContent.innerHTML = '<p class="streaming-status"><span class="status-icon">⏳</span> ' + data.data + '</p>';
                        } else if (data.type === 'token') {
                            // Append token to answer
                            if (currentStatus) {
                                // Clear status, start showing answer
                                answerContent.innerHTML = '';
                                currentStatus = '';
                            }
                            fullAnswer += data.data;
                            answerContent.innerHTML = formatAnswer(fullAnswer) + '<span class="cursor-blink">▊</span>';
                        } else if (data.type === 'sources') {
                            // Display sources
                            sources = data.data;
                            // Remove cursor
                            answerContent.innerHTML = formatAnswer(fullAnswer);
                            displaySources(sources);
                        } else if (data.type === 'error') {
                            throw new Error(data.data);
                        }
                    } catch (e) {
                        console.error('Failed to parse JSON:', jsonStr, e);
                    }
                }
            }
        }
        
        // Save to history
        if (fullAnswer && sources) {
            saveToHistory(query, { answer: fullAnswer, sources: sources });
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Query error:', error);
        answerContent.innerHTML = '<p class="error-message">❌ Error: ' + error.message + '</p>';
    } finally {
        // Re-enable button
        askButton.disabled = false;
        askButton.querySelector('.button-text').style.display = 'inline';
        askButton.querySelector('.button-loader').style.display = 'none';
    }
}

// Display Results (for history items)
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display answer with basic formatting
    answerContent.innerHTML = formatAnswer(result.answer);
    
    // Display sources
    displaySources(result.sources);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Sources Helper
function displaySources(sources) {
    if (!sources || sources.length === 0) {
        sourcesList.innerHTML = '<p class="no-sources">No sources available</p>';
        return;
    }
    
    sourcesList.innerHTML = sources
        .map((source, index) => createSourceElement(source, index))
        .join('');
}

function formatAnswer(answer) {
    // Simple formatting: preserve line breaks and add basic styling
    return answer
        .split('\n')
        .map(line => {
            line = line.trim();
            if (!line) return '<br>';
            
            // Bold text between **
            line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // Code blocks
            line = line.replace(/`(.*?)`/g, '<code>$1</code>');
            
            return '<p>' + line + '</p>';
        })
        .join('');
}

function createSourceElement(source, index) {
    const score = source.rerank_score 
        ? (source.rerank_score * 100).toFixed(1)
        : (source.score * 100).toFixed(1);
    
    const sourceId = 'source-' + index;
    
    return '<div class="source-item">' +
            '<div class="source-header">' +
                '<span class="source-title">Source ' + (index + 1) + '</span>' +
                '<span class="source-score">' + score + '% match</span>' +
            '</div>' +
            '<div class="source-content" id="' + sourceId + '">' +
                escapeHtml(source.content) +
            '</div>' +
            '<span class="source-toggle" onclick="toggleSource(\'' + sourceId + '\')">' +
                'Show more' +
            '</span>' +
        '</div>';
}

function toggleSource(sourceId) {
    const element = document.getElementById(sourceId);
    const toggle = element.nextElementSibling;
    
    if (element.classList.contains('expanded')) {
        element.classList.remove('expanded');
        toggle.textContent = 'Show more';
    } else {
        element.classList.add('expanded');
        toggle.textContent = 'Show less';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// History Management
function saveToHistory(query, result) {
    const historyItem = {
        query: query,
        answer: result.answer,
        timestamp: new Date().toISOString(),
        sources: result.sources
    };
    
    queryHistory.unshift(historyItem);
    
    // Keep only last 10 queries
    if (queryHistory.length > 10) {
        queryHistory = queryHistory.slice(0, 10);
    }
    
    localStorage.setItem('queryHistory', JSON.stringify(queryHistory));
    renderHistory();
}

function renderHistory() {
    if (queryHistory.length === 0) {
        historySection.style.display = 'none';
        return;
    }
    
    historySection.style.display = 'block';
    
    historyList.innerHTML = queryHistory
        .map((item, index) => {
            const date = new Date(item.timestamp);
            const timeStr = date.toLocaleString();
            
            return '<div class="history-item" onclick="loadHistoryItem(' + index + ')">' +
                    '<div class="history-question">' + escapeHtml(item.query) + '</div>' +
                    '<div class="history-time">' + timeStr + '</div>' +
                '</div>';
        })
        .join('');
}

function loadHistoryItem(index) {
    const item = queryHistory[index];
    queryInput.value = item.query;
    displayResults({
        answer: item.answer,
        sources: item.sources
    });
}

// Global functions for onclick handlers
window.toggleSource = toggleSource;
window.loadHistoryItem = loadHistoryItem;
