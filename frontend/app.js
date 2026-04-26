/**
 * app.js — Frontend logic for the AI Expense Chatbot.
 *
 * Handles all API communication, UI state, and interactions.
 */

const API = '';  // Same origin

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(`${API}${url}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    throw e;
  }
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = { success: 'Success', error: 'error', info: 'ℹinfo' };
  toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function formatCurrency(amount) {
  return `₹ ${Number(amount).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}


// ──────────────────────────────────────────────
// Tab Switching
// ──────────────────────────────────────────────

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  document.getElementById(`panel-${tabName}`).classList.add('active');

  // Refresh data when switching tabs
  if (tabName === 'history') loadHistory();
  if (tabName === 'chat') document.getElementById('chatInput').focus();
}


// ──────────────────────────────────────────────
// Mobile Sidebar
// ──────────────────────────────────────────────

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarOverlay').classList.toggle('visible');
}


// ──────────────────────────────────────────────
// Dashboard Stats
// ──────────────────────────────────────────────

async function loadStats() {
  try {
    const data = await apiFetch('/api/stats');

    document.getElementById('statMonth').textContent = formatCurrency(data.month_total);
    document.getElementById('statAllTime').textContent = formatCurrency(data.all_time_total);
    document.getElementById('statCount').textContent = data.receipt_count;
    document.getElementById('statCategories').textContent = Object.keys(data.category_breakdown).length;

    renderCategoryChart(data.category_breakdown);
  } catch (e) {
    console.error('Failed to load stats:', e);
  }
}

function renderCategoryChart(breakdown) {
  const container = document.getElementById('categoryChart');
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 12px 0;">No data yet</div>';
    return;
  }

  const maxVal = Math.max(...entries.map(([, v]) => v));

  container.innerHTML = entries.map(([cat, val]) => {
    const pct = maxVal > 0 ? (val / maxVal) * 100 : 0;
    return `
      <div class="category-bar">
        <span class="cat-name">${cat}</span>
        <div class="cat-bar-track">
          <div class="cat-bar-fill" style="width: 0%" data-target="${pct}"></div>
        </div>
        <span class="cat-amount">${val.toFixed(0)}</span>
      </div>
    `;
  }).join('');

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.cat-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 50);
  });
}


// ──────────────────────────────────────────────
// Upload Receipt
// ──────────────────────────────────────────────

let currentRawText = '';

// Drag & drop
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    processFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    processFile(fileInput.files[0]);
  }
});

async function processFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file (PNG, JPG)', 'error');
    return;
  }

  const section = document.getElementById('processingSection');
  const resultGrid = document.getElementById('resultGrid');
  const uploadError = document.getElementById('uploadError');

  section.classList.add('visible');
  resultGrid.style.display = 'none';
  uploadError.style.display = 'none';

  // Show preview
  const previewImg = document.getElementById('receiptPreview');
  previewImg.src = URL.createObjectURL(file);

  // Step 1: OCR
  setStepState('stepOcr', 'active');
  setStepState('stepAi', '');
  setStepState('stepDone', '');

  try {
    const formData = new FormData();
    formData.append('file', file);

    // Start upload
    setStepState('stepOcr', 'active');

    const result = await fetch(`${API}/api/upload`, {
      method: 'POST',
      body: formData,
    }).then(r => r.json());

    setStepState('stepOcr', 'done');

    if (!result.success) {
      setStepState('stepAi', 'error');
      uploadError.textContent = result.error || 'Failed to process receipt.';
      uploadError.style.display = 'block';
      return;
    }

    // Step 2: AI done
    setStepState('stepAi', 'done');

    if (!result.data) {
      setStepState('stepDone', 'error');
      uploadError.textContent = 'AI failed to parse the receipt. Please try a clearer photo.';
      uploadError.style.display = 'block';
      return;
    }

    // Step 3: Ready
    setStepState('stepDone', 'done');

    // Fill form
    currentRawText = result.raw_text;
    document.getElementById('fldMerchant').value = result.data.merchant || '';
    document.getElementById('fldDate').value = result.data.date || '';
    document.getElementById('fldTotal').value = result.data.total || 0;
    document.getElementById('fldCurrency').value = result.data.currency || '₹';
    document.getElementById('fldCategory').value = result.data.category || 'Other';
    document.getElementById('fldRawText').value = result.raw_text;

    resultGrid.style.display = 'grid';

  } catch (e) {
    setStepState('stepOcr', 'error');
    uploadError.textContent = `Error: ${e.message}`;
    uploadError.style.display = 'block';
    showToast('Failed to process receipt', 'error');
  }
}

function setStepState(stepId, state) {
  const el = document.getElementById(stepId);
  el.className = 'step' + (state ? ` ${state}` : '');
}

async function saveReceipt() {
  const data = {
    merchant: document.getElementById('fldMerchant').value,
    date: document.getElementById('fldDate').value,
    total: parseFloat(document.getElementById('fldTotal').value) || 0,
    currency: document.getElementById('fldCurrency').value,
    category: document.getElementById('fldCategory').value,
    raw_text: currentRawText,
  };

  try {
    const res = await apiFetch('/api/save', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    showToast(res.message, 'success');
    loadStats(); // Refresh dashboard
  } catch (e) {
    showToast(`Failed to save: ${e.message}`, 'error');
  }
}

function resetUpload() {
  document.getElementById('processingSection').classList.remove('visible');
  document.getElementById('resultGrid').style.display = 'none';
  document.getElementById('uploadError').style.display = 'none';
  fileInput.value = '';
  currentRawText = '';
}


// ──────────────────────────────────────────────
// Chat
// ──────────────────────────────────────────────

async function sendChat() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  input.value = '';
  const emptyState = document.getElementById('chatEmptyState');
  if (emptyState) emptyState.remove();

  appendChatBubble(message, 'user');

  // Show typing indicator
  const typingIndicator = document.getElementById('typingIndicator');
  typingIndicator.classList.add('visible');
  scrollChat();

  // Disable input
  input.disabled = true;
  document.getElementById('chatSendBtn').disabled = true;

  try {
    const data = await apiFetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    });

    typingIndicator.classList.remove('visible');
    appendChatBubble(data.response, 'assistant');
    loadStats(); // Refresh stats in case spending data changed
  } catch (e) {
    typingIndicator.classList.remove('visible');
    appendChatBubble(`Error: ${e.message}`, 'assistant');
    showToast('Chat error', 'error');
  } finally {
    input.disabled = false;
    document.getElementById('chatSendBtn').disabled = false;
    input.focus();
  }
}

function appendChatBubble(text, role) {
  const container = document.getElementById('chatMessages');
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;

  const label = role === 'user' ? 'You' : 'ExpenseAI';
  bubble.innerHTML = `
    <div class="msg-label">${label}</div>
    <div>${formatMessage(text)}</div>
  `;

  container.appendChild(bubble);
  scrollChat();
}

function formatMessage(text) {
  // Basic markdown-like formatting
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

function scrollChat() {
  const container = document.getElementById('chatMessages');
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}


// ──────────────────────────────────────────────
// History
// ──────────────────────────────────────────────

async function loadHistory() {
  const wrapper = document.getElementById('historyWrapper');
  const empty = document.getElementById('historyEmpty');

  try {
    const data = await apiFetch('/api/receipts');

    if (data.receipts.length === 0) {
      wrapper.innerHTML = '';
      wrapper.appendChild(createEmptyState());
      return;
    }

    const categoryBadge = (cat) => {
      const cls = `badge badge-${(cat || 'other').toLowerCase()}`;
      return `<span class="${cls}">${cat || 'Other'}</span>`;
    };

    wrapper.innerHTML = `
      <table class="history-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Date</th>
            <th>Merchant</th>
            <th>Amount</th>
            <th>Currency</th>
            <th>Category</th>
          </tr>
        </thead>
        <tbody>
          ${data.receipts.map(r => `
            <tr>
              <td>${r.id}</td>
              <td>${r.date || '—'}</td>
              <td style="color: var(--text-primary); font-weight: 500;">${r.merchant || '—'}</td>
              <td style="font-weight: 600; color: var(--text-primary);">${Number(r.total || 0).toFixed(2)}</td>
              <td>${r.currency || '—'}</td>
              <td>${categoryBadge(r.category)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (e) {
    showToast(`Failed to load history: ${e.message}`, 'error');
  }
}

function createEmptyState() {
  const div = document.createElement('div');
  div.className = 'empty-state';
  div.id = 'historyEmpty';
  div.innerHTML = `
    <p>No receipts found. Upload your first receipt to get started!</p>
  `;
  return div;
}


// ──────────────────────────────────────────────
// Sidebar Actions
// ──────────────────────────────────────────────

async function resetAgent() {
  try {
    const res = await apiFetch('/api/reset-agent', { method: 'POST' });
    showToast(res.message, 'success');
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

async function clearChat() {
  try {
    await apiFetch('/api/chat/history', { method: 'DELETE' });
    const container = document.getElementById('chatMessages');
    container.innerHTML = `
      <div class="chat-empty-state" id="chatEmptyState">
        <p>Ask me anything about your expenses</p>
        <p style="font-size: 0.8rem; color: var(--text-muted);">e.g. "How much did I spend on Food?" or "What's my biggest expense?"</p>
      </div>
    `;
    showToast('Chat cleared', 'success');
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

async function resetDatabase() {
  if (!confirm('This will permanently delete ALL receipts and chat history. Are you sure?')) return;

  try {
    await apiFetch('/api/reset-db', { method: 'POST' });
    showToast('Database completely wiped!', 'success');
    loadStats();
    clearChat();
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}


// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
});
