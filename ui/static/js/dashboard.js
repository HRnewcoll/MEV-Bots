/* ================================================================
   MEV Bot Dashboard — JavaScript
   Handles: tab switching, SSE stream, API polling, charts, controls
================================================================ */

'use strict';

// ----------------------------------------------------------------
// Tab switching
// ----------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'trades') refreshTrades();
    if (btn.dataset.tab === 'setup')  refreshSetupCheck();
  });
});

// ----------------------------------------------------------------
// Chart.js setup
// ----------------------------------------------------------------
const chartDefaults = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: '#6e7681', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#6e7681' }, grid: { color: '#21262d' } },
  },
  animation: { duration: 0 },
};

function mkLineChart(id, label, color) {
  return new Chart(document.getElementById(id).getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label,
        data: [],
        borderColor: color,
        backgroundColor: color + '22',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }],
    },
    options: { ...chartDefaults },
  });
}

function mkDoughnutChart(id) {
  return new Chart(document.getElementById(id).getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Arbitrage', 'Sandwich', 'Liquidation', 'Flash Loan'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#1f6feb', '#d29922', '#8957e5', '#3fb950'],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, position: 'bottom', labels: { color: '#8b949e', font: { size: 11 } } },
      },
      animation: { duration: 0 },
    },
  });
}

const pnlChart      = mkLineChart('chart-pnl', 'P&L (ETH)', '#3fb950');
const gasChart      = mkLineChart('chart-gas', 'Gas (Gwei)', '#ffa657');
const stratChart    = mkDoughnutChart('chart-strategy');

const MAX_CHART_POINTS = 120;

function pushChartPoint(chart, label, value) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > MAX_CHART_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

let cumulativePnl = 0;

// ----------------------------------------------------------------
// SSE — real-time event stream
// ----------------------------------------------------------------
let logCount = 0;
const logContainer   = document.getElementById('log-container');
const logCountEl     = document.getElementById('log-count');
const autoScrollChk  = document.getElementById('log-autoscroll');

function appendLog(cls, msg) {
  const entry = document.createElement('div');
  entry.className = `log-entry log-entry--${cls}`;
  const ts = new Date().toISOString().slice(11, 23);
  entry.innerHTML = `<span class="log-ts">[${ts}]</span>${msg}`;
  logContainer.appendChild(entry);
  logCount++;
  logCountEl.textContent = logCount + ' events';
  if (autoScrollChk.checked) logContainer.scrollTop = logContainer.scrollHeight;
}

document.getElementById('btn-clear-logs').addEventListener('click', () => {
  logContainer.innerHTML = '';
  logCount = 0;
  logCountEl.textContent = '0 events';
});

let _sseSource = null;

function connectSSE() {
  if (_sseSource) _sseSource.close();
  _sseSource = new EventSource('/api/stream');
  _sseSource.onmessage = evt => {
    let data;
    try { data = JSON.parse(evt.data); } catch { return; }
    handleEvent(data);
  };
  _sseSource.onerror = () => {
    appendLog('error', '⚡ Stream disconnected — reconnecting in 3s…');
    setTimeout(connectSSE, 3000);
  };
}

function handleEvent(data) {
  const type = data.type || 'info';
  switch (type) {
    case 'block':
      updateHeader(data);
      appendLog('block', `📦 Block ${data.block} | Gas ${data.gas_gwei?.toFixed(1)} Gwei`);
      pushChartPoint(gasChart, data.block, data.gas_gwei);
      break;
    case 'trade':
      cumulativePnl += (data.profit_eth || 0);
      pushChartPoint(pnlChart, data.block || '', cumulativePnl);
      updateStratChart(data.strategy);
      appendLog('trade',
        `✅ ${(data.strategy || 'trade').toUpperCase()} | ` +
        `profit: ${(data.profit_eth || 0).toFixed(6)} ETH | ` +
        `${data.buy_dex || ''} → ${data.sell_dex || ''}`);
      refreshStatus();
      break;
    case 'started':
      appendLog('started', `🚀 Simulation started | chain: ${data.chain} | strategies: ${(data.strategies || []).join(', ')}`);
      break;
    case 'stopped':
      appendLog('stopped', `🛑 Simulation stopped | P&L: ${(data.stats?.total_profit_eth || 0).toFixed(6)} ETH`);
      updateStatusDot(false, false);
      break;
    case 'paused':
      appendLog('info', '⏸ Simulation paused');
      updateStatusDot(true, true);
      break;
    case 'resumed':
      appendLog('info', '▶ Simulation resumed');
      updateStatusDot(true, false);
      break;
    case 'error':
      appendLog('error', `❌ Error: ${data.msg}`);
      break;
    case 'gas_too_high':
      appendLog('info', `⛽ Gas too high: ${data.gas_gwei?.toFixed(1)} Gwei — skipping`);
      break;
    default:
      appendLog('info', JSON.stringify(data));
  }
}

// ----------------------------------------------------------------
// Header updater
// ----------------------------------------------------------------
function updateHeader(data) {
  if (data.block)    document.getElementById('block-label').textContent = `Block: ${data.block}`;
  if (data.gas_gwei) document.getElementById('gas-label').textContent   = `Gas: ${data.gas_gwei.toFixed(1)} Gwei`;
}

function updateStatusDot(running, paused) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className = 'dot ' + (running ? (paused ? 'dot--paused' : 'dot--running') : 'dot--stopped');
  text.textContent = running ? (paused ? 'Paused' : 'Running') : 'Stopped';
}

function updateStratChart(strategy) {
  const stratMap = { arbitrage: 0, sandwich: 1, liquidation: 2, flash_loan: 3 };
  const idx = stratMap[strategy];
  if (idx !== undefined) {
    stratChart.data.datasets[0].data[idx]++;
    stratChart.update('none');
  }
}

// ----------------------------------------------------------------
// Status polling (every 3 s)
// ----------------------------------------------------------------
let _ethPrice = 3000; // default ETH price for USD estimates

async function refreshStatus() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) return;
    const s = await resp.json();

    // Header
    updateStatusDot(s.running, s.paused);
    if (s.chain) {
      document.getElementById('chain-label').textContent = s.chain;
    }

    const w = s.wallet || {};
    const pnl = w.total_profit_eth || 0;

    // Stat cards
    const pnlEl = document.getElementById('stat-pnl');
    pnlEl.textContent = pnl.toFixed(6) + ' ETH';
    pnlEl.className = 'card-value ' + (pnl >= 0 ? 'card-value--pos' : 'card-value--neg');
    document.getElementById('stat-pnl-usd').textContent = `≈ $${(pnl * _ethPrice).toFixed(2)}`;
    document.getElementById('stat-trades').textContent  = w.total_trades || 0;
    document.getElementById('stat-win-rate').textContent = `Win rate: ${((w.win_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById('stat-gas').textContent      = (w.total_gas_cost_eth || 0).toFixed(6) + ' ETH';
    document.getElementById('stat-gas-usd').textContent  = `≈ $${((w.total_gas_cost_eth || 0) * _ethPrice).toFixed(2)}`;
    document.getElementById('stat-best').textContent     = (w.best_trade_eth || 0).toFixed(6) + ' ETH';
    document.getElementById('stat-worst').textContent    = `Worst: ${(w.worst_trade_eth || 0).toFixed(6)} ETH`;
    document.getElementById('stat-tpm').textContent      = (w.trades_per_minute || 0).toFixed(2);
    document.getElementById('stat-elapsed').textContent  = `Elapsed: ${Math.round(s.elapsed_s || 0)}s`;

    // Simulation tab metrics
    document.getElementById('m-status').textContent = s.running ? (s.paused ? 'Paused' : 'Running') : 'Stopped';
    document.getElementById('m-chain').textContent  = s.chain || '—';
    document.getElementById('m-step').textContent   = s.step || 0;
    document.getElementById('m-eth').textContent    = (w.eth_balance || 0).toFixed(4) + ' ETH';
    const pnlMEl = document.getElementById('m-pnl');
    pnlMEl.textContent = pnl.toFixed(6) + ' ETH';
    pnlMEl.style.color = pnl >= 0 ? '#3fb950' : '#f85149';
    document.getElementById('m-trades').textContent = w.total_trades || 0;
    document.getElementById('m-win').textContent    = ((w.win_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('m-pairs').textContent  = (s.cache || {}).pair_cache_size || 0;
    document.getElementById('m-res').textContent    = (s.cache || {}).reserve_cache_size || 0;

    const errors = (s.recent_errors || []);
    document.getElementById('m-errors').textContent = errors.length || 0;
    document.getElementById('error-list').textContent = errors.length ? errors.join('\n') : 'None';

    // Update recent trades table on dashboard
    const tbody = document.getElementById('recent-trades-body');
    const trades = await (await fetch('/api/trades?n=15')).json();
    renderTradesTable(tbody, trades);
  } catch { /* ignore network errors while not connected */ }
}

// ----------------------------------------------------------------
// Trades tab
// ----------------------------------------------------------------
async function refreshTrades() {
  try {
    const [tradesResp, statsResp] = await Promise.all([
      fetch('/api/trades?n=100'),
      fetch('/api/stats'),
    ]);
    const trades = await tradesResp.json();
    const stats  = await statsResp.json();

    // Aggregate cards
    const cardsEl = document.getElementById('strat-cards');
    cardsEl.innerHTML = '';
    for (const [strat, info] of Object.entries(stats.by_strategy || {})) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-label">${strat}</div>
        <div class="card-value">${info.count}</div>
        <div class="card-sub ${info.profit >= 0 ? 'card-value--pos' : 'card-value--neg'}" style="font-size:13px">
          ${info.profit.toFixed(6)} ETH
        </div>`;
      cardsEl.appendChild(card);
    }

    const totalLabel = document.getElementById('agg-stats-label');
    totalLabel.textContent = `Total: ${stats.total_trades || 0} trades | P&L: ${(stats.total_profit_eth || 0).toFixed(6)} ETH`;

    renderTradesTable(document.getElementById('all-trades-body'), trades);
  } catch {}
}

function renderTradesTable(tbody, trades) {
  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-row">No trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => {
    const ts = new Date(t.timestamp * 1000).toISOString().slice(11, 19);
    const profitCls = t.profit_eth >= 0 ? 'profit-pos' : 'profit-neg';
    const stratCls  = `strategy-badge strat--${t.strategy || 'arbitrage'}`;
    return `<tr>
      <td>${ts}</td>
      <td><span class="${stratCls}">${t.strategy || '—'}</span></td>
      <td>${t.buy_dex || '—'}</td>
      <td>${t.sell_dex || '—'}</td>
      <td>${(t.amount_in || 0).toFixed(4)}</td>
      <td>${(t.amount_out || 0).toFixed(4)}</td>
      <td>${(t.gas_cost_eth || 0).toFixed(6)}</td>
      <td class="${profitCls}">${(t.profit_eth || 0).toFixed(6)}</td>
      <td>${t.block || '—'}</td>
    </tr>`;
  }).join('');
}

// ----------------------------------------------------------------
// Simulation controls
// ----------------------------------------------------------------
document.getElementById('btn-start').addEventListener('click', async () => {
  const strategies = [];
  if (document.getElementById('cfg-arb').checked)        strategies.push('arbitrage');
  if (document.getElementById('cfg-sandwich').checked)   strategies.push('sandwich');
  if (document.getElementById('cfg-liquidation').checked) strategies.push('liquidation');

  if (strategies.length === 0) {
    showSimMsg('Select at least one strategy', true);
    return;
  }

  const body = {
    chain:              document.getElementById('cfg-chain').value,
    rpc_url:            document.getElementById('cfg-rpc').value.trim(),
    initial_eth:        parseFloat(document.getElementById('cfg-eth').value),
    trade_amount_eth:   parseFloat(document.getElementById('cfg-trade').value),
    min_profit_eth:     parseFloat(document.getElementById('cfg-minprofit').value),
    max_gas_price_gwei: parseFloat(document.getElementById('cfg-maxgas').value),
    poll_interval_s:    parseFloat(document.getElementById('cfg-poll').value),
    strategies,
  };

  try {
    const resp = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (data.ok) {
      showSimMsg('✅ Simulation started!', false);
      cumulativePnl = 0;
      pnlChart.data.labels = [];
      pnlChart.data.datasets[0].data = [];
      pnlChart.update('none');
      gasChart.data.labels = [];
      gasChart.data.datasets[0].data = [];
      gasChart.update('none');
      stratChart.data.datasets[0].data = [0, 0, 0, 0];
      stratChart.update('none');
      updateStatusDot(true, false);
    } else {
      showSimMsg('❌ ' + (data.msg || 'Failed to start'), true);
    }
  } catch (e) {
    showSimMsg('❌ Network error: ' + e.message, true);
  }
});

document.getElementById('btn-stop').addEventListener('click', async () => {
  await fetch('/api/stop', { method: 'POST' });
  showSimMsg('Simulation stopped', false);
  updateStatusDot(false, false);
});

document.getElementById('btn-pause').addEventListener('click', async () => {
  const resp = await fetch('/api/pause', { method: 'POST' });
  const data = await resp.json();
  document.getElementById('btn-pause').textContent = data.paused ? '▶ Resume' : '⏸ Pause';
  updateStatusDot(true, data.paused);
});

document.getElementById('btn-refresh-trades').addEventListener('click', refreshTrades);
document.getElementById('btn-recheck').addEventListener('click', refreshSetupCheck);

// ----------------------------------------------------------------
// Setup tab
// ----------------------------------------------------------------
async function refreshSetupCheck() {
  const el = document.getElementById('setup-checklist');
  el.innerHTML = '<div class="check-loading">Checking…</div>';
  try {
    const resp = await fetch('/api/setup_check');
    const data = await resp.json();
    const checks = data.checks || [];
    if (checks.length === 0) {
      el.innerHTML = '<div class="check-loading">No checks available</div>';
      return;
    }
    el.innerHTML = checks.map(c => `
      <div class="check-item ${c.ok ? 'check-item--ok' : 'check-item--fail'}">
        <span class="check-icon">${c.ok ? '✅' : '❌'}</span>
        <span class="check-label">${c.label}</span>
        ${!c.ok ? `<span class="check-fix">${c.fix}</span>` : ''}
      </div>
    `).join('');
  } catch {
    el.innerHTML = '<div class="check-loading" style="color:#f85149">Could not load setup status</div>';
  }
}

function showSimMsg(msg, isErr) {
  const el = document.getElementById('sim-msg');
  el.textContent = msg;
  el.className = 'sim-msg ' + (isErr ? 'sim-msg--err' : 'sim-msg--ok');
  setTimeout(() => { el.textContent = ''; el.className = 'sim-msg'; }, 4000);
}

// ----------------------------------------------------------------
// Initialise
// ----------------------------------------------------------------
connectSSE();
refreshStatus();
setInterval(refreshStatus, 3000);

// On first load, auto-switch to Setup tab so new users see the quick-start guide.
// Uses localStorage so it only happens once — not every time a simulation is stopped.
(async () => {
  const seen = localStorage.getItem('setup_tab_seen');
  if (!seen) {
    try {
      const s = await (await fetch('/api/status')).json();
      if (!s.running) {
        const setupBtn = document.querySelector('[data-tab="setup"]');
        if (setupBtn) {
          setupBtn.click();
          localStorage.setItem('setup_tab_seen', '1');
        }
      }
    } catch {}
  }
})();
