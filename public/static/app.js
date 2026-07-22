const state = {
  summary: null,
  moneylines: null,
  props: null,
  bets: null,
  loaded: new Set(),
};

const LEGACY_ACCOUNT_KEY = 'moneyballPaperAccountV1';
const MIGRATION_FLAG = 'moneyballPostgresMigratedV1';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[character]));
const pct = (value, digits = 1) => value == null || !Number.isFinite(Number(value))
  ? '—'
  : `${(Number(value) * 100).toFixed(digits)}%`;
const money = (value) => value == null || !Number.isFinite(Number(value))
  ? '—'
  : `${Number(value) >= 0 ? '+' : '-'}$${Math.abs(Number(value)).toFixed(2)}`;
const dollars = (value) => value == null || !Number.isFinite(Number(value))
  ? '—'
  : `$${Number(value).toFixed(2)}`;
const cents = (value) => value == null || !Number.isFinite(Number(value))
  ? '—'
  : `${(Number(value) * 100).toFixed(1)}¢`;
const localTime = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? 'Unknown time' : date.toLocaleString([], {
    weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
  });
};
const teamLogo = (teamId) => teamId ? `https://www.mlbstatic.com/team-logos/${teamId}.svg` : '';
const headshot = (playerId) => `https://securea.mlb.com/mlb/images/players/head_shot/${playerId}.jpg`;

function replaceHtmlWithFragment(container, html) {
  const template = document.createElement('template');
  template.innerHTML = html;
  const fragment = document.createDocumentFragment();
  fragment.append(...template.content.childNodes);
  container.replaceChildren(fragment);
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove('show'), 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function accountCard(metrics, type) {
  const isProps = type === 'PLAYER_PROP';
  const pnlClass = Number(metrics.realized_pnl) >= 0 ? 'positive' : 'negative';
  return `<article class="portfolio-card">
    <header>
      <span class="account-tag"><span class="account-dot ${isProps ? 'props' : ''}"></span>${isProps ? 'PLAYER PROPS' : 'MONEYLINES'}</span>
      <span class="mono">Start ${dollars(metrics.starting_bankroll)}</span>
    </header>
    <div class="account-equity">${dollars(metrics.current_bankroll)}</div>
    <div class="metric-grid">
      <div class="metric"><span>Available cash</span><strong>${dollars(metrics.available_cash)}</strong></div>
      <div class="metric"><span>Open exposure</span><strong>${dollars(metrics.open_exposure)}</strong></div>
      <div class="metric"><span>Realized P&amp;L</span><strong class="${pnlClass}">${money(metrics.realized_pnl)}</strong></div>
      <div class="metric"><span>Total return</span><strong class="${pnlClass}">${pct(metrics.total_return)}</strong></div>
      <div class="metric"><span>Accuracy</span><strong>${pct(metrics.accuracy)}</strong></div>
      <div class="metric"><span>W / L</span><strong>${metrics.wins} / ${metrics.losses}</strong></div>
      <div class="metric"><span>Avg entry edge</span><strong>${pct(metrics.average_entry_edge)}</strong></div>
      <div class="metric"><span>Brier score</span><strong>${metrics.brier_score == null ? '—' : Number(metrics.brier_score).toFixed(3)}</strong></div>
    </div>
  </article>`;
}

function renderSummary(summary) {
  state.summary = summary;
  const accounts = summary.accounts;
  $('#overview-portfolios').innerHTML = accountCard(accounts.moneyline, 'MONEYLINE') + accountCard(accounts.player_props, 'PLAYER_PROP');
  $('#combined-capital').textContent = dollars(accounts.combined.current_bankroll);
  $('#combined-pnl').textContent = `Realized P&L ${money(accounts.combined.realized_pnl)}`;
  $('#combined-pnl').className = Number(accounts.combined.realized_pnl) >= 0 ? 'positive' : 'negative';
  const latest = summary.data_freshness.latest_job;
  $('#system-summary').innerHTML = `
    <div class="metric-row"><span>Confirmed lineup games archived</span><strong>${summary.data_freshness.confirmed_lineup_games_archived}</strong></div>
    <div class="metric-row"><span>Latest scheduled job</span><strong>${latest ? `${esc(latest.job_name)} · ${esc(latest.status)}` : 'No runs yet'}</strong></div>
    <div class="metric-row"><span>Dashboard version</span><strong>v${esc(summary.version)}</strong></div>`;
  $('#freshness').textContent = `Summary updated ${new Date(summary.generated_at).toLocaleTimeString()}`;
  renderLegacyMigration();
}

function renderLegacyMigration() {
  const container = $('#legacy-import');
  if (localStorage.getItem(MIGRATION_FLAG) === 'true') {
    container.classList.add('hidden');
    return;
  }
  let legacy = null;
  try { legacy = JSON.parse(localStorage.getItem(LEGACY_ACCOUNT_KEY)); } catch (_) {}
  if (!legacy || !Array.isArray(legacy.bets) || legacy.bets.length === 0) {
    container.classList.add('hidden');
    return;
  }
  container.classList.remove('hidden');
  container.innerHTML = `<div><strong>Existing browser paper bets found</strong><div class="section-copy">Import ${legacy.bets.length} old bets into the separate Postgres ledgers. The browser copy stays untouched as a backup.</div></div><button class="primary" id="import-legacy">Import bets</button>`;
  $('#import-legacy').addEventListener('click', async () => {
    const button = $('#import-legacy');
    button.disabled = true;
    try {
      const result = await api('/api/v1/paper/import-browser', { method: 'POST', body: JSON.stringify(legacy) });
      localStorage.setItem(MIGRATION_FLAG, 'true');
      toast(`Imported ${result.imported} bets; skipped ${result.skipped}.`);
      await loadSummary();
    } catch (error) {
      toast(error.message);
      button.disabled = false;
    }
  }, { once: true });
}

async function loadSummary() {
  const summary = await api('/api/v1/dashboard/summary');
  renderSummary(summary);
}

function sideCard(side, selected) {
  return `<div class="side-card ${selected ? 'selected' : ''}">
    <div class="side-name">${esc(side.team)}</div>
    <div class="side-price mono">${cents(side.market_buy_price)}</div>
    <div class="side-detail">Model ${pct(side.model_probability)} · Edge <span class="${side.edge >= 0 ? 'positive' : 'negative'}">${pct(side.edge)}</span><br>ROI ${pct(side.expected_roi)} · Stake ${dollars(side.kelly_stake)}</div>
  </div>`;
}

function moneylineCard(game) {
  const selected = game.recommended_side === game.side_a.team ? game.side_a : game.recommended_side === game.side_b.team ? game.side_b : null;
  const canBet = game.signal === 'BET' && game.lineup_bet_eligible && selected && Number(selected.kelly_stake) >= 0.01;
  return `<article class="game-card">
    <div>
      <div class="matchup">
        <div class="team-line">${game.away_team_id ? `<img class="team-logo" src="${teamLogo(game.away_team_id)}" alt="">` : ''}${esc(game.away_team)}</div>
        <div class="team-line">${game.home_team_id ? `<img class="team-logo" src="${teamLogo(game.home_team_id)}" alt="">` : ''}${esc(game.home_team)}</div>
      </div>
      <div class="game-meta">${localTime(game.start_time)} · ${esc(game.venue || '')}<br>${esc(game.away_probable_pitcher || 'TBD')} vs ${esc(game.home_probable_pitcher || 'TBD')}<br>${esc(game.lineup_note || '')}</div>
    </div>
    <div class="side-grid">${sideCard(game.side_a, game.recommended_side === game.side_a.team)}${sideCard(game.side_b, game.recommended_side === game.side_b.team)}</div>
    <div class="action-cell">
      <span class="signal ${esc(game.signal)}">${esc(game.signal)}${game.recommended_side ? ` · ${esc(game.recommended_side)}` : ''}</span>
      ${canBet ? `<button class="bet-button moneyline-bet" data-game="${game.game_pk}">Paper bet ${dollars(selected.kelly_stake)}</button>` : ''}
      <a class="secondary" href="${esc(game.polymarket_url)}" target="_blank" rel="noreferrer">Market</a>
    </div>
  </article>`;
}

async function loadMoneylines(force = false) {
  if (state.loaded.has('moneylines') && !force) return;
  const board = $('#moneyline-board');
  board.innerHTML = '<div class="skeleton block"></div>';
  $('#moneyline-status').textContent = 'Loading schedule, confirmed lineups, model probabilities, and executable asks…';
  try {
    const cash = state.summary?.accounts?.moneyline?.available_cash || 100;
    const data = await api(`/api/v1/polymarket/mlb?bankroll=${encodeURIComponent(cash)}&kelly_multiplier=0.25&days=2`);
    state.moneylines = data;
    state.loaded.add('moneylines');
    $('#moneyline-status').textContent = `${data.games.length} priced games · ${data.recommended_bets.length} BET signals · ${data.model_version} · updated ${new Date(data.generated_at).toLocaleTimeString()}`;
    replaceHtmlWithFragment(board, data.games.length ? data.games.map(moneylineCard).join('') : '<div class="empty">No upcoming priced moneyline markets.</div>');
  } catch (error) {
    board.innerHTML = `<div class="empty negative">${esc(error.message)}</div>`;
    $('#moneyline-status').textContent = 'Moneyline refresh failed.';
  }
}

async function placeMoneyline(gamePk) {
  const game = state.moneylines?.games?.find((item) => Number(item.game_pk) === Number(gamePk));
  if (!game) return;
  const side = game.recommended_side === game.side_a.team ? game.side_a : game.side_b;
  try {
    await api('/api/v1/paper/bets', {
      method: 'POST',
      body: JSON.stringify({
        account_type: 'MONEYLINE', market_type: 'MONEYLINE', game_pk: game.game_pk,
        market_id: game.market_id, selection: side.team, team: side.team,
        opponent: side.team === game.away_team ? game.home_team : game.away_team,
        model_version: game.model_version, model_probability: side.model_probability,
        entry_price: side.market_buy_price, entry_edge: side.edge,
        expected_roi: side.expected_roi, stake: side.kelly_stake,
        start_time: game.start_time, polymarket_url: game.polymarket_url,
        metadata: { lineup_status: game.lineup_status, value_grade: side.value_grade }
      })
    });
    toast(`Moneyline paper bet placed on ${side.team}.`);
    state.loaded.delete('portfolio');
    await loadSummary();
    await loadMoneylines(true);
  } catch (error) { toast(error.message); }
}

function contractRow(family, contract) {
  const isBest = contract.decision === 'BEST_BET';
  return `<tr>
    <td class="mono">${contract.threshold}+ K</td>
    <td><strong>${esc(contract.side)}</strong></td>
    <td class="mono">${pct(contract.raw_probability)}</td>
    <td class="mono">${pct(contract.conservative_probability)}</td>
    <td class="mono">${cents(contract.executable_price)}</td>
    <td class="mono ${Number(contract.edge) >= 0 ? 'positive' : 'negative'}">${pct(contract.edge)}</td>
    <td class="mono">${pct(contract.expected_roi)}</td>
    <td class="mono">${pct(contract.full_kelly_fraction)}</td>
    <td><span class="decision ${esc(contract.decision)}">${esc(contract.decision.replace('_', ' '))}</span></td>
    <td>${isBest ? `<button class="bet-button prop-bet" data-family="${esc(family.family_key)}" data-market="${esc(contract.market_id)}" data-side="${esc(contract.side)}">Paper bet ${dollars(contract.proposed_stake)}</button>` : `<a class="secondary" href="${esc(contract.polymarket_url)}" target="_blank" rel="noreferrer">View</a>`}</td>
  </tr>`;
}

function familyCard(family) {
  const warning = family.warning_code ? `<div class="warning"><strong>${esc(family.warning_code)}</strong><br>${esc(family.warning_message)}</div>` : '';
  return `<article class="family-card">
    <div class="family-head">
      <div class="player">
        <img class="headshot" src="${headshot(family.player_id)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
        <div><h3>${esc(family.player_name)}</h3><p>vs ${esc(family.opponent)} · ${localTime(family.start_time)} · Lineup ${esc(family.lineup_status)} (${family.hitters_used}/9)</p></div>
      </div>
      <div class="projection"><strong>${Number(family.expected_strikeouts).toFixed(2)} K</strong><span>${Number(family.projected_innings).toFixed(1)} projected innings · ${pct(family.matchup_k_rate)} matchup K%</span></div>
    </div>
    ${warning}
    <div class="table-shell"><table><thead><tr><th>Line</th><th>Side</th><th>Raw p</th><th>Shrunk p</th><th>VWAP</th><th>Edge</th><th>ROI</th><th>Full Kelly</th><th>Decision</th><th>Action</th></tr></thead><tbody>${family.contracts.map((contract) => contractRow(family, contract)).join('')}</tbody></table></div>
  </article>`;
}

async function loadProps(force = false) {
  if (state.loaded.has('props') && !force) return;
  $('#prop-board').innerHTML = '<div class="skeleton block"></div>';
  $('#prop-status').textContent = 'Grouping strikeout markets by pitcher and ranking each nested contract…';
  try {
    const data = await api('/api/v1/props/strikeouts?days=3');
    state.props = data;
    state.loaded.add('props');
    const best = data.families.filter((family) => family.best_market_id).length;
    const warnings = data.families.filter((family) => family.warning_code).length;
    $('#prop-status').textContent = `${data.families.length} pitcher families · ${best} best bets · ${warnings} safety warnings · ${data.model_version}`;
    replaceHtmlWithFragment($('#prop-board'), data.families.length ? data.families.map(familyCard).join('') : '<div class="empty">No upcoming mapped strikeout families.</div>');
  } catch (error) {
    $('#prop-board').innerHTML = `<div class="empty negative">${esc(error.message)}</div>`;
    $('#prop-status').textContent = 'Player-prop refresh failed.';
  }
}

async function placeProp(familyKey, marketId, sideName) {
  const family = state.props?.families?.find((item) => item.family_key === familyKey);
  const contract = family?.contracts?.find((item) => item.market_id === marketId && item.side === sideName);
  if (!family || !contract || contract.decision !== 'BEST_BET') return;
  try {
    await api('/api/v1/paper/bets', {
      method: 'POST',
      body: JSON.stringify({
        account_type: 'PLAYER_PROP', market_type: 'STRIKEOUT_PROP',
        game_pk: family.game_pk, market_id: contract.market_id, token_id: contract.token_id,
        selection: `${family.player_name} ${contract.threshold}+ K ${contract.side}`,
        opponent: family.opponent, player_id: family.player_id, player_name: family.player_name,
        prop_threshold: contract.threshold, prop_side: contract.side,
        model_version: family.model_version, model_probability: contract.conservative_probability,
        entry_price: contract.executable_price, entry_edge: contract.edge,
        expected_roi: contract.expected_roi, stake: contract.proposed_stake,
        start_time: family.start_time, polymarket_url: contract.polymarket_url,
        metadata: { expected_strikeouts: family.expected_strikeouts, raw_probability: contract.raw_probability, shrinkage: family.shrinkage }
      })
    });
    toast(`Best prop placed: ${family.player_name} ${contract.threshold}+ K ${contract.side}.`);
    state.loaded.delete('portfolio');
    await loadSummary();
    await loadProps(true);
  } catch (error) { toast(error.message); }
}

function betRow(bet) {
  const selection = bet.market_type === 'STRIKEOUT_PROP'
    ? `${bet.player_name} ${bet.prop_threshold}+ K · ${bet.prop_side}`
    : bet.selection;
  const pnl = bet.realized_pnl == null ? '—' : money(bet.realized_pnl);
  return `<tr><td><strong>${esc(selection)}</strong><br><span class="section-copy">${localTime(bet.start_time || bet.placed_at)}</span></td><td>${esc(bet.account_type)}</td><td class="mono">${dollars(bet.stake)}</td><td class="mono">${cents(bet.entry_price)}</td><td class="mono">${pct(bet.entry_edge)}</td><td>${esc(bet.status)}</td><td class="mono ${Number(bet.realized_pnl) >= 0 ? 'positive' : 'negative'}">${pnl}</td></tr>`;
}

async function loadPortfolio(force = false) {
  if (state.loaded.has('portfolio') && !force) return;
  $('#portfolio-cards').innerHTML = '<div class="skeleton card-skeleton"></div><div class="skeleton card-skeleton"></div>';
  $('#bet-history').innerHTML = '<div class="skeleton block"></div>';
  try {
    const [accounts, bets] = await Promise.all([api('/api/v1/paper/accounts'), api('/api/v1/paper/bets')]);
    state.loaded.add('portfolio');
    state.bets = bets.bets;
    replaceHtmlWithFragment($('#portfolio-cards'), accountCard(accounts.moneyline, 'MONEYLINE') + accountCard(accounts.player_props, 'PLAYER_PROP'));
    replaceHtmlWithFragment($('#bet-history'), bets.bets.length ? `<table><thead><tr><th>Selection</th><th>Ledger</th><th>Stake</th><th>Price</th><th>Edge</th><th>Status</th><th>P&amp;L</th></tr></thead><tbody>${bets.bets.map(betRow).join('')}</tbody></table>` : '<div class="empty">No Postgres paper bets yet.</div>');
  } catch (error) { $('#bet-history').innerHTML = `<div class="empty negative">${esc(error.message)}</div>`; }
}

async function loadResearch(force = false) {
  if (state.loaded.has('research') && !force) return;
  $('#research-content').innerHTML = '<div class="skeleton block"></div>';
  try {
    const data = await api('/api/v1/edge-performance/mlb?entry_horizon_minutes=60');
    state.loaded.add('research');
    const rows = data.metrics.map((item) => `<tr><td>${esc(item.strategy)}</td><td>${pct(item.min_edge)}</td><td>${item.eligible_signals}</td><td>${item.settled_bets}</td><td>${pct(item.win_rate)}</td><td>${pct(item.flat_stake_roi)}</td><td>${item.brier_score == null ? '—' : Number(item.brier_score).toFixed(3)}</td></tr>`).join('');
    replaceHtmlWithFragment($('#research-content'), `<div class="card table-card"><div class="table-shell"><table><thead><tr><th>Strategy</th><th>Min edge</th><th>Signals</th><th>Settled</th><th>Win rate</th><th>Flat ROI</th><th>Brier</th></tr></thead><tbody>${rows}</tbody></table></div></div>`);
  } catch (error) { $('#research-content').innerHTML = `<div class="empty negative">${esc(error.message)}</div>`; }
}

async function loadModel(force = false) {
  if (state.loaded.has('model') && !force) return;
  $('#model-content').innerHTML = '<div class="skeleton block"></div>';
  try {
    const season = new Date().getFullYear();
    const data = await api(`/api/v1/backtest/mlb?season=${season}&min_games=10`);
    state.loaded.add('model');
    const champion = data.optimized;
    $('#model-content').innerHTML = `<div class="portfolio-grid"><article class="portfolio-card"><span class="account-tag">SELECTED CHAMPION</span><div class="account-equity">${esc(data.tournament_champion || data.optimized_variant)}</div><div class="metric-grid"><div class="metric"><span>Accuracy</span><strong>${pct(champion.accuracy)}</strong></div><div class="metric"><span>Brier</span><strong>${champion.brier_score == null ? '—' : Number(champion.brier_score).toFixed(3)}</strong></div><div class="metric"><span>Log loss</span><strong>${champion.log_loss == null ? '—' : Number(champion.log_loss).toFixed(3)}</strong></div><div class="metric"><span>Predictions</span><strong>${champion.prediction_count}</strong></div></div></article><article class="portfolio-card"><span class="account-tag">LEAKAGE AUDIT</span><div class="account-equity">${data.leakage_audit.passed ? 'PASS' : 'FAIL'}</div><p class="section-copy">${esc(data.methodology)}</p></article></div>`;
  } catch (error) { $('#model-content').innerHTML = `<div class="empty negative">${esc(error.message)}</div>`; }
}

async function activateTab(name) {
  $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  $$('.panel-view').forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === name));
  history.replaceState(null, '', `#${name}`);
  if (name === 'moneylines') await loadMoneylines();
  if (name === 'props') await loadProps();
  if (name === 'portfolio') await loadPortfolio();
  if (name === 'research') await loadResearch();
  if (name === 'model') await loadModel();
}

document.addEventListener('click', (event) => {
  const tab = event.target.closest('.tab');
  if (tab) activateTab(tab.dataset.tab);
  const moneyline = event.target.closest('.moneyline-bet');
  if (moneyline) placeMoneyline(moneyline.dataset.game);
  const prop = event.target.closest('.prop-bet');
  if (prop) placeProp(prop.dataset.family, prop.dataset.market, prop.dataset.side);
});

$('#refresh-moneylines').addEventListener('click', () => loadMoneylines(true));
$('#refresh-props').addEventListener('click', () => loadProps(true));
$('#refresh-portfolio').addEventListener('click', () => loadPortfolio(true));
$('#refresh-research').addEventListener('click', () => loadResearch(true));
$('#refresh-model').addEventListener('click', () => loadModel(true));

(async function boot() {
  try {
    await loadSummary();
    const requested = location.hash.replace('#', '');
    if (['overview', 'moneylines', 'props', 'portfolio', 'research', 'model'].includes(requested)) {
      await activateTab(requested);
    }
  } catch (error) {
    toast(error.message);
    $('#freshness').textContent = 'Summary unavailable';
  }
})();
