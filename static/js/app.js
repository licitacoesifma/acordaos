/* ════════════════════════════════════════
       APP STATE & CONFIG
       ════════════════════════════════════════ */
    const API_BASE = window.location.origin;
    const state = {
        query: '',
        resultados: [],
        isLoading: false,
        searchStartTime: 0,
        offset: 0,
        currentAcordaoKey: null
    };

    /* ════════════════════════════════════════
       DOM REFERENCES
       ════════════════════════════════════════ */
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        heroSection: $('#hero-section'),
        searchInput: $('#search-input'),
        searchBtn: $('#search-btn'),
        searchSuggestions: $('#search-suggestions'),
        filtersSection: $('#filters-section'),
        filterAno: $('#filter-ano'),
        filterColegiado: $('#filter-colegiado'),
        filterTipo: $('#filter-tipo'),
        filterRelator: $('#filter-relator'),
        filterClear: $('#filter-clear'),
        loadingContainer: $('#loading-container'),
        loadingText: $('#loading-text'),
        emptyState: $('#empty-state'),
        emptyMessage: $('#empty-message'),
        resultsSection: $('#results-section'),
        resultsCount: $('#results-count'),
        resultsMeta: $('#results-meta'),
        resultsGrid: $('#results-grid'),
        modalOverlay: $('#modal-overlay'),
        modalContent: $('#modal-content'),
        modalTitleText: $('#modal-title-text'),
        modalBody: document.getElementById('modal-body'),
        modalClose: document.getElementById('modal-close'),
        headerStats: document.getElementById('header-stats'),
        loadMoreContainer: document.getElementById('load-more-container'),
        loadMoreBtn: document.getElementById('load-more-btn'),
        btnResumirIA: document.getElementById('btn-resumir-ia'),
        iaSummaryContainer: document.getElementById('ia-summary-container'),
        iaSummaryContent: document.getElementById('ia-summary-content')
    };

    /* ════════════════════════════════════════
       INIT — Load filters
       ════════════════════════════════════════ */
    async function init() {
        try {
            const res = await fetch(`${API_BASE}/api/filtros`, { cache: 'no-store' });
            const data = await res.json();

            // Populate filter dropdowns
            data.anos.forEach(a => {
                const opt = document.createElement('option');
                opt.value = a; opt.textContent = a;
                els.filterAno.appendChild(opt);
            });

            data.colegiados.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c; opt.textContent = c;
                els.filterColegiado.appendChild(opt);
            });

            data.tipos_processo.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t; opt.textContent = t;
                els.filterTipo.appendChild(opt);
            });

            data.relatores.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r; opt.textContent = r;
                els.filterRelator.appendChild(opt);
            });

            // Populate top assuntos dynamically
            if (data.top_assuntos && data.top_assuntos.length > 0) {
                const emojis = ['📌', '🔍', '⚖️', '📑', '💡', '📊', '🏛️', '💼'];
                els.searchSuggestions.innerHTML = '';
                data.top_assuntos.forEach((assunto, i) => {
                    let textLimit = assunto.length > 35 ? assunto.substring(0, 35) + '...' : assunto;
                    const btn = document.createElement('button');
                    btn.className = 'suggestion-chip';
                    btn.dataset.query = assunto;
                    btn.innerHTML = `${emojis[i % emojis.length]} ${textLimit}`;
                    els.searchSuggestions.appendChild(btn);
                });
            }

            els.headerStats.textContent = `${data.total_acordaos} acórdãos indexados`;
        } catch (e) {
            console.error('Failed to load filters:', e);
        }
    }

    /* ════════════════════════════════════════
       SEARCH
       ════════════════════════════════════════ */
    async function performSearch(isLoadMore = false) {
        if (!isLoadMore) {
            state.offset = 0;
            state.resultados = [];
            state.query = els.searchInput.value.trim();
        }
        
        const query = state.query;
        if (!query || state.isLoading) return;

        state.isLoading = true;
        state.searchStartTime = performance.now();

        // UI: compact hero, show loading
        if (!isLoadMore) {
            els.heroSection.classList.add('compact');
            els.filtersSection.classList.add('visible');
            els.resultsSection.style.display = 'none';
            els.emptyState.classList.remove('visible');
            els.loadingContainer.classList.add('visible');
        }
        
        els.searchBtn.disabled = true;

        // Animate loading messages
        const loadingMessages = [
            'Buscando acórdãos relevantes...',
            'A IA está analisando os resultados...',
            'Ranqueando por relevância semântica...',
            'Preparando os resultados...',
        ];
        let msgIndex = 0;
        const msgInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % loadingMessages.length;
            els.loadingText.textContent = loadingMessages[msgIndex];
        }, 2500);

        try {
            const res = await fetch(`${API_BASE}/api/buscar`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    ano: els.filterAno.value,
                    colegiado: els.filterColegiado.value,
                    tipo_processo: els.filterTipo.value,
                    relator: els.filterRelator.value,
                    offset: state.offset
                }),
            });

            const data = await res.json();
            clearInterval(msgInterval);

            const elapsed = ((performance.now() - state.searchStartTime) / 1000).toFixed(1);
            const novosResultados = data.resultados || [];
            state.resultados = isLoadMore ? [...state.resultados, ...novosResultados] : novosResultados;

            if (state.resultados.length === 0) {
                els.loadingContainer.classList.remove('visible');
                els.emptyState.classList.add('visible');
                els.emptyMessage.textContent = data.mensagem || 'Tente reformular sua busca usando termos diferentes.';
            } else {
                renderResults(elapsed, isLoadMore, novosResultados.length);
            }
        } catch (e) {
            clearInterval(msgInterval);
            console.error('Search error:', e);
            els.loadingContainer.classList.remove('visible');
            els.emptyState.classList.add('visible');
            els.emptyMessage.textContent = 'Erro ao conectar com o servidor. Verifique se o backend está rodando.';
        } finally {
            state.isLoading = false;
            els.searchBtn.disabled = false;
        }
    }

    /* ════════════════════════════════════════
       RENDER RESULTS
       ════════════════════════════════════════ */
    function renderResults(elapsed, isLoadMore, newResultsCount) {
        els.loadingContainer.classList.remove('visible');
        els.resultsSection.style.display = 'block';

        els.resultsCount.innerHTML = `<strong>${state.resultados.length}</strong> resultado${state.resultados.length !== 1 ? 's' : ''} encontrados`;
        els.resultsMeta.textContent = `${elapsed}s · IA + busca textual`;

        if (!isLoadMore) {
            els.resultsGrid.innerHTML = '';
        }

        const resultsToRender = isLoadMore ? state.resultados.slice(state.resultados.length - newResultsCount) : state.resultados;

        resultsToRender.forEach((r, i) => {
            const relevClass = r.relevancia >= 7 ? 'relevance-high' :
                              r.relevancia >= 4 ? 'relevance-medium' : 'relevance-low';

            const card = document.createElement('article');
            card.className = 'result-card';
            card.setAttribute('role', 'button');
            card.setAttribute('tabindex', '0');
            card.setAttribute('aria-label', `Ver detalhes: ${r.titulo}`);
            card.style.animationDelay = `${i * 0.05}s`;

            card.innerHTML = `
                <div class="card-top">
                    <h2 class="card-title">${escapeHtml(r.titulo)}</h2>
                    <div class="relevance-badge ${relevClass}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                        ${r.relevancia}/10
                    </div>
                </div>
                <div class="card-meta">
                    ${r.colegiado ? `<span class="meta-tag colegiado">${escapeHtml(r.colegiado)}</span>` : ''}
                    ${r.tipo_processo ? `<span class="meta-tag tipo">${escapeHtml(r.tipo_processo)}</span>` : ''}
                    ${r.relator ? `<span class="meta-tag">${escapeHtml(r.relator)}</span>` : ''}
                </div>
                ${r.assunto ? `<p class="card-summary">${escapeHtml(truncate(r.assunto, 200))}</p>` : ''}
                <div class="card-ai-insight">
                    <svg class="ai-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><circle cx="9" cy="14" r="1"/><circle cx="15" cy="14" r="1"/></svg>
                    <p>${escapeHtml(r.resumo)}</p>
                </div>
            `;

            card.addEventListener('click', () => openModal(r.key));
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openModal(r.key);
                }
            });

            els.resultsGrid.appendChild(card);
        });

        if (newResultsCount === 25) {
            els.loadMoreContainer.style.display = 'flex';
        } else {
            els.loadMoreContainer.style.display = 'none';
        }
    }

    /* ════════════════════════════════════════
       MODAL
       ════════════════════════════════════════ */
    async function openModal(key) {
        try {
            state.currentAcordaoKey = key;
            els.iaSummaryContainer.style.display = 'none';
            els.iaSummaryContent.innerHTML = '';
            els.btnResumirIA.innerHTML = '✨ Resumir com IA';
            els.btnResumirIA.disabled = false;

            const res = await fetch(`${API_BASE}/api/acordao/${encodeURIComponent(key)}`);
            const ac = await res.json();

            if (ac.error) {
                alert('Acórdão não encontrado.');
                return;
            }

            els.modalTitleText.textContent = ac.titulo;

            els.modalBody.innerHTML = `
                <div class="modal-section">
                    <h3 class="modal-section-title">Informações Gerais</h3>
                    <div class="modal-detail-grid">
                        <div class="detail-item">
                            <span class="detail-label">Número</span>
                            <span class="detail-value">${escapeHtml(ac.numero)}/${escapeHtml(ac.ano)}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Colegiado</span>
                            <span class="detail-value">${escapeHtml(ac.colegiado)}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Relator</span>
                            <span class="detail-value">${escapeHtml(ac.relator)}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Tipo de Processo</span>
                            <span class="detail-value">${escapeHtml(ac.tipo_processo)}</span>
                        </div>
                    </div>
                </div>
                <div class="modal-section">
                    <h3 class="modal-section-title">Entidade</h3>
                    <p class="detail-value">${escapeHtml(ac.entidade)}</p>
                </div>
                <div class="modal-section">
                    <h3 class="modal-section-title">Assunto</h3>
                    <p class="detail-value">${escapeHtml(ac.assunto)}</p>
                </div>
                ${ac.sumario && ac.sumario !== 'NA' ? `
                <div class="modal-section">
                    <h3 class="modal-section-title">Sumário</h3>
                    <p class="detail-value">${escapeHtml(ac.sumario)}</p>
                </div>
                ` : ''}
                <div class="modal-section">
                    <h3 class="modal-section-title">Decisão</h3>
                    <div class="modal-decisao">${formatDecisao(ac.decisao)}</div>
                </div>
                <div class="modal-key">
                    🔑 ${escapeHtml(ac.key)}
                </div>
            `;

            els.modalOverlay.classList.add('visible');
            document.body.style.overflow = 'hidden';
            els.modalClose.focus();
        } catch (e) {
            console.error('Modal error:', e);
        }
    }

    function closeModal() {
        els.modalOverlay.classList.remove('visible');
        document.body.style.overflow = '';
    }

    /* ════════════════════════════════════════
       HELPERS
       ════════════════════════════════════════ */
    function escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function truncate(str, max) {
        if (!str || str.length <= max) return str;
        return str.substring(0, max) + '…';
    }

    function formatDecisao(text) {
        if (!text) return '<em>Não disponível</em>';
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/(9\.\d+\.?)/g, '<br><br><strong>$1</strong>');
        formatted = formatted.replace(/^(<br>)+/, '');
        return formatted;
    }

    /* ════════════════════════════════════════
       EVENT LISTENERS
       ════════════════════════════════════════ */

    // Search
    els.searchBtn.addEventListener('click', () => performSearch(false));
    els.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') performSearch(false);
    });

    els.loadMoreBtn.addEventListener('click', () => {
        state.offset += 25;
        performSearch(true);
    });

    // Suggestion chips
    els.searchSuggestions.addEventListener('click', (e) => {
        const chip = e.target.closest('.suggestion-chip');
        if (chip) {
            els.searchInput.value = chip.dataset.query;
            performSearch(false);
        }
    });

    // Filters
    els.filterClear.addEventListener('click', () => {
        els.filterAno.value = '';
        els.filterColegiado.value = '';
        els.filterTipo.value = '';
        els.filterRelator.value = '';
        if (state.query) performSearch(false);
    });

    [els.filterAno, els.filterColegiado, els.filterTipo, els.filterRelator].forEach(el => {
        el.addEventListener('change', () => {
            if (state.query) performSearch(false);
        });
    });

    els.btnResumirIA.addEventListener('click', async () => {
        if (!state.currentAcordaoKey) return;
        
        els.btnResumirIA.disabled = true;
        els.btnResumirIA.innerHTML = '<span class="loading-spinner" style="width: 14px; height: 14px; border-width: 2px;"></span> Analisando...';
        els.iaSummaryContainer.style.display = 'block';
        els.iaSummaryContent.innerHTML = '<i>Lendo o acórdão e gerando resumo, aguarde...</i>';

        try {
            const res = await fetch(`${API_BASE}/api/resumir`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chave: state.currentAcordaoKey })
            });
            
            const data = await res.json();
            if (res.ok) {
                els.iaSummaryContent.innerHTML = formatDecisao(data.resumo);
                els.btnResumirIA.innerHTML = '✨ Resumo Concluído';
            } else {
                els.iaSummaryContent.innerHTML = `<span style="color: var(--color-danger)">Erro: ${data.detail || 'Falha ao resumir'}</span>`;
                els.btnResumirIA.innerHTML = '✨ Tentar Novamente';
                els.btnResumirIA.disabled = false;
            }
        } catch (e) {
            console.error(e);
            els.iaSummaryContent.innerHTML = `<span style="color: var(--color-danger)">Erro de conexão com o servidor.</span>`;
            els.btnResumirIA.innerHTML = '✨ Tentar Novamente';
            els.btnResumirIA.disabled = false;
        }
    });

    // Initialize
    init();

    // Modal
    els.modalClose.addEventListener('click', closeModal);
    els.modalOverlay.addEventListener('click', (e) => {
        if (e.target === els.modalOverlay) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // Logo click — reset
    $('#logo-link').addEventListener('click', (e) => {
        e.preventDefault();
        els.heroSection.classList.remove('compact');
        els.filtersSection.classList.remove('visible');
        els.resultsSection.style.display = 'none';
        els.emptyState.classList.remove('visible');
        els.loadingContainer.classList.remove('visible');
        els.searchInput.value = '';
        state.query = '';
        state.resultados = [];
        els.searchInput.focus();
    });

    // Focus search on page load
    els.searchInput.focus();