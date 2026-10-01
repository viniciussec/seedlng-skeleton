// ==============================================================================
// Logic & Interactivity — Seedling Skeleton Studio
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const samplesList = document.getElementById("samples-list");
  const pxPerCmInput = document.getElementById("px-per-cm");
  const minCompSizeInput = document.getElementById("min-comp-size");
  const reprocessBtn = document.getElementById("reprocess-btn");
  
  const loadingState = document.getElementById("loading-state");
  const resultsSection = document.getElementById("results-section");
  
  const gtOriginalMask = document.getElementById("gt-original-mask");
  const gtBinaryHyp = document.getElementById("gt-binary-hyp");
  const gtFilename = document.getElementById("gt-filename");
  const gtDimensions = document.getElementById("gt-dimensions");
  const gtHypPixels = document.getElementById("gt-hyp-pixels");
  const gtHypComponents = document.getElementById("gt-hyp-components");
  const gtRootPixels = document.getElementById("gt-root-pixels");
  const gtRootComponents = document.getElementById("gt-root-components");

  const algoCardsContainer = document.getElementById("algo-cards-container");
  const fourPanelDisplay = document.getElementById("four-panel-display");
  const modelAlgoSelect = document.getElementById("model-algo-select");
  const metricsTableBody = document.getElementById("metrics-table-body");
  const exportCsvBtn = document.getElementById("export-csv-btn");

  const zoomModal = document.getElementById("zoom-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalImg = document.getElementById("modal-img");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalDownloadBtn = document.getElementById("modal-download-btn");
  const modalViewport = document.getElementById("modal-viewport");
  const zoomLevelText = document.getElementById("zoom-level-text");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const zoomResetBtn = document.getElementById("zoom-reset-btn");

  // State
  let currentResults = null;
  let currentPayload = { sample: null, image: null };
  let selectedSampleName = null;
  let zoomScale = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;

  function updateModalTransform() {
    if (!modalImg) return;
    modalImg.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
    if (zoomLevelText) {
      zoomLevelText.textContent = `${Math.round(zoomScale * 100)}%`;
    }
  }

  function resetModalZoom() {
    zoomScale = 1.0;
    panX = 0;
    panY = 0;
    updateModalTransform();
  }

  const organFilterState = {
    poda: "composite",
    zhang_suen: "composite",
    lee: "composite",
    mat_puro: "composite"
  };

  // 1. Load Samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();
      samplesList.innerHTML = "";
      
      if (!samples || samples.length === 0) {
        samplesList.innerHTML = `<span class="text-muted">Nenhuma amostra encontrada em labeled-dataset.</span>`;
        return;
      }

      // Show up to 14 samples
      const displaySamples = samples.slice(0, 14);
      displaySamples.forEach((s, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `sample-pill ${idx === 0 ? "active" : ""}`;
        btn.textContent = s.name.replace(".png", "").replace(".jpg", "");
        btn.title = s.name;
        btn.addEventListener("click", () => {
          document.querySelectorAll(".sample-pill").forEach(p => p.classList.remove("active"));
          btn.classList.add("active");
          processSample(s.name);
        });
        samplesList.appendChild(btn);
      });

      // Automatically process first sample on load
      if (displaySamples.length > 0) {
        processSample(displaySamples[0].name);
      }
    } catch (err) {
      samplesList.innerHTML = `<span class="text-muted">Falha ao carregar amostras locais. Use o upload acima.</span>`;
    }
  }

  // 2. Process API Call
  async function processSample(sampleName) {
    selectedSampleName = sampleName;
    currentPayload = {
      sample: sampleName
    };
    await sendProcessRequest();
  }

  async function processUploadedImage(base64Data, filename) {
    selectedSampleName = filename || "Imagem personalizada";
    currentPayload = {
      image: base64Data
    };
    document.querySelectorAll(".sample-pill").forEach(p => p.classList.remove("active"));
    await sendProcessRequest();
  }

  async function sendProcessRequest() {
    loadingState.style.display = "block";
    resultsSection.style.display = "none";
    
    try {
      const payload = {
        ...currentPayload,
        pixels_per_cm: parseFloat(pxPerCmInput.value) || 100.0,
        min_component_size: parseInt(minCompSizeInput.value) || 30
      };

      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Erro ao processar imagem");
      }

      currentResults = await res.json();
      renderResults(currentResults);
    } catch (err) {
      alert("Erro no processamento: " + err.message);
    } finally {
      loadingState.style.display = "none";
      resultsSection.style.display = "block";
    }
  }

  // 3. Render Results
  function renderResults(data) {
    // Top Bar Metadata
    gtOriginalMask.src = data.original_mask;
    gtBinaryHyp.src = data.binary_hypocotyl;
    gtFilename.textContent = selectedSampleName || "Amostra";
    gtDimensions.textContent = `${data.metadata.width} × ${data.metadata.height} px`;
    gtHypPixels.textContent = `${data.metadata.hypocotyl_pixels.toLocaleString()} px`;
    gtHypComponents.textContent = `${data.metadata.hypocotyl_components} componentes`;
    gtRootPixels.textContent = `${data.metadata.root_pixels.toLocaleString()} px`;
    gtRootComponents.textContent = `${data.metadata.root_components} componentes`;

    // Render Grid Cards
    renderAlgorithmCards(data.algorithms);

    // Render 4-Panel Model View
    renderFourPanelView(modelAlgoSelect.value);

    // Render Metrics Table
    renderMetricsTable(data.algorithms);
  }

  function renderAlgorithmCards(algorithms) {
    algoCardsContainer.innerHTML = "";

    const keys = ["poda", "zhang_suen", "lee", "mat_puro"];
    
    keys.forEach(key => {
      const algo = algorithms[key];
      if (!algo) return;

      const isHighlight = key === "poda";
      const card = document.createElement("div");
      card.className = `algo-card ${isHighlight ? "highlight" : ""}`;
      card.id = `card-${key}`;

      const activeFilter = organFilterState[key] || "composite";
      let displayImg = algo.overlay_composite;
      if (activeFilter === "hyp") displayImg = algo.overlay_hyp;
      if (activeFilter === "root") displayImg = algo.overlay_root;

      card.innerHTML = `
        <div class="algo-card-header">
          <div class="algo-card-title">
            <h3>${algo.name}</h3>
            <span class="badge ${isHighlight ? "badge-gold" : "badge-primary"}">${algo.tag}</span>
          </div>
          <div class="algo-time">⚡ ${algo.time_ms} ms</div>
        </div>

        <div class="algo-preview-box" data-key="${key}" title="Clique para ampliar o esqueleto">
          <img id="img-${key}" src="${displayImg}" alt="${algo.name}">
          <button type="button" class="zoom-overlay-btn" title="Ampliar Esqueleto" data-key="${key}">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
          </button>
        </div>

        <div class="algo-organ-toggles">
          <button type="button" class="organ-toggle-btn ${activeFilter === 'composite' ? 'active' : ''}" data-key="${key}" data-mode="composite">Ambos (H+R)</button>
          <button type="button" class="organ-toggle-btn ${activeFilter === 'hyp' ? 'active' : ''}" data-key="${key}" data-mode="hyp">Hipocótilo</button>
          <button type="button" class="organ-toggle-btn ${activeFilter === 'root' ? 'active' : ''}" data-key="${key}" data-mode="root">Raiz</button>
        </div>

        <div class="algo-stats">
          <div class="stat-item">
            <span class="stat-label">Comprimento Total</span>
            <span class="stat-value highlight-val">${algo.total.length_cm} cm</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Pixels do Esqueleto</span>
            <span class="stat-value">${algo.total.pixels} px</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Bifurcações (Spurs)</span>
            <span class="stat-value">${algo.total.branchpoints}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Hipocótilo / Raiz</span>
            <span class="stat-value" style="font-size: 0.95rem;">${algo.hypocotyl.length_cm} / ${algo.root.length_cm} cm</span>
          </div>
        </div>

        <div class="algo-footer">
          <p>${algo.description}</p>
        </div>
      `;

      algoCardsContainer.appendChild(card);
    });

    // Event listeners for organ filters
    document.querySelectorAll(".organ-toggle-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.key;
        const mode = btn.dataset.mode;
        organFilterState[key] = mode;

        const parentCard = document.getElementById(`card-${key}`);
        parentCard.querySelectorAll(".organ-toggle-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        const targetImg = document.getElementById(`img-${key}`);
        const algo = currentResults.algorithms[key];
        if (mode === "composite") targetImg.src = algo.overlay_composite;
        if (mode === "hyp") targetImg.src = algo.overlay_hyp;
        if (mode === "root") targetImg.src = algo.overlay_root;
      });
    });

    // Zoom modal: Click on either preview box or the zoom icon
    document.querySelectorAll(".algo-preview-box").forEach(box => {
      box.addEventListener("click", () => {
        const key = box.dataset.key;
        openZoomModal(key);
      });
    });
  }

  function openZoomModal(key) {
    if (!currentResults) return;
    const algo = currentResults.algorithms[key];
    if (!algo) return;
    const mode = organFilterState[key] || "composite";
    
    let src = algo.overlay_composite;
    if (mode === "hyp") src = algo.overlay_hyp;
    if (mode === "root") src = algo.overlay_root;

    const organLabel = mode === "composite" ? "AMBOS (H+R)" : mode === "hyp" ? "HIPOCÓTILO" : "RAIZ";
    modalTitle.textContent = `${algo.name} — Detalhe do Esqueleto (${organLabel})`;
    modalImg.src = src;
    modalDownloadBtn.href = src;
    modalDownloadBtn.download = `esqueleto_${key}_${mode}.png`;
    resetModalZoom();
    zoomModal.style.display = "flex";
  }

  function renderFourPanelView(algoKey) {
    if (!currentResults) return;
    const algo = currentResults.algorithms[algoKey];
    if (!algo) return;

    fourPanelDisplay.innerHTML = `
      <div class="panel-col">
        <div class="panel-header">
          <h4>1. Máscara Anotada Original</h4>
        </div>
        <div class="panel-img-box">
          <img src="${currentResults.original_mask}" alt="Original">
        </div>
      </div>

      <div class="panel-col">
        <div class="panel-header">
          <h4>2. Hipocótilo Binário Limpo</h4>
        </div>
        <div class="panel-img-box">
          <img src="${currentResults.binary_hypocotyl}" alt="Binário Hipocótilo">
        </div>
      </div>

      <div class="panel-col">
        <div class="panel-header">
          <h4>3. Hipocótilo + Esqueleto 1 px (Amarelo)</h4>
        </div>
        <div class="panel-img-box">
          <img src="${algo.overlay_hyp}" alt="Esqueleto Hipocótilo">
        </div>
      </div>

      <div class="panel-col">
        <div class="panel-header">
          <h4>4. Raiz + Esqueleto 1 px (Ciano)</h4>
        </div>
        <div class="panel-img-box">
          <img src="${algo.overlay_root}" alt="Esqueleto Raiz">
        </div>
      </div>
    `;
  }

  function renderMetricsTable(algorithms) {
    metricsTableBody.innerHTML = "";
    
    Object.keys(algorithms).forEach(key => {
      const a = algorithms[key];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${a.name}</strong></td>
        <td><span class="badge ${key === 'poda' ? 'badge-gold' : 'badge-primary'}">${a.tag}</span></td>
        <td>${a.hypocotyl.length_cm} cm (${a.hypocotyl.pixels} px)</td>
        <td>${a.root.length_cm} cm (${a.root.pixels} px)</td>
        <td><strong style="color: var(--accent-yellow);">${a.total.length_cm} cm</strong></td>
        <td>${a.total.pixels} px</td>
        <td>${a.total.branchpoints}</td>
        <td><span style="color: var(--accent-emerald); font-weight:600;">${a.time_ms} ms</span></td>
      `;
      metricsTableBody.appendChild(tr);
    });
  }

  // 4. View Switcher Tabs
  document.querySelectorAll(".tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const view = tab.dataset.view;
      document.querySelectorAll(".view-panel").forEach(p => p.style.display = "none");
      
      if (view === "grid") document.getElementById("view-grid").style.display = "block";
      if (view === "model") document.getElementById("view-model").style.display = "block";
      if (view === "table") document.getElementById("view-table").style.display = "block";
    });
  });

  modelAlgoSelect.addEventListener("change", (e) => {
    renderFourPanelView(e.target.value);
  });

  // 5. Drag & Drop File Upload
  browseBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
  });

  ["dragenter", "dragover"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.match("image.*")) {
      alert("Por favor selecione um arquivo de imagem (PNG ou JPG).");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      processUploadedImage(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
  }

  // Reprocess button
  reprocessBtn.addEventListener("click", () => {
    if (currentPayload.sample || currentPayload.image) {
      sendProcessRequest();
    }
  });

  // Export CSV
  exportCsvBtn.addEventListener("click", () => {
    if (!currentResults) return;
    const rows = [
      ["Tecnica", "Tipo", "Hipocotilo_cm", "Raiz_cm", "Total_cm", "Pixels_Totais", "Bifurcacoes", "Tempo_ms"]
    ];

    Object.keys(currentResults.algorithms).forEach(k => {
      const a = currentResults.algorithms[k];
      rows.push([
        a.name,
        a.tag,
        a.hypocotyl.length_cm,
        a.root.length_cm,
        a.total.length_cm,
        a.total.pixels,
        a.total.branchpoints,
        a.time_ms
      ]);
    });

    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `metricas_esqueletizacao_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // Modal close handlers
  function closeModal() {
    zoomModal.style.display = "none";
    resetModalZoom();
  }

  modalCloseBtn.addEventListener("click", closeModal);

  zoomModal.addEventListener("click", (e) => {
    if (e.target === zoomModal) {
      closeModal();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && zoomModal.style.display === "flex") {
      closeModal();
    }
  });

  // ==============================================================================
  // Modal Interactive Zoom & Pan (Wheel Scroll + Click-and-Drag)
  // ==============================================================================
  if (modalViewport) {
    // 1. Zoom via Mouse Wheel
    modalViewport.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = modalViewport.getBoundingClientRect();
      const cursorX = e.clientX - rect.left - rect.width / 2;
      const cursorY = e.clientY - rect.top - rect.height / 2;
      
      const prevScale = zoomScale;
      // Scroll Up (deltaY < 0) = Zoom In, Scroll Down (deltaY > 0) = Zoom Out
      const factor = e.deltaY < 0 ? 1.18 : 0.847;
      zoomScale = Math.min(Math.max(zoomScale * factor, 0.4), 25.0);
      
      // Zoom focused on cursor position
      panX -= (cursorX - panX) * (zoomScale / prevScale - 1);
      panY -= (cursorY - panY) * (zoomScale / prevScale - 1);
      
      updateModalTransform();
    }, { passive: false });

    // 2. Drag / Pan via Mouse
    modalViewport.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return; // Left click only
      isDragging = true;
      dragStartX = e.clientX - panX;
      dragStartY = e.clientY - panY;
      modalViewport.style.cursor = "grabbing";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      panX = e.clientX - dragStartX;
      panY = e.clientY - dragStartY;
      updateModalTransform();
    });

    window.addEventListener("mouseup", () => {
      if (isDragging) {
        isDragging = false;
        if (modalViewport) modalViewport.style.cursor = "grab";
      }
    });

    // 3. Double-Click to toggle 3x zoom or reset to fit
    modalViewport.addEventListener("dblclick", () => {
      if (zoomScale > 1.25) {
        resetModalZoom();
      } else {
        zoomScale = 3.2;
        panX = 0;
        panY = 0;
        updateModalTransform();
      }
    });

    // 4. Touch support (Mobile / Trackpad gestures)
    let initialTouchDist = null;
    let initialTouchScale = 1.0;

    modalViewport.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        dragStartX = e.touches[0].clientX - panX;
        dragStartY = e.touches[0].clientY - panY;
      } else if (e.touches.length === 2) {
        isDragging = false;
        initialTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        initialTouchScale = zoomScale;
      }
    }, { passive: true });

    modalViewport.addEventListener("touchmove", (e) => {
      if (isDragging && e.touches.length === 1) {
        panX = e.touches[0].clientX - dragStartX;
        panY = e.touches[0].clientY - dragStartY;
        updateModalTransform();
      } else if (e.touches.length === 2 && initialTouchDist) {
        const currentDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        zoomScale = Math.min(Math.max(initialTouchScale * (currentDist / initialTouchDist), 0.4), 25.0);
        updateModalTransform();
      }
    }, { passive: true });

    modalViewport.addEventListener("touchend", () => {
      isDragging = false;
      initialTouchDist = null;
    });
  }

  // 5. Modal Toolbar Zoom Buttons
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      zoomScale = Math.min(zoomScale * 1.3, 25.0);
      updateModalTransform();
    });
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      zoomScale = Math.max(zoomScale / 1.3, 0.4);
      updateModalTransform();
    });
  }

  if (zoomResetBtn) {
    zoomResetBtn.addEventListener("click", () => {
      resetModalZoom();
    });
  }

  // Initialize
  loadSamples();
});
