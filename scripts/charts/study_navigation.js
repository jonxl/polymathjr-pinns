// Progressive enhancement: the full report remains readable without JavaScript.
(() => {
  const sections = [...document.querySelectorAll('body > section')];
  const controls = document.getElementById('study-controls');
  const experiment = document.getElementById('experiment-select');
  const chart = document.getElementById('chart-select');
  const all = document.getElementById('show-all');
  const status = document.getElementById('figure-status');
  const previous = document.getElementById('previous-chart');
  const next = document.getElementById('next-chart');
  const buttons = [...document.querySelectorAll('[data-study-view]')];
  const classify = section => section.id === 'geometry' ? 'overview' :
    section.id === 'training-history' ? 'training' :
    section.id === 'training-diagnostics' ? 'diagnostics' : 'generalization';
  const rememberedSections = new Map();
  const rememberedCharts = new Map();
  let view = 'generalization';
  let groups = [];
  let activeSection;

  function options(select, items) {
    select.replaceChildren(...items.map(([value, label]) => new Option(label, value)));
  }

  function showCharts() {
    const index = Number(chart.value || 0);
    const selected = groups[index];
    for (const [i, group] of groups.entries()) {
      group.nodes.forEach(node => { node.hidden = !all.checked && i !== index; });
    }
    rememberedCharts.set(activeSection.id, chart.value);
    previous.disabled = all.checked || index === 0 || !groups.length;
    next.disabled = all.checked || index >= groups.length - 1;
    chart.disabled = all.checked;
    status.textContent = groups.length ? (all.checked ?
      `Showing all ${groups.length} chart groups` : `${index + 1} of ${groups.length} · ${selected.label}`) : '';
  }

  function showSection(updateHash = true) {
    activeSection = sections.find(section => section.id === experiment.value);
    if (!activeSection) return;
    sections.forEach(section => { section.hidden = section !== activeSection; });
    rememberedSections.set(view, activeSection.id);
    const items = [...activeSection.querySelectorAll(':scope > figure, :scope > .status, :scope > .figure-pair')];
    const grouped = new Map();
    for (const [i, node] of items.entries()) {
      const key = node.dataset.chartGroup || String(i);
      const label = node.dataset.chartLabel || node.querySelector('h3')?.textContent || `Figure ${i + 1}`;
      if (!grouped.has(key)) grouped.set(key, {label, nodes: []});
      grouped.get(key).nodes.push(node);
    }
    groups = [...grouped.values()];
    options(chart, groups.map((group, i) => [String(i), group.label]));
    chart.value = rememberedCharts.get(activeSection.id) || '0';
    if (chart.selectedIndex < 0 && groups.length) chart.selectedIndex = 0;
    document.getElementById('chart-controls').hidden = groups.length === 0;
    document.getElementById('chart-label').textContent = view === 'training' ? 'Base model / run' : 'Chart';
    document.getElementById('experiment-field').hidden = view !== 'generalization';
    document.getElementById('comparison-key').hidden = view === 'overview';
    showCharts();
    if (updateHash && location.hash !== `#${activeSection.id}`) {
      // replaceState works for a served page; location.replace also supports file://.
      try { history.replaceState(null, '', `#${activeSection.id}`); }
      catch { location.replace(`#${activeSection.id}`); }
    }
  }

  function showView(name, sectionId, updateHash = true) {
    view = name;
    const available = sections.filter(section => classify(section) === view);
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.studyView === view)));
    options(experiment, available.map(section => [section.id, section.querySelector('h2').textContent]));
    const preferred = sectionId || rememberedSections.get(view);
    if (available.some(section => section.id === preferred)) experiment.value = preferred;
    showSection(updateHash);
  }

  buttons.forEach(button => {
    button.hidden = !sections.some(section => classify(section) === button.dataset.studyView);
    button.addEventListener('click', () => showView(button.dataset.studyView));
  });
  experiment.addEventListener('change', () => showSection());
  chart.addEventListener('change', showCharts);
  all.addEventListener('change', showCharts);
  previous.addEventListener('click', () => { chart.selectedIndex--; showCharts(); });
  next.addEventListener('click', () => { chart.selectedIndex++; showCharts(); });
  window.addEventListener('hashchange', () => {
    const section = sections.find(s => `#${s.id}` === location.hash);
    if (section) showView(classify(section), section.id, false);
  });
  window.addEventListener('beforeprint', () => {
    sections.forEach(s => { s.hidden = false; });
    document.querySelectorAll('figure, .status, .figure-pair').forEach(node => { node.hidden = false; });
  });
  window.addEventListener('afterprint', () => showSection(false));
  document.getElementById('fallback-navigation').hidden = true;
  controls.hidden = false;
  const linked = sections.find(s => `#${s.id}` === location.hash);
  const initial = linked || sections.find(s => s.id.endsWith('shared_genradius_family')) || sections[0];
  showView(classify(initial), initial.id, false);
})();
