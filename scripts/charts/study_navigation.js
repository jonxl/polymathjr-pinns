// Progressive enhancement: the full report remains readable without JavaScript.
(() => {
  const sections = [...document.querySelectorAll('body > section')];
  const controls = document.getElementById('study-controls');
  const run = document.getElementById('run-select');
  let rememberedRun = '';
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
  let allGroups = [];
  const chartType = document.getElementById('chart-type-select');
  const region = document.getElementById('region-select');
  const rememberedFilters = new Map();
  const regionNames = ['unstable_spiral', 'stable_spiral', 'unstable_node', 'stable_node', 'saddle', 'center', 'origin', 'global'];
  function facets(label) {
    const text = label.toLowerCase().replaceAll(' ', '_');
    const family = regionNames.find(name => text.includes(name)) || 'summary';
    const types = [['local-plane|near_training_cloud', 'Local error maps'], ['plane|heatmap', 'Broad error maps'], ['directional.reach', 'Directional reach'], ['shell|radial', 'Radial errors'], ['area', 'Accurate area'], ['training.solution|training_equations', 'Training fit'], ['parabola', 'Parabola'], ['transfer|probe_errors', 'Transfer'], ['reach|margin', 'Reach summaries']];
    return {family, type: view === 'training' ? 'Loss histories' : types.find(([pattern]) => new RegExp(pattern).test(text))?.[1] || 'Other charts'};
  }
  function filterCharts() {
    allGroups.forEach(group => group.nodes.forEach(node => { node.hidden = true; }));
    groups = allGroups.filter(group => (!chartType.value || group.type === chartType.value) && (!region.value || group.family === region.value));
    rememberedFilters.set(activeSection.id, [chartType.value, region.value]);
    options(chart, groups.map((group, i) => [String(i), group.label]));
    chart.value = rememberedCharts.get(activeSection.id) || '0';
    if (chart.selectedIndex < 0 && groups.length) chart.selectedIndex = 0;
    showCharts();
  }
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
      `Showing all ${groups.length} chart groups` : `${index + 1} of ${groups.length} · ${selected.label}`) : 'No charts match these filters.';
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
    allGroups = [...grouped.values()].map(group => ({...group, ...facets(group.nodes[0].dataset.chartId || group.label)}));
    options(chartType, [['', 'All chart types'], ...[...new Set(allGroups.map(g => g.type))].map(type => [type, type])]);
    options(region, [['', 'All regions / summaries'], ...[...new Set(allGroups.map(g => g.family))].map(name => [name, name.replaceAll('_', ' ')])]);
    const saved = rememberedFilters.get(activeSection.id) || ['', ''];
    chartType.value = saved[0]; region.value = saved[1];
    filterCharts();
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
    let available = sections.filter(section => classify(section) === view);
    document.getElementById('run-field').hidden = view !== 'generalization';
    if (view === 'generalization') {
      const folders = [...new Set(available.map(s => s.dataset.runGroup || 'Earlier saved experiments'))];
      options(run, [['', 'All experiment folders'], ...folders.map(name => [name, name])]);
      const linkedSection = available.find(s => s.id === sectionId);
      if (linkedSection) rememberedRun = linkedSection.dataset.runGroup || 'Earlier saved experiments';
      run.value = rememberedRun;
      if (run.selectedIndex < 0) run.value = '';
      available = available.filter(s => !run.value || (s.dataset.runGroup || 'Earlier saved experiments') === run.value);
    }
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
  run.addEventListener('change', () => { rememberedRun = run.value; showView('generalization'); });
  experiment.addEventListener('change', () => showSection());
  chart.addEventListener('change', showCharts);
  chartType.addEventListener('change', () => { rememberedCharts.delete(activeSection.id); filterCharts(); });
  region.addEventListener('change', () => { rememberedCharts.delete(activeSection.id); filterCharts(); });
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
