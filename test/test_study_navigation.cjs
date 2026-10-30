// Exercise navigation state without requiring a browser installation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(text = '') { this.textContent = text; this.dataset = {}; this.events = {}; this.children = []; this.selectedIndex = -1; }
  addEventListener(event, fn) { this.events[event] = fn; }
  fire(event) { this.events[event]?.(); }
  setAttribute(key, value) { this[key] = value; }
  replaceChildren(...nodes) { this.children = nodes; this.selectedIndex = nodes.length ? 0 : -1; }
  get value() { return this.children[this.selectedIndex]?.value ?? ''; }
  set value(value) { this.selectedIndex = this.children.findIndex(node => node.value === value); }
  querySelector() { return new Element(this.textContent); }
}
const figure = (title, group) => {
  const node = new Element(title);
  if (group) Object.assign(node.dataset, {chartGroup: group, chartLabel: group});
  return node;
};
const section = (id, title, figures) => {
  const node = new Element(title); node.id = id;
  node.querySelectorAll = () => figures; node.figures = figures;
  return node;
};
const sections = [
  section('geometry', 'Guide', []),
  section('03-shared_genradius_family', 'Family', [figure('Saddle'), figure('Center'), figure('Area')]),
  section('training-history', 'Training', [figure('PS global', 'global'), figure('EV global', 'global'), figure('PS saddle', 'saddle'), figure('EV saddle', 'saddle')]),
  section('training-diagnostics', 'Diagnostics', [figure('PS supervision', 'supervision'), figure('EV supervision', 'supervision')]),
];
const buttons = ['overview', 'generalization', 'training', 'diagnostics'].map(view => {
  const button = new Element(); button.dataset.studyView = view; return button;
});
const controls = Object.fromEntries(['study-controls', 'experiment-select', 'chart-select', 'show-all', 'figure-status', 'previous-chart', 'next-chart', 'chart-controls', 'chart-label', 'experiment-field', 'comparison-key', 'fallback-navigation'].map(id => [id, new Element()]));
const events = {};
const location = {hash: ''};
const context = {
  document: {
    getElementById: id => controls[id],
    querySelectorAll: query => query === 'body > section' ? sections : query === '[data-study-view]' ? buttons : sections.flatMap(s => s.figures),
  },
  window: {addEventListener: (event, fn) => { events[event] = fn; }},
  Option: function(text, value) { this.textContent = text; this.value = value; },
  location, history: {replaceState: (_, __, hash) => { location.hash = hash; }},
};
vm.runInNewContext(fs.readFileSync(require('node:path').join(__dirname, '../scripts/charts/study_navigation.js'), 'utf8'), context);
const visible = () => sections.filter(s => !s.hidden).flatMap(s => s.figures.filter(f => !f.hidden)).map(f => f.textContent);
assert.deepEqual(visible(), ['Saddle']);
controls['next-chart'].fire('click');
assert.deepEqual(visible(), ['Center']);
buttons.find(b => b.dataset.studyView === 'training').fire('click');
assert.deepEqual(visible(), ['PS global', 'EV global']);
controls['next-chart'].fire('click');
assert.deepEqual(visible(), ['PS saddle', 'EV saddle']);
controls['show-all'].checked = true; controls['show-all'].fire('change');
assert.equal(visible().length, 4);
assert.equal(controls['next-chart'].disabled, true);
controls['show-all'].checked = false; controls['show-all'].fire('change');
buttons.find(b => b.dataset.studyView === 'generalization').fire('click');
assert.deepEqual(visible(), ['Center']); // Preserve the selected chart when returning.
location.hash = '#training-diagnostics'; events.hashchange();
assert.deepEqual(visible(), ['PS supervision', 'EV supervision']);
assert.equal(controls['comparison-key'].hidden, false);
events.beforeprint(); assert.equal(visible().length, 9);
events.afterprint(); assert.deepEqual(visible(), ['PS supervision', 'EV supervision']);
buttons.find(b => b.dataset.studyView === 'overview').fire('click');
assert.equal(controls['chart-controls'].hidden, true);
assert.equal(sections[0].hidden, false);
console.log('Navigation passed: views, chart stepping, paired models, show-all, remembered selection, deep links, print.');
