'use strict';
const search = document.getElementById('filter');
const rows = Array.from(document.querySelectorAll('#results-table tbody tr'));
search.addEventListener('input', () => {
  const q = search.value.trim().toLowerCase();
  let visible = 0;
  for (const row of rows) {
    const show = row.dataset.benchmark.includes(q);
    row.hidden = !show;
    visible += Number(show);
  }
  document.getElementById('empty').style.display = visible ? 'none' : 'block';
  document.getElementById('row-status').textContent = visible + ' benchmarks shown';
});
