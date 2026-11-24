/* =====================================================
   UI.JS — Helper JS bersama untuk semua halaman.
   Dimuat oleh base.html SEBELUM block extra_js.
   ===================================================== */

// ---------- 1. ESCAPING ----------
function escapeHtml(t) {
    if (t === null || t === undefined) return '';
    const d = document.createElement('div');
    d.textContent = String(t);
    return d.innerHTML;
}

// Untuk nilai di dalam atribut HTML (aman terhadap kutip ' dan ")
function escapeAttr(t) {
    return String(t === null || t === undefined ? '' : t)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ---------- 2. HIGHLIGHT TEKS HASIL PENCARIAN ----------
function highlightText(rawText, rawQuery) {
    if (!rawText) return '-';
    let s = escapeHtml(rawText);
    if (!rawQuery) return s;
    let terms = rawQuery.trim().split(/\s+/).filter(t => t.length > 0);
    if (terms.length === 0) return s;
    terms = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    return s.replace(new RegExp(`(${terms.join('|')})`, 'gi'), '<span class="highlight-text">$1</span>');
}

// ---------- 3. TOGGLE TEMA (light/dark) ----------
function initThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    const icon = document.getElementById('theme-icon');
    if (!btn || !icon) return;
    const updateIcon = (theme) => {
        if (theme === 'dark') {
            icon.className = 'bi bi-sun-fill text-warning';
            btn.classList.replace('btn-outline-secondary', 'btn-outline-warning');
        } else {
            icon.className = 'bi bi-moon-stars-fill text-navy';
            btn.classList.replace('btn-outline-warning', 'btn-outline-secondary');
        }
    };
    updateIcon(document.documentElement.getAttribute('data-bs-theme'));
    btn.addEventListener('click', () => {
        const newTheme = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-bs-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateIcon(newTheme);
        // Halaman dengan chart bisa mendefinisikan onThemeChange() untuk reload
        if (typeof window.onThemeChange === 'function') window.onThemeChange();
    });
}
document.addEventListener('DOMContentLoaded', initThemeToggle);

// ---------- 4. GANTI BATCH EVALUASI ----------
function changeBatch(val, basePath) {
    if (val) window.location.href = basePath + '?batch=' + encodeURIComponent(val);
}

// ---------- 5. EXPORT TABEL KE EXCEL (.xls via HTML) ----------
function exportToExcelStyled(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) return;
    const html = `
        <html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
        <head><meta charset="utf-8"></head>
        <body>${table.outerHTML}</body>
        </html>`;
    const blob = new Blob([html], { type: 'application/vnd.ms-excel' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// ---------- 6. TAB IMPORT (File / Teks) ----------
// Halaman cukup mendefinisikan fungsi global processImportedText(text).
function switchImportTab(mode, btn) {
    document.querySelectorAll('.import-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('#tab-file, #tab-text').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = document.getElementById('imp-' + mode);
    if (target) target.style.display = 'block';
}

function handleFileImport(input) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            if (typeof window.processImportedText === 'function') window.processImportedText(e.target.result);
        };
        reader.readAsText(file);
    }
    input.value = '';
}