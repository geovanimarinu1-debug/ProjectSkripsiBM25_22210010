import hmac
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.security import check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('KAMUS_SECRET_KEY',
                                'kamus-bm25-localhost-session')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('KAMUS_COOKIE_SECURE', '0') == '1',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

CSV_FILE = 'kamus.csv'
BATCH_FILE = 'evaluasi_batch.json'
DRAFT_FILE = 'evaluasi_draft.json'
PER_PAGE_SEARCH = 20

FIELDS = ['Kata', 'Ejaan', 'Arti', 'Contoh_Manado', 'Contoh_Indonesia']

INDEXED_FIELDS = ['Kata', 'Arti', 'Contoh_Manado', 'Contoh_Indonesia']

K1_GRID = [round(x * 0.5, 1) for x in range(0, 7)]
B_GRID = [round(x * 0.2, 1) for x in range(0, 6)]

DEFAULT_TOP_N = 5

CHART_TYPES = {
    'mean', 'category_map_ndcg', 'ndcg_compare_per_query',
    'map_compare_per_query', 'ndcg_per_query', 'f1_score_per_query',
    'precision_per_query', 'recall_per_query', 'ap_score_per_query',
    'mrr_per_query', 'category',
}

CORRUPT_BATCHES = {
    'draft_1788964215710_1789125303',
    'draft_1788964215710_1789219943',
    'draft_1788964215710_1789221908',
}

_file_lock = threading.Lock()
_model_lock = threading.Lock()
_chart_lock = threading.Lock()


# ================= AUTENTIKASI ADMIN =================

if not (os.environ.get('KAMUS_ADMIN_HASH')
        or os.environ.get('KAMUS_ADMIN_PASSWORD')):
    print('[PERINGATAN] KAMUS_ADMIN_HASH belum diset. '
          'Password default sementara: kamus-admin')


def _cek_password(teks):
    """True jika password cocok. Prioritas hash, fallback password polos."""
    hashed = os.environ.get('KAMUS_ADMIN_HASH', '').strip()
    if hashed:
        return check_password_hash(hashed, teks)
    polos = os.environ.get('KAMUS_ADMIN_PASSWORD', 'kamus-admin')
    return hmac.compare_digest(str(teks), polos)


def _minta_json():
    return request.path.startswith('/api/') or request.is_json


def admin_required(view):
    """Hanya bisa diakses setelah login admin."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get('is_admin'):
            return view(*args, **kwargs)
        if _minta_json():
            return jsonify({'success': False,
                            'error': 'Akses ditolak. Login admin dulu.'}), 401
        return redirect(url_for('login', next=request.full_path))
    return wrapped


def _safe_next(target, default_endpoint='kelola'):
    """Cegah open redirect: hanya izinkan path relatif internal."""
    if target and target.startswith('/') and not target.startswith('//'):
        return target
    return url_for(default_endpoint)


@app.route('/login', methods=['GET', 'POST'])
def login():
    nxt = request.values.get('next', '')
    if request.method == 'POST':
        if _cek_password(request.form.get('password', '')):
            session.clear()                 # cegah session fixation
            session['is_admin'] = True
            session.permanent = True
            return redirect(_safe_next(nxt))
        time.sleep(0.5)                     # perlambat brute force
        return render_template('login.html', error='Password salah', next=nxt)
    if session.get('is_admin'):
        return redirect(_safe_next(nxt))
    return render_template('login.html', error='', next=nxt)


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

# =====================================================


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default, lo=None, hi=None):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def _safe_row_index(value, size):
    """Kembalikan index valid, atau None jika di luar jangkauan."""
    idx = _to_int(value, -1)
    return idx if 0 <= idx < size else None


def preprocess_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = (sum(len(doc) for doc in corpus) / self.corpus_size) if self.corpus_size else 0
        self.doc_freqs = []
        self.doc_len = []
        self.idf = {}
        self._initialize(corpus)

    def _initialize(self, corpus):
        df = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for term in frequencies:
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self.idf[term] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1)

    def _base_score(self, idx, query_tokens, k1_val, b_val, safe_avgdl):
        score = 0.0
        doc_len = self.doc_len[idx]
        frequencies = self.doc_freqs[idx]
        for term in query_tokens:
            tf = frequencies.get(term)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            numerator = idf * tf * (k1_val + 1)
            denominator = tf + k1_val * (1 - b_val + b_val * (doc_len / safe_avgdl))
            score += numerator / denominator
        return score

    @staticmethod
    def _boost(q_low, kata_val, arti_val):
        boost = 0.0
        if q_low == kata_val:
            boost += 5.0
        elif kata_val.startswith(q_low):
            boost += 2.0
        if q_low in arti_val.split():
            boost += 1.5
        elif q_low in arti_val:
            boost += 0.5
        return boost

    def _ranked(self, raw_query, query_tokens, documents_df, n, k1_val, b_val, use_boost):
        q_low = str(raw_query).lower().strip()
        kata_s = documents_df['Kata'].astype(str).str.lower().str.strip()
        arti_s = documents_df['Arti'].astype(str).str.lower()
        safe_avgdl = self.avgdl if self.avgdl > 0 else 1.0
        scored = []
        for idx in range(self.corpus_size):
            base = self._base_score(idx, query_tokens, k1_val, b_val, safe_avgdl)
            if base <= 0:
                continue
            boost = self._boost(q_low, kata_s.iat[idx], arti_s.iat[idx]) if use_boost else 0.0
            scored.append((idx, base + boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def get_top_n(self, raw_query, query_tokens, documents_df, n=100, k1_dyn=None, b_dyn=None, use_boost=False):
        k1_val = k1_dyn if k1_dyn is not None else self.k1
        b_val = b_dyn if b_dyn is not None else self.b
        results = []
        for idx, score in self._ranked(raw_query, query_tokens, documents_df, n, k1_val, b_val, use_boost):
            doc_data = documents_df.iloc[idx].to_dict()
            doc_data['bm25_score'] = round(score, 4)
            doc_data['score'] = round(score, 4)
            results.append(doc_data)
        return results

    def get_top_n_light(self, raw_query, query_tokens, documents_df, n=100, k1_dyn=None, b_dyn=None, use_boost=False):
        """Versi cepat untuk grid-search tuning: hanya kolom Kata & Arti."""
        k1_val = k1_dyn if k1_dyn is not None else self.k1
        b_val = b_dyn if b_dyn is not None else self.b
        kata_s = documents_df['Kata']
        arti_s = documents_df['Arti']
        results = []
        for idx, score in self._ranked(raw_query, query_tokens, documents_df, n, k1_val, b_val, use_boost):
            results.append({'Kata': kata_s.iat[idx], 'Arti': arti_s.iat[idx], 'score': round(score, 4)})
        return results

    def get_scored(self, raw_query, query_tokens, documents_df, k1_dyn=None, b_dyn=None, use_boost=False):
        """Kembalikan [(doc_index, skor), ...] untuk SEMUA entri berskor > 0,
        terurut descending. Dipakai untuk memetakan skor per entri TANPA
        terpotong cutoff Top-N dan TANPA ambiguitas Kata duplikat."""
        k1_val = k1_dyn if k1_dyn is not None else self.k1
        b_val = b_dyn if b_dyn is not None else self.b
        scored = self._ranked(raw_query, query_tokens, documents_df,
                              self.corpus_size, k1_val, b_val, use_boost)
        return [(idx, round(score, 4)) for idx, score in scored]


class TFIDF:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.doc_vectors = []
        self.idf = {}
        df = {}
        for document in corpus:
            frequencies = Counter(document)
            self.doc_vectors.append(frequencies)
            for term in frequencies:
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size / freq)) if freq > 0 else 0.0

    def get_top_n(self, raw_query, query_tokens, documents_df, n=100):
        scored = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            doc_len = sum(doc_vec.values()) or 1
            score = 0.0
            for term in query_tokens:
                if term in doc_vec:
                    score += (doc_vec[term] / doc_len) * self.idf.get(term, 0.0)
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:n]:
            doc_data = documents_df.iloc[idx].to_dict()
            doc_data['score'] = round(score, 4)
            results.append(doc_data)
        return results


def load_data():
    try:
        df = pd.read_csv(CSV_FILE).fillna('')
        return df[df['Kata'].astype(str).str.strip() != ''].reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame(columns=FIELDS)


def build_corpus(df):
    """Bangun korpus dari INDEXED_FIELDS (4 kolom) — TANPA Ejaan fonetik."""
    corpus_tokens = []
    for _, row in df.iterrows():
        combined = ' '.join(str(row.get(c, '')) for c in INDEXED_FIELDS)
        corpus_tokens.append(preprocess_text(combined))
    return corpus_tokens


df_kamus = load_data()
corpus = build_corpus(df_kamus)
bm25_model = BM25(corpus)
tfidf_model = TFIDF(corpus)


def rebuild_models():
    global corpus, bm25_model, tfidf_model
    with _model_lock:
        corpus = build_corpus(df_kamus)
        bm25_model = BM25(corpus)
        tfidf_model = TFIDF(corpus)


def get_pagination(data_list, page, per_page):
    page = max(1, page)
    total = len(data_list)
    start = (page - 1) * per_page
    return data_list[start:start + per_page], math.ceil(total / per_page) if total > 0 else 1


def load_json_file(path, default=None):
    if default is None:
        default = []
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path, data):
    """Atomic write: tulis ke .tmp lalu replace, agar tidak korup saat tabrakan."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _unique_name(existing_names, name):
    """Anti-duplikat: 'Batch 1' -> 'Batch 1 (1)' -> 'Batch 1 (2)' dst."""
    name = str(name).strip()
    taken = {str(n).strip().lower() for n in existing_names}
    if name.lower() not in taken:
        return name
    n = 1
    while f"{name} ({n})".lower() in taken:
        n += 1
    return f"{name} ({n})"


def _batch_context():
    batches = load_json_file(BATCH_FILE)
    batch_id = request.args.get('batch')
    options = [{'id': b['group_id'], 'nama': b.get('group_name', b['group_id'])} for b in reversed(batches)]
    if batch_id:
        selected = next((b for b in reversed(batches) if b['group_id'] == batch_id), None)
    else:
        selected = batches[-1] if batches else None
    return options, selected


def _resolve_gt_to_lemmas(graded, df):
    """Perbaiki kunci GT agar memakai KATA DASAR ASLI (lema) entri kamus.
    Jika kunci GT (mis. kata kueri 'makan') tidak persis sama dengan lema kamus
    mana pun, cocokkan ke lema via awalan (mis. 'makana')."""
    lemmas = {str(k).strip().lower() for k in df['Kata'].astype(str).tolist() if str(k).strip()}
    resolved = {}
    for key, score in graded.items():
        k = str(key).strip().lower()
        if not k:
            continue
        if k in lemmas:
            resolved[k] = max(resolved.get(k, 0), score)
            continue
        matches = sorted(l for l in lemmas if l.startswith(k) or k.startswith(l))
        if matches:
            for m in matches:
                resolved[m] = max(resolved.get(m, 0), score)
        else:
            resolved[k] = score
    return resolved


def calculate_metrics(search_results, all_gt, graded_gt, top_n_limit):
    graded_gt = graded_gt or {}
    all_gt = all_gt or []
    all_gt_set = set(all_gt)
    target_keys = all_gt_set if not graded_gt else set(k for k, v in graded_gt.items() if v > 0)
    retrieved_full = [f"{str(res.get('Kata', '')).strip().lower()} | {str(res.get('Arti', '')).strip().lower()}" for res in search_results]
    retrieved_katas = []

    tp_list, fp_list = [], []
    hits, sum_precs, mrr, dcg = 0, 0.0, 0.0, 0.0

    for rank, w in enumerate(retrieved_full, start=1):
        kata_only = w.split(' | ')[0].strip()
        retrieved_katas.append(kata_only)
        rel_score = graded_gt.get(kata_only, 0)
        if not graded_gt and kata_only in all_gt_set:
            rel_score = 3
        if rank <= top_n_limit and rel_score > 0:
            dcg += (2 ** rel_score - 1) / math.log2(rank + 1)
        if rel_score > 0:
            tp_list.append(w)
            hits += 1
            sum_precs += hits / rank
            if hits == 1:
                mrr = 1.0 / rank
        else:
            fp_list.append(w)

    fn_list = [gt for gt in target_keys if gt not in retrieved_katas]
    TP, FP, FN = len(tp_list), len(fp_list), len(fn_list)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    ap_score = sum_precs / (TP + FN) if (TP + FN) > 0 else 0.0

    ideal_rels = sorted([v for v in graded_gt.values() if v > 0] if graded_gt else [3] * len(target_keys), reverse=True)[:top_n_limit]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg = min(1.0, dcg / idcg) if idcg > 0 else 0.0

    return TP, FP, FN, round(float(precision), 4), round(float(recall), 4), round(float(f1), 4), round(float(ap_score), 4), round(float(mrr), 4), round(float(ndcg), 4)


def _metrics_dict(m):
    tp, fp, fn, p, r, f1, ap, mrr, ndcg = m
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r,
            'f1_score': f1, 'ap_score': ap, 'mrr': mrr, 'ndcg': ndcg}


# ================= HELPER: BM25 MURNI & GRAFIK =================

def _hitung_bm25_murni(selected, top_n, k1_val, b_val):
    """Hitung metrik BM25 murni vs boost untuk seluruh kueri satu draft.

    Skor diambil dari get_scored() (semua entri berskor > 0) lalu dipetakan
    per indeks dokumen, sehingga entri yang tergeser dari Top-N boost tetap
    menampilkan skor boost aslinya.
    """
    kata_asli = df_kamus['Kata'].astype(str).str.strip().tolist()
    kata_lower = [k.lower() for k in kata_asli]
    arti_list = df_kamus['Arti'].astype(str).tolist()

    queries_eval = []
    if not selected:
        return queries_eval, None, None

    for q_item in selected.get('queries', []):
        kueri = str(q_item.get('kueri', '')).strip()
        if not kueri:
            continue

        graded_raw = {
            str(k).split('|')[0].strip().lower(): _to_int(v, 0, lo=0, hi=3)
            for k, v in (q_item.get('graded_gt') or {}).items()
        }
        graded = _resolve_gt_to_lemmas(graded_raw, df_kamus)
        all_gt = [k for k, v in graded.items() if v > 0]
        tokens = preprocess_text(kueri)

        pure_full = bm25_model.get_scored(kueri, tokens, df_kamus,
                                          k1_dyn=k1_val, b_dyn=b_val, use_boost=False)
        boost_full = bm25_model.get_scored(kueri, tokens, df_kamus,
                                           k1_dyn=k1_val, b_dyn=b_val, use_boost=True)
        boost_by_idx = {idx: (rank, score)
                        for rank, (idx, score) in enumerate(boost_full, start=1)}

        def metric_rows(scored):
            return [{'Kata': kata_asli[i], 'Arti': arti_list[i]}
                    for i, _ in scored[:top_n]]

        m_pure = _metrics_dict(calculate_metrics(metric_rows(pure_full), all_gt, graded, top_n))
        m_boost = _metrics_dict(calculate_metrics(metric_rows(boost_full), all_gt, graded, top_n))

        entries = []
        for rank, (idx, score) in enumerate(pure_full[:top_n], start=1):
            b_rank, b_score = boost_by_idx.get(idx, ('-', 0.0))
            entries.append({
                'Kata': kata_asli[idx],
                'Arti': arti_list[idx],
                'pure_rank': rank,
                'pure_score': score,
                'boost_rank': b_rank,
                'boost_score': b_score,
                'gt_score': graded.get(kata_lower[idx], 0),
            })

        retr_pure = {kata_lower[i] for i, _ in pure_full[:top_n]}
        retr_boost = {kata_lower[i] for i, _ in boost_full[:top_n]}

        queries_eval.append({
            'kueri': kueri,
            'kategori': str(q_item.get('kategori', 'Umum')).strip() or 'Umum',
            'm_pure': m_pure,
            'm_boost': m_boost,
            'entries': entries,
            'gt_display': [
                {'kata': k, 'score': v}
                for k, v in sorted(graded.items(), key=lambda kv: (-kv[1], kv[0]))
                if v > 0
            ],
            'total_gt': len(all_gt),
            'gt_found_pure': len(retr_pure & set(all_gt)),
            'gt_found_boost': len(retr_boost & set(all_gt)),
        })

    if not queries_eval:
        return queries_eval, None, None

    n = len(queries_eval)
    kunci = ['precision', 'recall', 'f1_score', 'ap_score', 'mrr', 'ndcg']
    mean_pure = {k: round(sum(q['m_pure'][k] for q in queries_eval) / n, 4) for k in kunci}
    mean_boost = {k: round(sum(q['m_boost'][k] for q in queries_eval) / n, 4) for k in kunci}
    return queries_eval, mean_pure, mean_boost


def _agregat_kategori(queries_eval):
    """Rata-rata MAP/NDCG per kategori, diurutkan dari kategori terbanyak."""
    bucket = {}
    for q in queries_eval:
        bucket.setdefault(q['kategori'], []).append(q)

    hasil = []
    for nama, items in bucket.items():
        n = len(items)
        hasil.append({
            'kategori': nama,
            'jumlah': n,
            'map_pure': sum(i['m_pure']['ap_score'] for i in items) / n,
            'map_boost': sum(i['m_boost']['ap_score'] for i in items) / n,
            'ndcg_pure': sum(i['m_pure']['ndcg'] for i in items) / n,
            'ndcg_boost': sum(i['m_boost']['ndcg'] for i in items) / n,
        })
    hasil.sort(key=lambda r: (-r['jumlah'], r['kategori'].lower()))
    return hasil


def _render_chart_png(tipe, queries_eval, mean_pure, mean_boost):
    """Render grafik batang HD (matplotlib, PNG) sesuai tipe yang diminta.

    Aturan tampilan: TANPA judul/subjudul di dalam gambar. Keterangan hanya
    berupa legenda di atas kanvas. Latar putih, label nilai 4 desimal,
    sumbu Y 0–1,1, dan resolusi 300 dpi agar tajam di artikel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    WARNA = {
        'map_murni': '#6D28D9',
        'map_boost': '#A78BFA',
        'ndcg_murni': '#BE123C',
        'ndcg_boost': '#F9A8D4',
        'murni': '#8B5CF6',
        'boost': '#3B82F6',
    }

    def siapkan_kanvas(n_label, lebar=None):
        if lebar is None:
            lebar = max(11.0, min(20.0, 7.5 + 1.9 * n_label))
        fig, ax = plt.subplots(figsize=(lebar, 6.6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(axis='y', color='#E5E7EB', linewidth=0.9)
        ax.set_axisbelow(True)
        for sisi in ('top', 'right'):
            ax.spines[sisi].set_visible(False)
        ax.spines['left'].set_color('#9CA3AF')
        ax.spines['bottom'].set_color('#9CA3AF')
        ax.tick_params(axis='y', labelsize=13, colors='#111827')
        ax.tick_params(axis='x', labelsize=13, colors='#111827')
        ax.set_ylabel('Skor (0–1)', fontsize=14, color='#111827', labelpad=8)
        return fig, ax

    def rapikan_x(ax, labels, miring=None):
        n = len(labels)
        ukuran = 15 if n <= 4 else (12 if n <= 10 else (9 if n <= 24 else 7.5))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=ukuran, fontweight='bold', color='#111827')
        if miring is None:
            miring = 0 if n <= 4 else (30 if n <= 10 else (60 if n <= 24 else 75))
        if miring:
            plt.setp(ax.get_xticklabels(), rotation=miring, ha='right',
                     rotation_mode='anchor')

    def beri_label(ax, bars, n, desimal=4):
        if n > 24:
            return
        ukuran = 11 if n <= 4 else (9 if n <= 10 else 7.5)
        ax.bar_label(bars, fmt=f'%.{desimal}f', padding=3,
                     fontsize=ukuran, fontweight='bold', color='#111827')

    def legenda(ax, jumlah_kolom):
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                        ncol=jumlah_kolom, frameon=False, fontsize=13,
                        handlelength=1.1, handleheight=1.1, columnspacing=1.8)
        for teks in leg.get_texts():
            teks.set_color('#111827')

    def simpan(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, facecolor='white',
                    bbox_inches='tight', pad_inches=0.22)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # ---- 1. MAP & NDCG per kategori (4 batang) ----
    if tipe == 'category_map_ndcg':
        rows = _agregat_kategori(queries_eval)
        labels = [f"{r['kategori']} ({r['jumlah']})" for r in rows]
        n = len(rows)
        fig, ax = siapkan_kanvas(n)
        x = np.arange(n)
        w = 0.2
        b1 = ax.bar(x - 1.5 * w, [r['map_pure'] for r in rows], w,
                    label='MAP — BM25 Murni', color=WARNA['map_murni'],
                    edgecolor='white', linewidth=0.8)
        b2 = ax.bar(x - 0.5 * w, [r['map_boost'] for r in rows], w,
                    label='MAP — BM25 + Boost', color=WARNA['map_boost'],
                    edgecolor='white', linewidth=0.8)
        b3 = ax.bar(x + 0.5 * w, [r['ndcg_pure'] for r in rows], w,
                    label='NDCG — BM25 Murni', color=WARNA['ndcg_murni'],
                    edgecolor='white', linewidth=0.8)
        b4 = ax.bar(x + 1.5 * w, [r['ndcg_boost'] for r in rows], w,
                    label='NDCG — BM25 + Boost', color=WARNA['ndcg_boost'],
                    edgecolor='white', linewidth=0.8)
        rapikan_x(ax, labels)
        for b in (b1, b2, b3, b4):
            beri_label(ax, b, n)
        legenda(ax, 4)
        return simpan(fig)

    # ---- 2. Rata-rata semua metrik (murni vs boost) ----
    if tipe == 'mean':
        metrik = [
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1_score'),
            ('MAP', 'ap_score'),
            ('MRR', 'mrr'),
            ('NDCG', 'ndcg'),
        ]
        labels = [m[0] for m in metrik]
        n = len(metrik)
        fig, ax = siapkan_kanvas(n, lebar=12.5)
        x = np.arange(n)
        w = 0.34
        b1 = ax.bar(x - w / 2, [mean_pure[k] for _, k in metrik], w,
                    label='BM25 Murni', color=WARNA['murni'],
                    edgecolor='white', linewidth=0.8)
        b2 = ax.bar(x + w / 2, [mean_boost[k] for _, k in metrik], w,
                    label='BM25 + Boost', color=WARNA['boost'],
                    edgecolor='white', linewidth=0.8)
        rapikan_x(ax, labels)
        beri_label(ax, b1, n)
        beri_label(ax, b2, n)
        legenda(ax, 2)
        return simpan(fig)

    # ---- 3. NDCG / MAP per kueri (murni vs boost) ----
    if tipe in ('ndcg_compare_per_query', 'map_compare_per_query'):
        if tipe.startswith('ndcg'):
            kunci = 'ndcg'
            judul_murni = 'NDCG — BM25 Murni'
            judul_boost = 'NDCG — BM25 + Boost'
            warna_m, warna_b = WARNA['ndcg_murni'], WARNA['ndcg_boost']
        else:
            kunci = 'ap_score'
            judul_murni = 'MAP — BM25 Murni'
            judul_boost = 'MAP — BM25 + Boost'
            warna_m, warna_b = WARNA['map_murni'], WARNA['map_boost']

        labels = [q['kueri'] for q in queries_eval]
        n = len(labels)
        fig, ax = siapkan_kanvas(n, lebar=max(12.0, min(22.0, 8.0 + 0.34 * n)))
        x = np.arange(n)
        w = 0.38
        b1 = ax.bar(x - w / 2, [q['m_pure'][kunci] for q in queries_eval], w,
                    label=judul_murni, color=warna_m,
                    edgecolor='white', linewidth=0.7)
        b2 = ax.bar(x + w / 2, [q['m_boost'][kunci] for q in queries_eval], w,
                    label=judul_boost, color=warna_b,
                    edgecolor='white', linewidth=0.7)
        rapikan_x(ax, labels)
        beri_label(ax, b1, n)
        beri_label(ax, b2, n)
        legenda(ax, 2)
        return simpan(fig)

    # ---- 4. Metrik tunggal per kueri (murni) ----
    if tipe in ('ndcg_per_query', 'f1_score_per_query', 'precision_per_query',
                'recall_per_query', 'ap_score_per_query', 'mrr_per_query'):
        peta = {
            'ndcg_per_query': ('ndcg', 'NDCG', '#EC4899'),
            'f1_score_per_query': ('f1_score', 'F1-Score', '#06B6D4'),
            'precision_per_query': ('precision', 'Precision', '#1E3A8A'),
            'recall_per_query': ('recall', 'Recall', '#2563EB'),
            'ap_score_per_query': ('ap_score', 'MAP', '#8B5CF6'),
            'mrr_per_query': ('mrr', 'MRR', '#64748B'),
        }
        kunci, nama, warna = peta[tipe]
        labels = [q['kueri'] for q in queries_eval]
        n = len(labels)
        fig, ax = siapkan_kanvas(n, lebar=max(12.0, min(22.0, 8.0 + 0.34 * n)))
        x = np.arange(n)
        b = ax.bar(x, [q['m_pure'][kunci] for q in queries_eval], 0.55,
                   label=f'{nama} — BM25 Murni', color=warna,
                   edgecolor='white', linewidth=0.7)
        rapikan_x(ax, labels)
        beri_label(ax, b, n)
        legenda(ax, 1)
        return simpan(fig)

    # ---- 5. NDCG per kategori (murni vs boost) ----
    if tipe == 'category':
        rows = _agregat_kategori(queries_eval)
        labels = [f"{r['kategori']} ({r['jumlah']})" for r in rows]
        n = len(rows)
        fig, ax = siapkan_kanvas(n)
        x = np.arange(n)
        w = 0.34
        b1 = ax.bar(x - w / 2, [r['ndcg_pure'] for r in rows], w,
                    label='NDCG — BM25 Murni', color=WARNA['ndcg_murni'],
                    edgecolor='white', linewidth=0.8)
        b2 = ax.bar(x + w / 2, [r['ndcg_boost'] for r in rows], w,
                    label='NDCG — BM25 + Boost', color=WARNA['ndcg_boost'],
                    edgecolor='white', linewidth=0.8)
        rapikan_x(ax, labels)
        beri_label(ax, b1, n)
        beri_label(ax, b2, n)
        legenda(ax, 2)
        return simpan(fig)

    raise ValueError(f'Tipe grafik tidak dikenal: {tipe}')


def _jsonify_rows(rows):
    return jsonify([
        {k: ("" if pd.isna(v) else (v.item() if hasattr(v, 'item') else v) if isinstance(v, (np.number, np.bool_)) else str(v))
         for k, v in r.items()} for r in rows
    ])


@app.route('/', methods=['GET', 'POST'])
def index():
    src = request.form if request.method == 'POST' else request.args
    query = str(src.get('query', '')).strip()
    algo = src.get('algo', 'bm25_boost')
    k1_val = _to_float(src.get('k1', 1.2), 1.2)
    b_val = _to_float(src.get('b', 0.75), 0.75)
    page = max(1, request.args.get('page', 1, type=int) or 1)

    results = []
    if query:
        q_tokens = preprocess_text(query)
        if algo == 'bm25_boost':
            results = bm25_model.get_top_n(query, q_tokens, df_kamus, n=500, k1_dyn=k1_val, b_dyn=b_val, use_boost=True)
        elif algo == 'bm25_pure':
            results = bm25_model.get_top_n(query, q_tokens, df_kamus, n=500, k1_dyn=k1_val, b_dyn=b_val, use_boost=False)
        elif algo == 'tfidf':
            results = tfidf_model.get_top_n(query, q_tokens, df_kamus, n=500)
        else:
            q_low = query.lower()
            mask = (df_kamus['Kata'].astype(str).str.lower().str.contains(q_low, regex=False) |
                    df_kamus['Arti'].astype(str).str.lower().str.contains(q_low, regex=False) |
                    df_kamus['Contoh_Manado'].astype(str).str.lower().str.contains(q_low, regex=False) |
                    df_kamus['Contoh_Indonesia'].astype(str).str.lower().str.contains(q_low, regex=False))
            results = df_kamus[mask].to_dict('records')
        for r in results:
            r.setdefault('bm25_score', r.get('score', 0))

    res_page, total_pages = get_pagination(results, page, PER_PAGE_SEARCH)
    return render_template('index.html', query=query, results=res_page, page=page,
                           total_pages=total_pages, algo=algo, k1=k1_val, b=b_val,
                           per_page=PER_PAGE_SEARCH)


# ================= ROUTE: KELOLA DATASET =================
@app.route('/kelola', methods=['GET', 'POST'])
@admin_required
def kelola():
    global df_kamus
    pesan = request.args.get('pesan', '')

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'tambah':
            baru = {k: str(request.form.get(k.lower(), '')).strip() for k in ['Kata', 'Ejaan', 'Arti']}
            baru.update({'Contoh_Manado': request.form.get('contoh_manado', ''),
                         'Contoh_Indonesia': request.form.get('contoh_indonesia', '')})
            if not baru['Kata'] or not baru['Arti']:
                return redirect(url_for('kelola', pesan='Kolom Kata dan Arti wajib diisi'))
            df_kamus = pd.concat([df_kamus, pd.DataFrame([baru])], ignore_index=True)
            pesan = f"Kata '{baru['Kata']}' berhasil ditambahkan"
        elif action in ('edit', 'hapus'):
            idx = _safe_row_index(request.form.get('index'), len(df_kamus))
            if idx is None:
                return redirect(url_for('kelola', pesan='Indeks tidak valid (data mungkin sudah berubah, muat ulang halaman)'))
            if action == 'edit':
                for k in FIELDS:
                    df_kamus.at[idx, k] = request.form.get(k.lower(), '')
                pesan = "Data berhasil diperbarui"
            else:
                kata = df_kamus.iloc[idx]['Kata']
                df_kamus = df_kamus.drop(idx).reset_index(drop=True)
                pesan = f"Kata '{kata}' berhasil dihapus"
        else:
            return redirect(url_for('kelola', pesan='Aksi tidak dikenal'))

        df_kamus.to_csv(CSV_FILE, index=False)
        rebuild_models()
        return redirect(url_for('kelola', pesan=pesan))

    q = request.args.get('q', '').lower().strip()
    page = max(1, request.args.get('page', 1, type=int) or 1)
    per_page = _to_int(request.args.get('per_page', 50), 50, lo=1, hi=500)

    if q:
        view_df = df_kamus[df_kamus['Kata'].astype(str).str.lower().str.contains(q, regex=False) |
                           df_kamus['Arti'].astype(str).str.lower().str.contains(q, regex=False)]
    else:
        view_df = df_kamus.copy()

    data_page, total_pages = get_pagination(view_df.to_dict('records'), page, per_page)
    start = (page - 1) * per_page
    for i, d in enumerate(data_page):
        d['_idx'] = int(view_df.index[start + i])
        d['_no'] = start + i + 1

    return render_template('kelola.html', data_list=data_page, pesan=pesan, page=page,
                           total_pages=total_pages, search_filter=q,
                           total_data=len(view_df), total_keseluruhan=len(df_kamus))


# ================= ROUTE: DASHBOARD =================
@app.route('/dashboard/evaluasi')
def evaluasi():
    return render_template('evaluasi.html')


@app.route('/dashboard/tuning')
def tuning():
    options, selected = _batch_context()
    if selected is not None and not selected.get('tuning_data'):
        batches = load_json_file(BATCH_FILE)
        selected = next((b for b in reversed(batches) if b.get('tuning_data')), selected)
    return render_template('tuning.html', batches=options, selected_batch=selected, has_data=selected is not None)


@app.route('/dashboard')
def dashboard():
    options, selected = _batch_context()
    return render_template('dashboard.html', batches=options, selected_batch=selected, has_data=selected is not None)


@app.route('/dashboard/perbandingan')
def perbandingan():
    options, selected = _batch_context()
    has_pure = selected is not None and 'pure' in selected.get('stats', {})
    has_tfidf = selected is not None and 'tfidf' in selected.get('stats', {})
    return render_template('perbandingan.html', batches=options, selected_batch=selected,
                           has_data=selected is not None, has_pure=has_pure, has_tfidf=has_tfidf)


# ================= ROUTE: MANAJEMEN BATCH =================
def _parse_ts(ts):
    try:
        return datetime.strptime(str(ts), '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return None


def _is_corrupt(b):
    """Batch dianggap korup jika mean_ndcg/mrr = 1.0 (mustahil) atau stat hilang."""
    st = b.get('stats', {})
    if not st:
        return True
    ndcg = st.get('mean_ndcg', 0)
    mrr = st.get('mean_mrr', 0)
    return (ndcg == 1.0 or mrr == 1.0) or b.get('group_id') in CORRUPT_BATCHES


@app.route('/admin/batch', methods=['GET', 'POST'])
@admin_required
def admin_batch():
    """Halaman manajemen batch (tidak di-link dari navbar)."""
    if request.method == 'POST':
        action = request.form.get('action')
        with _file_lock:
            batches = load_json_file(BATCH_FILE)
            if action == 'hapus':
                bid = request.form.get('group_id')
                batches = [b for b in batches if b.get('group_id') != bid]
                save_json_file(BATCH_FILE, batches)
            elif action == 'hapus_semua':
                save_json_file(BATCH_FILE, [])
            elif action == 'hapus_korup':
                batches = [b for b in batches if not _is_corrupt(b)]
                save_json_file(BATCH_FILE, batches)
            elif action == 'rename':
                bid = request.form.get('group_id')
                new_name = str(request.form.get('new_name', '')).strip()
                if bid and new_name:
                    existing = [b.get('group_name', '') for b in batches if b.get('group_id') != bid]
                    for b in batches:
                        if b.get('group_id') == bid:
                            b['group_name'] = _unique_name(existing, new_name)
                            break
                    save_json_file(BATCH_FILE, batches)
            elif action == 'hapus_sebelum':
                cutoff = str(request.form.get('cutoff', '')).strip()
                if cutoff:
                    try:
                        cutoff_dt = datetime.strptime(cutoff, '%Y-%m-%d %H:%M:%S')
                        batches = [b for b in batches
                                   if _parse_ts(b.get('timestamp')) is None or _parse_ts(b.get('timestamp')) >= cutoff_dt]
                        save_json_file(BATCH_FILE, batches)
                    except ValueError:
                        pass
        return redirect(url_for('admin_batch', pesan=request.form.get('action', '')))

    batches = load_json_file(BATCH_FILE)
    rows = []
    for b in reversed(batches):
        st = b.get('stats', {})
        rows.append({
            'group_id': b.get('group_id', ''),
            'group_name': b.get('group_name', b.get('group_id', '')),
            'timestamp': b.get('timestamp', ''),
            'top_n': b.get('top_n', ''),
            'total_query': st.get('total_query', ''),
            'mean_ndcg': st.get('mean_ndcg', ''),
            'mean_mrr': st.get('mean_mrr', ''),
            'grid_size': len(b.get('tuning_data', [])) if isinstance(b.get('tuning_data'), list) else 0,
            'korup': _is_corrupt(b),
            'duplikat': None,
        })
    name_count = {}
    for r in rows:
        key = r['group_name'].strip().lower()
        name_count.setdefault(key, 0)
        name_count[key] += 1
    seen = set()
    for r in rows:
        key = r['group_name'].strip().lower()
        r['duplikat'] = name_count[key] > 1 and key in seen
        seen.add(key)

    total_korup = sum(1 for r in rows if r['korup'])
    file_info = {
        'path': os.path.abspath(BATCH_FILE),
        'size_kb': round(os.path.getsize(BATCH_FILE) / 1024, 1) if os.path.exists(BATCH_FILE) else 0,
        'jumlah': len(batches),
        'jumlah_korup': total_korup,
    }
    return render_template('admin_batch.html', rows=rows, file_info=file_info,
                           pesan=request.args.get('pesan', ''))


# ================= API: DRAFT =================
@app.route('/api/group_draft', methods=['GET', 'POST', 'DELETE'])
def api_group_draft():
    if request.method in ('POST', 'DELETE') and not session.get('is_admin'):
        return jsonify({'success': False,
                        'error': 'Akses ditolak. Login admin dulu.'}), 401

    if request.method == 'GET':
        gid = request.args.get('id')
        with _file_lock:
            drafts = load_json_file(DRAFT_FILE)
        if not gid:
            return jsonify(drafts)
        return jsonify(next((d for d in drafts if d.get('group_id') == gid), {}))

    if request.method == 'DELETE':
        gid = request.args.get('id')
        with _file_lock:
            drafts = load_json_file(DRAFT_FILE)
            if gid:
                save_json_file(DRAFT_FILE, [d for d in drafts if d.get('group_id') != gid])
        return jsonify({'success': True})

    # POST
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'success': False, 'error': 'Payload tidak valid'}), 400
    with _file_lock:
        drafts = load_json_file(DRAFT_FILE)
        gid = payload.get('group_id')
        if gid:
            for i, d in enumerate(drafts):
                if d.get('group_id') == gid:
                    drafts[i] = payload
                    break
            else:
                payload['group_name'] = _unique_name([d.get('group_name', '') for d in drafts],
                                                     payload.get('group_name', 'Draft'))
                drafts.append(payload)
        else:
            payload['group_id'] = 'draft_' + str(int(time.time() * 1000))
            payload['group_name'] = _unique_name([d.get('group_name', '') for d in drafts],
                                                 payload.get('group_name', 'Draft'))
            drafts.append(payload)
        save_json_file(DRAFT_FILE, drafts)
    return jsonify({'success': True, 'group_id': payload['group_id']})


@app.route('/api/import_literal', methods=['POST'])
def api_import_literal():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or 'text' not in payload:
        return jsonify({'success': False, 'error': 'Tidak ada teks yang dikirim'}), 400

    text = str(payload.get('text', '')).strip()
    if not text:
        return jsonify({'success': False, 'error': 'Teks kosong'})

    queries = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    queries.append({'kueri': item.strip(), 'kategori': 'Kata Umum', 'graded_gt': {}})
                elif isinstance(item, dict):
                    queries.append(item)
    except Exception:
        for line in text.split('\n'):
            line = line.strip()
            if line:
                queries.append({'kueri': line, 'kategori': 'Kata Umum', 'graded_gt': {}})

    return jsonify({'success': True, 'queries': queries})


# ================= API: PREVIEW =================
@app.route('/api/preview_retrieval')
def preview_retrieval():
    q = request.args.get('q', '').strip()
    if not q or df_kamus.empty:
        return jsonify([])
    top_n = _to_int(request.args.get('top_n', DEFAULT_TOP_N), DEFAULT_TOP_N, lo=1, hi=500)
    use_boost = request.args.get('use_bm25', 'true') == 'true'
    res = bm25_model.get_top_n(q, preprocess_text(q), df_kamus, n=top_n,
                               k1_dyn=_to_float(request.args.get('k1', 1.2), 1.2),
                               b_dyn=_to_float(request.args.get('b', 0.75), 0.75),
                               use_boost=use_boost)
    return _jsonify_rows(res)


@app.route('/api/preview_inverted')
def preview_inverted():
    q = request.args.get('q', '').strip().lower()
    if not q or df_kamus.empty:
        return jsonify([])
    mask = df_kamus['Kata'].astype(str).str.lower().str.contains(q, regex=False) | \
           df_kamus['Arti'].astype(str).str.lower().str.contains(q, regex=False)
    return _jsonify_rows(df_kamus[mask].to_dict('records'))


# ================= API: EVALUASI GROUP =================
@app.route('/api/evaluate_group', methods=['POST'])
@admin_required
def api_evaluate_group():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload.get('queries'):
        return jsonify({'success': False, 'error': 'Payload tidak valid'}), 400

    gid = payload.get('group_id')
    gname = str(payload.get('group_name', 'Untitled')).strip() or 'Untitled'
    top_n = _to_int(payload.get('top_n', DEFAULT_TOP_N), DEFAULT_TOP_N, lo=1, hi=1000)
    k1_val = _to_float(payload.get('k1', 1.2), 1.2)
    b_val = _to_float(payload.get('b', 0.75), 0.75)

    # --- 1. Validasi & normalisasi kueri (GT di-resolver ke lema asli kamus) ---
    valid_queries = []
    for q_item in payload.get('queries', []):
        if not isinstance(q_item, dict):
            continue
        kueri = str(q_item.get('kueri', '')).strip()
        if not kueri:
            continue
        graded = q_item.get('graded_gt') or {}
        graded = _resolve_gt_to_lemmas(
            {str(k).split('|')[0].strip().lower(): _to_int(v, 0, lo=0, hi=3) for k, v in graded.items()},
            df_kamus)
        valid_queries.append({
            'kueri': kueri,
            'kategori': str(q_item.get('kategori', 'Umum')).strip() or 'Umum',
            'tokens': preprocess_text(kueri),
            'all_gt': [k for k, v in graded.items() if v > 0],
            'graded_gt': graded,
        })

    if not valid_queries:
        return jsonify({'success': False, 'error': 'Tidak ada kueri valid'}), 400
    total_q = len(valid_queries)

    # --- 2. Hitung metrik ---
    evaluated_queries = []
    for vq in valid_queries:
        res_pure = bm25_model.get_top_n(vq['kueri'], vq['tokens'], df_kamus, n=top_n, k1_dyn=k1_val, b_dyn=b_val, use_boost=False)
        res_boost = bm25_model.get_top_n(vq['kueri'], vq['tokens'], df_kamus, n=top_n, k1_dyn=k1_val, b_dyn=b_val, use_boost=True)
        res_tfidf = tfidf_model.get_top_n(vq['kueri'], vq['tokens'], df_kamus, n=top_n)

        m_pure = _metrics_dict(calculate_metrics(res_pure, vq['all_gt'], vq['graded_gt'], top_n))
        m_boost = _metrics_dict(calculate_metrics(res_boost, vq['all_gt'], vq['graded_gt'], top_n))
        m_tfidf = _metrics_dict(calculate_metrics(res_tfidf, vq['all_gt'], vq['graded_gt'], top_n))

        evaluated_queries.append({
            'kueri': vq['kueri'], 'kategori': vq['kategori'],
            'ground_truth': vq['all_gt'], 'graded_gt': vq['graded_gt'],
            'pure_metrics': m_pure, **m_boost,
            'tfidf_metrics': m_tfidf,
        })

    # --- 3. Rata-rata ---
    def summarize(getter):
        def avg(key):
            return round(sum(getter(q).get(key, 0) for q in evaluated_queries) / total_q, 4)
        return {'avg_precision': avg('precision'), 'avg_recall': avg('recall'),
                'avg_f1': avg('f1_score'), 'map_score': avg('ap_score'),
                'mean_mrr': avg('mrr'), 'mean_ndcg': avg('ndcg')}

    stats = summarize(lambda q: q)
    stats.update(total_query=total_q,
                 pure=summarize(lambda q: q['pure_metrics']),
                 tfidf=summarize(lambda q: q['tfidf_metrics']))

    # --- 4. Statistik per kategori ---
    category_stats = {}
    for q in evaluated_queries:
        c = category_stats.setdefault(q['kategori'], {'precision': [], 'recall': [], 'f1': [], 'ap': [], 'mrr': [], 'ndcg': [], 'count': 0})
        c['precision'].append(q['precision']); c['recall'].append(q['recall'])
        c['f1'].append(q['f1_score']); c['ap'].append(q['ap_score'])
        c['mrr'].append(q['mrr']); c['ndcg'].append(q['ndcg'])
        c['count'] += 1

    grouped_stats = [{'kategori': cat,
                      'avg_precision': round(sum(v['precision']) / v['count'], 4),
                      'avg_recall': round(sum(v['recall']) / v['count'], 4),
                      'avg_f1': round(sum(v['f1']) / v['count'], 4),
                      'avg_map': round(sum(v['ap']) / v['count'], 4),
                      'avg_mrr': round(sum(v['mrr']) / v['count'], 4),
                      'avg_ndcg': round(sum(v['ndcg']) / v['count'], 4),
                      'total': v['count']} for cat, v in category_stats.items()]

    gt_lengths = [len(q['ground_truth']) for q in evaluated_queries]
    gt_stats = {
        'min': min(gt_lengths) if gt_lengths else 0,
        'max': max(gt_lengths) if gt_lengths else 0,
        'mean': round(sum(gt_lengths) / len(gt_lengths), 2) if gt_lengths else 0,
        'count_gt_gt_topn': sum(1 for x in gt_lengths if x > top_n) if gt_lengths else 0
    }

    # --- 5. Grid-search tuning ---
    kata_arti = df_kamus[['Kata', 'Arti']]
    tuning_results = []
    for k1 in K1_GRID:
        for b in B_GRID:
            t_ndcg = t_f1 = t_map = 0.0
            for vq in valid_queries:
                res = bm25_model.get_top_n_light(vq['kueri'], vq['tokens'], kata_arti, n=top_n,
                                                 k1_dyn=k1, b_dyn=b, use_boost=True)
                m = calculate_metrics(res, vq['all_gt'], vq['graded_gt'], top_n)
                t_ndcg += m[8]; t_f1 += m[5]; t_map += m[6]
            tuning_results.append({'k1': float(k1), 'b': float(b),
                                   'ndcg': round(t_ndcg / total_q, 6),
                                   'f1': round(t_f1 / total_q, 6),
                                   'map': round(t_map / total_q, 6)})

    # --- 6. Simpan batch ---
    current_time = datetime.now()
    batch_result = {
        'group_id': (gid or 'eval') + '_' + str(int(time.time())),
        'group_name': gname,
        'original_draft_id': gid,
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'top_n': top_n,
        'queries': evaluated_queries,
        'grouped_stats': grouped_stats,
        'gt_stats': gt_stats,
        'tuning_data': tuning_results,
        'stats': stats,
    }

    with _file_lock:
        batches = load_json_file(BATCH_FILE)
        existing_names = [b.get('group_name', '') for b in batches]
        batch_result['group_name'] = _unique_name(existing_names, gname)
        batches.append(batch_result)
        save_json_file(BATCH_FILE, batches)

        drafts = load_json_file(DRAFT_FILE)
        for d in drafts:
            if d.get('group_id') == gid:
                d['last_evaluated'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
        save_json_file(DRAFT_FILE, drafts)

    return jsonify({'success': True, 'group_id': batch_result['group_id'],
                    'group_name': batch_result['group_name']})


# ================= ROUTE: LAIN-LAIN =================
@app.route('/reset_evaluasi', methods=['POST'])
@admin_required
def reset_evaluasi():
    with _file_lock:
        for f in [BATCH_FILE, DRAFT_FILE]:
            if os.path.exists(f):
                os.remove(f)
    return redirect(url_for('dashboard'))


@app.route('/reset_batch', methods=['POST'])
@admin_required
def reset_batch():
    """Hapus semua hasil batch. Draft (kueri + GT) TIDAK disentuh."""
    with _file_lock:
        if os.path.exists(BATCH_FILE):
            os.remove(BATCH_FILE)
    return redirect(url_for('dashboard'))


@app.route('/hapus_batch/<batch_id>', methods=['POST'])
@admin_required
def hapus_batch(batch_id):
    with _file_lock:
        batches = load_json_file(BATCH_FILE)
        save_json_file(BATCH_FILE, [b for b in batches if b.get('group_id') != batch_id])
    return redirect(url_for('dashboard'))


# ================= ROUTE: BM25 MURNI =================
@app.route('/BM25_murni.html')
def bm25_murni():
    """Halaman uji BM25 murni vs boost untuk seluruh kueri draft."""
    with _file_lock:
        drafts = load_json_file(DRAFT_FILE)

    options = [{'id': d.get('group_id'), 'nama': d.get('group_name', d.get('group_id'))}
               for d in reversed(drafts)]
    draft_id = request.args.get('draft')
    selected = next((d for d in drafts if d.get('group_id') == draft_id), None)

    top_n = _to_int(request.args.get('top_n', DEFAULT_TOP_N), DEFAULT_TOP_N, lo=1, hi=500)
    k1_val = _to_float(request.args.get('k1', 1.2), 1.2)
    b_val = _to_float(request.args.get('b', 0.75), 0.75)

    queries_eval, mean_pure, mean_boost = _hitung_bm25_murni(
        selected, top_n, k1_val, b_val)

    return render_template('BM25_murni.html', options=options, selected=selected,
                           queries_eval=queries_eval, mean_pure=mean_pure,
                           mean_boost=mean_boost, top_n=top_n, k1=k1_val,
                           b=b_val, has_drafts=bool(options))


@app.route('/BM25_murni/chart.png')
def bm25_murni_chart():
    """Render grafik batang HD (PNG) untuk halaman BM25 Murni.

    Parameter: draft, top_n, k1, b, type, download=1.
    Gambar TANPA judul/subjudul; keterangan hanya legenda di atas kanvas.
    """
    with _file_lock:
        drafts = load_json_file(DRAFT_FILE)

    draft_id = request.args.get('draft')
    selected = next((d for d in drafts if d.get('group_id') == draft_id), None)
    if selected is None and drafts:
        selected = drafts[-1]

    top_n = _to_int(request.args.get('top_n', DEFAULT_TOP_N), DEFAULT_TOP_N, lo=1, hi=500)
    k1_val = _to_float(request.args.get('k1', 1.2), 1.2)
    b_val = _to_float(request.args.get('b', 0.75), 0.75)

    tipe = request.args.get('type', 'category_map_ndcg')
    if tipe not in CHART_TYPES:
        tipe = 'category_map_ndcg'

    queries_eval, mean_pure, mean_boost = _hitung_bm25_murni(
        selected, top_n, k1_val, b_val)
    if not queries_eval:
        return ('Tidak ada kueri valid untuk draft ini.', 404,
                {'Content-Type': 'text/plain; charset=utf-8'})

    try:
        with _chart_lock:
            png = _render_chart_png(tipe, queries_eval, mean_pure, mean_boost)
    except ImportError:
        return ('Matplotlib belum terpasang. Jalankan: pip install matplotlib',
                503, {'Content-Type': 'text/plain; charset=utf-8'})
    except Exception as exc:
        app.logger.exception('Gagal merender grafik BM25: %s', exc)
        return ('Gagal merender grafik. Lihat log server.', 500,
                {'Content-Type': 'text/plain; charset=utf-8'})

    resp = send_file(io.BytesIO(png), mimetype='image/png', max_age=0)
    if request.args.get('download') == '1':
        resp.headers['Content-Disposition'] = (
            f'attachment; filename="grafik_bm25_{tipe}.png"')
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    return resp


if __name__ == '__main__':
    app.run(debug=False)