import os
import re
import io
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from rapidfuzz.distance import DamerauLevenshtein

# Matikan auto-conversion safetensors untuk mencegah error thread di cloud
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT (STANDAR HKI)
# ==========================================
st.set_page_config(
    page_title="BEDSpell Biologi - Sistem Koreksi Ejaan Esai",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# FUNGSI DASAR & PEMUATAN MODEL (CACHED)
# ==========================================
def tokenize(teks):
    """Mengubah kalimat menjadi daftar kata huruf kecil (hanya a-z)."""
    if pd.isna(teks):
        return []
    return re.findall(r'\b[a-z]+\b', str(teks).lower())

@st.cache_resource(show_spinner="Memuat Model IndoBERT MLM (indolem/indobert-base-uncased)...")
def load_indobert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nama_model = "indolem/indobert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(nama_model)
    model = AutoModelForMaskedLM.from_pretrained(nama_model, use_safetensors=False).to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_data(show_spinner="Membangun Kamus Master Multi-Sumber (Tri-Dictionary)...")
def load_master_dictionary():
    kamus_kbbi = set()
    kamus_kunci_guru = set()
    kamus_biologi = set()

    # 1. Muat KBBI (Jika folder tersedia di repo)
    path_stems = "./NormalisasiKata/data_stems.csv"
    if os.path.exists(path_stems):
        with open(path_stems, "r", encoding="utf-8", errors="ignore") as f:
            kamus_kbbi.update(tokenize(f.read()))

    path_kateglo = "./NormalisasiKata/data_kamus_kateglo.json"
    if os.path.exists(path_kateglo):
        with open(path_kateglo, "r", encoding="utf-8", errors="ignore") as f:
            for w in tokenize(f.read()):
                if len(w) >= 3 and any(v in w for v in "aiueo") and w not in {"rpoprasi", "presrasi", "splusinya", "silusinya"}:
                    kamus_kbbi.add(w)

    # 2. Muat Kunci Guru (Jika file tersedia di repo)
    for file_kunci in ["KunciGuru.xlsx", "DatasetSiswa.xlsx"]:
        if os.path.exists(file_kunci):
            try:
                df_k = pd.read_excel(file_kunci)
                df_k.columns = df_k.columns.str.strip()
                if "Kunci Guru" in df_k.columns:
                    for teks in df_k["Kunci Guru"].dropna().unique():
                        kamus_kunci_guru.update(tokenize(teks))
                elif file_kunci == "KunciGuru.xlsx":
                    for col in df_k.columns:
                        for teks in df_k[col].dropna().unique():
                            kamus_kunci_guru.update(tokenize(teks))
            except Exception:
                pass

    # 3. Muat Kamus Terminologi Biologi
    file_bio = "Kamus_Terminologi_Biologi_Lengkap.xlsx"
    if os.path.exists(file_bio):
        try:
            df_bio = pd.read_excel(file_bio)
            for col in ["Kata_Terminologi", "Bentuk_Dasar_Konsep"]:
                if col in df_bio.columns:
                    for kata in df_bio[col].dropna():
                        kamus_biologi.update(tokenize(kata))
        except Exception:
            pass

    # Fallback kosakata esensial Transpor Membran jika file Excel belum diunggah ke GitHub
    fallback_vocab = {
        "difusi", "osmosis", "transpor", "aktif", "pasif", "membran", "sel", "fosfolipid",
        "bilayer", "semipermeabel", "selektif", "permeabel", "konsentrasi", "tinggi", "rendah",
        "gradien", "molekul", "partikel", "air", "pelarut", "zat", "terlarut", "hipotonik",
        "hipertonik", "isotonik", "isotonis", "lisis", "krenasi", "plasmolisis", "turgid",
        "turgor", "flasid", "endositosis", "eksositosis", "fagositosis", "pinositosis",
        "vesikel", "protein", "pembawa", "saluran", "channel", "carrier", "pompa", "ion",
        "natrium", "kalium", "energi", "atp", "adp", "fosfat", "glukosa", "oksigen", "karbon",
        "dioksida", "alveolus", "paru", "darah", "merah", "tubuh", "membutuhkan", "memerlukan",
        "perpindahan", "bergerak", "melalui", "tanpa", "dengan", "bantuan", "larutan",
        "seimbang", "keseimbangan", "terfasilitasi", "terbantu", "sederhana", "menuju",
        "daerah", "dari", "ke", "dalam", "luar", "sitoplasma", "ekstraseluler", "intraseluler",
        "kotranspor", "kontransporter", "simpor", "antipor", "sglt", "glut", "neurotransmiter",
        "sinapsis", "berkonformasi", "diperantarai", "melekukkan", "pelekukan", "menyebabkan",
        "diletakkan", "berdasarkan", "berwarna", "mentransfer", "berbeda", "memanfaatkan",
        "melewati", "lembek", "membrannya", "diffusion", "regulasi", "respirasi", "mitokondria",
        "berfungsi", "sebagai", "tempat", "seluler", "untuk", "menghasilkan", "tujuan",
        "memasukkan", "mengeluarkan", "sisa", "sisi", "siswa", "setelah", "menit", "tinta",
        "awal", "terkumpul", "satu", "badan", "akan", "menyebar", "seluruh", "sehingga",
        "warna", "menjadi", "lebih", "merata", "karena", "hingga", "mencapai", "sama", "pada"
    }
    if not kamus_biologi and not kamus_kunci_guru:
        kamus_biologi.update(fallback_vocab)

    kamus_master = kamus_kbbi.union(kamus_kunci_guru).union(kamus_biologi).union(fallback_vocab)
    return kamus_master, kamus_kunci_guru, len(kamus_kbbi), len(kamus_kunci_guru), len(kamus_biologi)

# ==========================================
# FUNGSI PIPELINE KOREKSI EJAAN (TAHAP 1 - 3)
# ==========================================
def generasi_kandidat(kata_typo, list_kamus_master, kamus_kunci_guru, K=5, max_e=3):
    panjang_typo = len(kata_typo)
    semua_kandidat = []

    for kata_baku in list_kamus_master:
        if abs(len(kata_baku) - panjang_typo) > max_e:
            continue
        nilai_e = DamerauLevenshtein.distance(kata_typo, kata_baku)
        if nilai_e <= max_e:
            prioritas_kunci = 0 if kata_baku in kamus_kunci_guru else 1
            semua_kandidat.append((kata_baku, nilai_e, prioritas_kunci))

    semua_kandidat.sort(key=lambda x: (x[1], x[2], abs(len(x[0]) - panjang_typo)))
    return [(k[0], k[1]) for k in semua_kandidat[:K]]

def pilih_kandidat_bedspell(kalimat_masked, list_kandidat, tokenizer, model, device, alpha=2.5):
    if not list_kandidat:
        return None, []

    inputs = tokenizer(kalimat_masked, return_tensors="pt", truncation=True, max_length=512).to(device)
    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    
    if len(mask_token_index) == 0:
        return list_kandidat[0][0], []

    mask_idx = mask_token_index[0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_idx, :]
        probs = torch.softmax(logits, dim=-1)

    hasil_ranking = []
    for kata_kand, nilai_e in list_kandidat:
        sub_tokens = tokenizer.encode(kata_kand, add_special_tokens=False)
        skor_s = probs[sub_tokens].mean().item() if len(sub_tokens) > 0 else 1e-10
        skor_f = ((1.0 / (nilai_e + 1.0)) ** alpha) * skor_s

        hasil_ranking.append({
            "Kandidat Kata": kata_kand,
            "Jarak Edit (e)": nilai_e,
            "Skor Konteks IndoBERT (s)": skor_s,
            "Skor Akhir BEDSpell f(s,e)": skor_f
        })

    hasil_ranking.sort(key=lambda x: x["Skor Akhir BEDSpell f(s,e)"], reverse=True)
    return hasil_ranking[0]["Kandidat Kata"], hasil_ranking

def proses_satu_kalimat(teks_input, kamus_master, kamus_kunci_guru, tokenizer, model, device, K=5, max_e=3, alpha=2.5):
    tokens = tokenize(teks_input)
    if not tokens:
        return "", "", "", [], {}

    list_kamus = list(kamus_master)
    daftar_typo = [k for k in tokens if k not in kamus_master]
    
    tokens_b = tokens.copy()
    tokens_c = tokens.copy()
    detail_per_typo = {}

    for i, kata in enumerate(tokens):
        if kata in daftar_typo:
            kandidat_list = generasi_kandidat(kata, list_kamus, kamus_kunci_guru, K=K, max_e=max_e)
            if kandidat_list:
                # Skenario B: Ambil kandidat pertama (jarak edit terkecil)
                tokens_b[i] = kandidat_list[0][0]

                # Skenario C: IndoBERT MLM + BEDSpell
                temp_tokens = tokens_c.copy()
                temp_tokens[i] = tokenizer.mask_token
                kalimat_masked = " ".join(temp_tokens)

                kata_terpilih, ranking_detail = pilih_kandidat_bedspell(
                    kalimat_masked, kandidat_list, tokenizer, model, device, alpha=alpha
                )
                if kata_terpilih:
                    tokens_c[i] = kata_terpilih
                    detail_per_typo[f"{kata} (posisi ke-{i+1})"] = {
                        "kalimat_masked": kalimat_masked,
                        "terpilih_b": tokens_b[i],
                        "terpilih_c": kata_terpilih,
                        "ranking": ranking_detail
                    }

    return " ".join(tokens), " ".join(tokens_b), " ".join(tokens_c), daftar_typo, detail_per_typo

# ==========================================
# ANTARMUKA UTAMA (UI STREAMLIT)
# ==========================================
tokenizer_bert, model_bert, device = load_indobert_model()
kamus_master, kamus_kunci_guru, n_kbbi, n_kunci, n_bio = load_master_dictionary()

# Header Aplikasi
st.title("🧬 Sistem Normalisasi Ejaan Esai Biologi Berbasis BEDSpell")
st.markdown(
    "**Integrasi Algoritma Damerau-Levenshtein Distance & IndoBERT Masked Language Model (MLM)**  \n"
    "*Studi Kasus: Normalisasi Teks Esai Siswa Materi Transpor Membran — SMAN 1 Bangkalan*"
)
st.divider()

# Sidebar Pengaturan Parameter & Info Kamus
with st.sidebar:
    st.header("⚙️ Parameter Algoritma")
    param_k = st.slider("Jumlah Kandidat Top-K (K)", min_value=1, max_value=10, value=5)
    param_max_e = st.slider("Batas Toleransi Edit Distance (max e)", min_value=1, max_value=5, value=3)
    param_alpha = st.slider("Bobot Penalti Alpha (α BEDSpell)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
    
    st.divider()
    st.subheader("📚 Status Tri-Dictionary")
    st.metric("Total Kata Kamus Master", f"{len(kamus_master):,} kata")
    st.caption(f"• Kamus KBBI: {n_kbbi:,} kata")
    st.caption(f"• Kunci Jawaban Guru: {n_kunci:,} kata")
    st.caption(f"• Terminologi Biologi: {n_bio:,} kata")

    # Fitur tambah kamus via upload jika dibutuhkan saat demo
    uploaded_kamus = st.file_uploader("Unggah Tambahan Kamus (.xlsx)", type=["xlsx"])
    if uploaded_kamus:
        df_extra = pd.read_excel(uploaded_kamus)
        for col in df_extra.columns:
            for val in df_extra[col].dropna():
                kamus_master.update(tokenize(val))
        st.success(f"Kamus diperbarui! Total: {len(kamus_master):,} kata")

# Tiga Tab Utama untuk Demo HKI
tab1, tab2, tab3 = st.tabs([
    "🔍 1. Uji Coba Normalisasi Kalimat (Real-Time)",
    "📂 2. Pemrosesan Batch Dataset Siswa (Excel)",
    "📊 3. Evaluasi Akurasi & Ablation Study"
])

# ------------------------------------------
# TAB 1: UJI COBA SATU KALIMAT INTERAKTIF
# ------------------------------------------
with tab1:
    st.subheader("Simulasi Koreksi Ejaan Kontekstual (Tahap 1 s.d. Tahap 3)")
    contoh_default = (
        "Pada difusi terbantu channel protein, molekul atau ion yang tidak dapat melewati "
        "membran fosfolid secara langsung akan melewati saluran yang dibentuk oleh protein "
        "membran dan neutransmiter melewati celah sinopsi."
    )
    teks_uji = st.text_area("Masukkan Kalimat Jawaban Siswa:", value=contoh_default, height=110)

    if st.button("🚀 Jalankan Normalisasi Ejaan", type="primary"):
        with st.spinner("Mendeteksi typo dan menghitung probabilitas kontekstual IndoBERT..."):
            skenario_a, skenario_b, skenario_c, list_typo, detail_log = proses_satu_kalimat(
                teks_uji, kamus_master, kamus_kunci_guru, tokenizer_bert, model_bert, device,
                K=param_k, max_e=param_max_e, alpha=param_alpha
            )

        # Ringkasan Deteksi Tahap 1
        st.markdown("#### Hasil Deteksi Non-Word Error (Tahap 1)")
        if list_typo:
            st.warning(f"**Ditemukan {len(list_typo)} kata typo:** `{', '.join(list_typo)}`")
        else:
            st.success("Tidak ditemukan kata typo! Seluruh kata telah terdaftar di Kamus Master.")

        # Perbandingan 3 Skenario
        st.markdown("#### Perbandingan Hasil Normalisasi (Skenario A, B, dan C)")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("**Skenario A (Teks Asli / Baseline)**")
            st.write(skenario_a)
        with col_b:
            st.warning("**Skenario B (Damerau-Levenshtein Murni)**")
            st.write(skenario_b)
        with col_c:
            st.success("**Skenario C (IndoBERT MLM + BEDSpell)**")
            st.write(skenario_c)

        # Tabel Perhitungan Skor f(s,e)
        if detail_log:
            st.markdown("#### Detail Perhitungan Probabilitas BEDSpell: $f(s, e) = ((1 / (e + 1))^\\alpha) \\times s$")
            for label_typo, info in detail_log.items():
                with st.expander(f"📌 Kata Typo: **{label_typo}** ➔ Terpilih: **{info['terpilih_c']}**", expanded=True):
                    st.caption(f"**Kalimat Masking:** `{info['kalimat_masked']}`")
                    df_rank = pd.DataFrame(info["ranking"])
                    st.dataframe(
                        df_rank.style.format({
                            "Skor Konteks IndoBERT (s)": "{:.6e}",
                            "Skor Akhir BEDSpell f(s,e)": "{:.6e}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

# ------------------------------------------
# TAB 2: PEMROSESAN BATCH FILE EXCEL
# ------------------------------------------
with tab2:
    st.subheader("Pemrosesan Massal Dataset Jawaban Esai Siswa")
    st.markdown("Unggah file Excel yang memiliki kolom **`Jawaban Siswa`** untuk dinormalisasi secara otomatis.")
    
    uploaded_dataset = st.file_uploader("Unggah File DatasetSiswa.xlsx", type=["xlsx"], key="batch_upload")
    
    if uploaded_dataset:
        df_batch = pd.read_excel(uploaded_dataset)
        df_batch.columns = df_batch.columns.str.strip()
        st.write(f"**Preview Data Terunggah ({len(df_batch)} baris):**")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if "Jawaban Siswa" not in df_batch.columns:
            st.error("Kolom 'Jawaban Siswa' tidak ditemukan di dalam file Excel Anda!")
        else:
            max_baris = st.number_input(
                "Batasi jumlah baris yang diproses (agar tidak timeout di CPU Cloud):",
                min_value=1, max_value=len(df_batch), value=min(25, len(df_batch))
            )

            if st.button("⚙️ Proses Dataset Sekarang", type="primary"):
                df_proc = df_batch.head(max_baris).copy()
                progress_bar = st.progress(0, text="Memulai pemrosesan...")

                hasil_a, hasil_b, hasil_c, daftar_typo_col, jml_typo_col = [], [], [], [], []
                total_rows = len(df_proc)

                for idx, row in enumerate(df_proc.itertuples()):
                    teks_jawaban = getattr(row, "Jawaban_Siswa", df_proc.iloc[idx]["Jawaban Siswa"])
                    sa, sb, sc, l_typo, _ = proses_satu_kalimat(
                        teks_jawaban, kamus_master, kamus_kunci_guru,
                        tokenizer_bert, model_bert, device,
                        K=param_k, max_e=param_max_e, alpha=param_alpha
                    )
                    hasil_a.append(sa)
                    hasil_b.append(sb)
                    hasil_c.append(sc)
                    daftar_typo_col.append(", ".join(l_typo) if l_typo else "-")
                    jml_typo_col.append(len(l_typo))
                    progress_bar.progress((idx + 1) / total_rows, text=f"Memproses baris {idx + 1} dari {total_rows}...")

                df_proc["Kata Typo (Teks)"] = daftar_typo_col
                df_proc["Jumlah Typo"] = jml_typo_col
                df_proc["Teks_Skenario_A"] = hasil_a
                df_proc["Teks_Skenario_B"] = hasil_b
                df_proc["Teks_Skenario_C (BEDSpell)"] = hasil_c

                st.success("Pemrosesan selesai!")
                st.dataframe(df_proc, use_container_width=True)

                # Tombol Download Excel
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                    df_proc.to_excel(writer, index=False, sheet_name="Hasil_Normalisasi")
                
                st.download_button(
                    label="📥 Unduh Hasil Normalisasi (.xlsx)",
                    data=output_buffer.getvalue(),
                    file_name="Hasil_Akhir_Normalisasi_BEDSpell.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ------------------------------------------
# TAB 3: EVALUASI GROUND TRUTH
# ------------------------------------------
with tab3:
    st.subheader("Evaluasi Akurasi Spelling Correction (Ground Truth Terverifikasi)")
    st.markdown(
        "Komparasi performa koreksi ejaan antara **Skenario B (Edit Distance Murni)** dan "
        "**Skenario C (IndoBERT MLM + BEDSpell)** pada daftar kata typo istilah Biologi."
    )

    kamus_ground_truth = {
        "fosfolid"      : ("fosfolipid", "Pada difusi terbantu molekul tidak dapat melewati membran fosfolid secara langsung"),
        "diletakan"     : ("diletakkan", "Sel tumbuhan yang diletakan pada larutan hipotonik akan mengalami turgid"),
        "sinopsi"       : ("sinapsis", "Neurotransmiter dilepaskan menuju celah sinopsi untuk menghantarkan impuls"),
        "mennyebabkan"  : ("menyebabkan", "Perbedaan konsentrasi ini mennyebabkan terjadinya perpindahan molekul air"),
        "slgt"          : ("sglt", "Glukosa diangkut masuk ke dalam sel epitel usus menggunakan protein slgt"),
        "mebrannya"     : ("membrannya", "Sel hewan akan mengalami lisis karena mebrannya pecah saat berada di larutan hipotonik"),
        "semipermiabel" : ("semipermeabel", "Osmosis terjadi melalui membran semipermiabel dari konsentrasi rendah ke tinggi"),
        "kontrasporter" : ("kontransporter", "Protein kontrasporter memindahkan dua molekul secara bersamaan melalui membran"),
        "berkonfirmasi" : ("berkonformasi", "Protein pembawa akan berkonfirmasi untuk memindahkan molekul glukosa"),
        "neutransmiter" : ("neurotransmiter", "Vesikel eksositosis melepaskan neutransmiter ke luar sel saraf"),
        "moleul"        : ("molekul", "Difusi adalah perpindahan moleul dari konsentrasi tinggi ke konsentrasi rendah"),
        "diperantai"    : ("diperantarai", "Endositosis yang diperantai reseptor sangat selektif terhadap molekul ligan"),
        "memanfaat"     : ("memanfaatkan", "Transpor aktif sekunder memanfaat gradien konsentrasi ion natrium"),
        "melewatin"     : ("melewati", "Ion tidak dapat melewatin membran lipid bilayer secara langsung"),
    }

    if st.button("📈 Jalankan Evaluasi Ground Truth"):
        data_eval = []
        list_kamus = list(kamus_master)

        for typo, (aktual, kalimat_konteks) in kamus_ground_truth.items():
            kandidat_list = generasi_kandidat(typo, list_kamus, kamus_kunci_guru, K=param_k, max_e=param_max_e)
            tebakan_b = kandidat_list[0][0] if kandidat_list else typo
            
            kalimat_masked = kalimat_konteks.replace(typo, tokenizer_bert.mask_token)
            tebakan_c, _ = pilih_kandidat_bedspell(
                kalimat_masked, kandidat_list, tokenizer_bert, model_bert, device, alpha=param_alpha
            )
            if not tebakan_c:
                tebakan_c = typo

            data_eval.append({
                "Kata Typo": typo,
                "Kata Aktual (Ground Truth)": aktual,
                "Tebakan Skenario B": tebakan_b,
                "Validasi B": 1 if tebakan_b == aktual else 0,
                "Tebakan Skenario C (BEDSpell)": tebakan_c,
                "Validasi C": 1 if tebakan_c == aktual else 0
            })

        df_eval = pd.DataFrame(data_eval)
        total_uji = len(df_eval)
        akurasi_b = (df_eval["Validasi B"].sum() / total_uji) * 100
        akurasi_c = (df_eval["Validasi C"].sum() / total_uji) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Kata Typo Diuji", f"{total_uji} Kata")
        m2.metric("Akurasi Skenario B (Damerau-Levenshtein)", f"{akurasi_b:.2f}%")
        m3.metric("Akurasi Skenario C (BEDSpell)", f"{akurasi_c:.2f}%")

        st.dataframe(df_eval, use_container_width=True, hide_index=True)