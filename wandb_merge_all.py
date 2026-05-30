"""
WandB PLANET runs merger
========================
Aynı modelin farklı checkpoint'lerden devam edilmiş runlarını
NEW_PROJECT içinde tek bir "unified" run'a birleştirir.

Kesim mantığı (kullanıcı isteğine göre):
  Her run'dan sadece  step < (kronolojik olarak sonraki run'ın min_step'i)
  olan satırlar alınır. Yani bir run checkpoint sonrası fazladan adım
  attıysa, o fazla kısım atlanır; bir sonraki run'ın başladığı yerde kesilir.

Cutoff'lar API'den OTOMATİK okunur — manuel sayı girilmez.

NOT: Eğer NEW_PROJECT içinde bu MASTER_ID'ler önceden varsa "resume" eder
ve aynı step'lere yazmayı dener. Tertemiz başlamak istersen WandB UI'dan
ilgili run'ları sil, sonra çalıştır.
"""

import sys
import wandb
import pandas as pd

# ============================== AYARLAR ==============================
ENTITY       = "h-nurydn-hacettepe-university"
OLD_PROJECT  = "PLANET_TRAINING"
NEW_PROJECT  = "PLANET_COMBINED_RESULT"

# (master_run_id, display_name, [run_id_kronolojik_sırada])
# Not: sıra önemli değil, script min_step'e göre otomatik sıralar.
# İstemediğin satırı yorum satırına alabilirsin.
MODELS = [
    ("ems_uw_vector_gating_1b11d_unified",
     "EMS_UW_VectorGating_1b11d_Unified",
     ["mt7codlk", "6rf5hest", "dc4imbr1"]),

    ("esm_nouw_1b11d_unified",
     "ESM_noUW_1b11d_Unified",
     ["erig627c"]),  # tek run; sadece NEW_PROJECT'e kopyalanır

    ("esm_uw_1b11d_unified",
     "ESM_UW_1b11d_Unified",
     ["9z78legk", "zt73fiek", "k2ukfze3"]),

    ("esm_uw_vg_contrastive_1b11d_unified",
     "ESM_UW_VG_Contrastive_1b11d_Unified",
     ["mkn0gjdx", "ikdke5xm", "sfxlen6x", "jh11ik8z"]),

    ("esm_uw_vg_contrastive_v2_mean_m0_unified",
     "ESM_UW_VG_Contrastive_v2_Mean_m0_Unified",
     ["ijahk9l6", "fidq6bw2", "r76gkxii"]),

    ("esm_uw_vg_contrastive_v2_softmax_m0_unified",
     "ESM_UW_VG_Contrastive_v2_Softmax_m0_Unified",
     ["nbgsttm6", "3efxasfx", "pme4gdmz"]),

    ("esm_uw_vg_contrastive_v2_softmax_m01_unified",
     "ESM_UW_VG_Contrastive_v2_Softmax_m01_Unified",
     ["in5lh0dw", "1w0b8k1k", "vg0m1yyx"]),

    ("esm_uw_vg_contrastive_v4_projection_scratch_unified",
     "ESM_UW_VG_Contrastive_v4_Projection_Scratch_Unified",
     ["lvs1v8zp", "trk2biao", "zmalp2a6"]),

    ("noesm_uw_1b11d_512_unified",
     "NoESM_UW_1b11d_512_Unified",
     ["92b2p3xm", "jmb0ipn8", "hlf2yab8", "0xsxc96i"]),
]


# ============================== YARDIMCI ==============================
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fetch_run_df(api, run_id: str) -> pd.DataFrame | None:
    """Bir run'ın tüm history'sini DataFrame olarak döner (step'e göre sıralı)."""
    try:
        run = api.run(f"{ENTITY}/{OLD_PROJECT}/{run_id}")
        rows = list(run.scan_history())
        if not rows:
            log(f"     [{run_id}] BOŞ run")
            return None
        df = pd.DataFrame(rows)
        if "_step" not in df.columns:
            log(f"     [{run_id}] _step kolonu yok, atlanıyor")
            return None
        df = df.sort_values("_step").reset_index(drop=True)
        log(f"     [{run_id}] {len(df):>6} satır | "
            f"step {int(df['_step'].min())} → {int(df['_step'].max())}")
        return df
    except Exception as e:
        log(f"     [{run_id}] !! API hatası: {e}")
        return None


# ============================== ANA İŞ ==============================
def process_model(api, master_id: str, display_name: str, run_ids: list[str]) -> str:
    log(f"\n{'='*70}")
    log(f"MODEL: {display_name}")
    log(f"master_id: {master_id}")
    log(f"runs:      {run_ids}")
    log("=" * 70)

    # 1) Tüm run'ları çek
    log("\n[1/3] Run verileri WandB'den çekiliyor...")
    dfs: list[tuple[str, pd.DataFrame]] = []
    for rid in run_ids:
        df = fetch_run_df(api, rid)
        if df is not None:
            dfs.append((rid, df))

    if not dfs:
        log(">> Hiçbir run verisi alınamadı, model atlanıyor.")
        return "EMPTY"

    # 2) min_step'e göre kronolojik sırala, sonra cutoff'ları belirle
    dfs.sort(key=lambda x: x[1]["_step"].min())
    cutoffs: list[int | None] = []
    log("\n[2/3] Kesim noktaları:")
    for i, (rid, df) in enumerate(dfs):
        next_min = int(dfs[i + 1][1]["_step"].min()) if i + 1 < len(dfs) else None
        cutoffs.append(next_min)
        my_min, my_max = int(df["_step"].min()), int(df["_step"].max())
        if next_min is not None:
            kept = int((df["_step"] < next_min).sum())
            dropped = len(df) - kept
            log(f"   {rid}: step ∈ [{my_min}, {my_max}]  →  KEEP step<{next_min} "
                f"({kept} satır, {dropped} atılacak)")
        else:
            log(f"   {rid}: step ∈ [{my_min}, {my_max}]  →  HEPSİ (son run)")

    # 3) Master run'ı aç ve sıralı şekilde yaz
    log(f"\n[3/3] '{master_id}' master run'a aktarım başlıyor...")
    total_logged = 0
    try:
        with wandb.init(
            project=NEW_PROJECT,
            id=master_id,
            resume="allow",
            name=display_name,
            settings=wandb.Settings(init_timeout=600),
        ) as master:
            for (rid, df), cutoff in zip(dfs, cutoffs):
                sub = df if cutoff is None else df[df["_step"] < cutoff]
                if len(sub) == 0:
                    log(f"   -> {rid}: filtreden sonra boş, atlandı")
                    continue
                log(f"   -> {rid}: {len(sub)} satır yazılıyor...")

                # to_dict('records') iterrows'tan ~10x hızlı
                for row in sub.to_dict("records"):
                    step = int(row["_step"])
                    metrics = {
                        k: v for k, v in row.items()
                        if not k.startswith("_") and pd.notnull(v)
                    }
                    master.log(metrics, step=step)
                    total_logged += 1
                    if total_logged % 2000 == 0:
                        log(f"      ... toplam {total_logged} satır yazıldı")

                log(f"   -> {rid}: tamam")

        log(f"\n>> BİTTİ. Master'a TOPLAM {total_logged} satır yazıldı.")
        return "OK"
    except Exception as e:
        log(f"\n!! WandB init/log hatası: {e}")
        return f"FAIL ({type(e).__name__})"


def main() -> None:
    wandb.login()
    api = wandb.Api(timeout=600)

    results: list[tuple[str, str]] = []
    for master_id, display_name, run_ids in MODELS:
        try:
            status = process_model(api, master_id, display_name, run_ids)
        except Exception as e:
            log(f"\n!! [{display_name}] BEKLENMEDİK HATA: {e}")
            status = f"CRASH ({type(e).__name__})"
        results.append((display_name, status))

    # Özet tablo
    log("\n\n" + "=" * 70)
    log("SONUÇ ÖZETİ")
    log("=" * 70)
    for name, status in results:
        log(f"  [{status:<10}] {name}")
    log("=" * 70)


if __name__ == "__main__":
    main()
