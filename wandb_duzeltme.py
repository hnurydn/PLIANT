import wandb
import pandas as pd
import sys
import os

'''# 1. Giriş ve API Hazırlığı
wandb.login()
api = wandb.Api(timeout=300) 

# --- AYARLAR ---
ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
# Bu sizin ID'nizdir, WandB bunu kullanarak doğru run'ı bulur ve üzerine yazar
MASTER_RUN_ID = "vector_gating_merged_final"

# Sadece eklemek istediğiniz 3. Run ID'si
RUN_3_ID = "dc4imbr1"

# 2. Mevcut Run'ı "Resume" ile Aç (Böylece üzerine ekleme yapar)
with wandb.init(project=NEW_PROJECT, id=MASTER_RUN_ID, resume="allow") as master:
    
    def append_run_data(run_id, min_step=None):
        path = f"{ENTITY}/{OLD_PROJECT}/{run_id}"
        print(f"\n>> {run_id} verisi okunuyor ve MASTER üzerine ekleniyor...", file=sys.stderr, flush=True)
        
        run = api.run(path)
        count = 0
        for row in run.scan_history():
            step = row.get("_step")
            
            # Çakışma olmaması için 1.420.000'den büyük adımları alıyoruz
            if min_step and step < min_step: continue
            
            # Veriyi temizle ve gönder
            metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
            master.log(metrics, step=int(step))
            
            count += 1
            if count % 100 == 0:
                print(f"   [{run_id}] {count} satır aktarıldı...", file=sys.stderr, flush=True)
        
        print(f">> {run_id} aktarımı bitti. Toplam {count} yeni satır eklendi.", file=sys.stderr, flush=True)

    # 3. Sadece 3. Run'ı ekle (1.420.000 adımından sonrası için 1420001 sınırı)
    append_run_data(RUN_3_ID, min_step=1420001)

print("\n!!! 3. RUN MEVCUT GRAFİĞE BAŞARIYLA EKLENDİ !!!", flush=True)'''

'''# 1. Giriş ve API Hazırlığı
wandb.login()
api = wandb.Api(timeout=300) 

# --- AYARLAR ---
ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
# Bu yeni model için farklı bir MASTER_ID veriyoruz
MASTER_RUN_ID = "esm_uw_1b11d_unified_final"

# Run ID'lerin
RUN_1_ID = "9z78legk" # 0 - 700.000
RUN_2_ID = "zt73fiek" # 700.000 - 1.440.000
RUN_3_ID = "k2ukfze3" # 1.440.000 - GÜNCEL (Hala devam ediyor)

# 2. Yeni Master Run'ı Başlat
with wandb.init(project=NEW_PROJECT, id=MASTER_RUN_ID, resume="allow", name="ESM_UW_Unified_Run") as master:
    
    def process_and_upload(run_id, label, max_step=None, min_step=None):
        path = f"{ENTITY}/{OLD_PROJECT}/{run_id}"
        print(f"\n>> {label} ({run_id}) verisi işleniyor...", file=sys.stderr, flush=True)
        
        try:
            run = api.run(path)
            count = 0
            # scan_history o an sunucuda ne kadar veri varsa hepsini çeker
            for row in run.scan_history():
                step = row.get("_step")
                
                # Sınırlandırma Filtreleri
                if max_step and step > max_step: continue
                if min_step and step < min_step: continue
                
                metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
                master.log(metrics, step=int(step))
                
                count += 1
                if count % 500 == 0:
                    print(f"   [{run_id}] {count} satır aktarıldı...", file=sys.stderr, flush=True)
            
            print(f">> {run_id} bitti. Toplam {count} satır.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"!! {run_id} aktarılırken hata oluştu: {e}", file=sys.stderr, flush=True)

    # 3. SIRAYLA AKTARIM
    
    # Parça 1: Başlangıçtan 700k'ya
    process_and_upload(RUN_1_ID, "RUN 1", max_step=700000)
    
    # Parça 2: 700.001'den 1.440.000'e
    process_and_upload(RUN_2_ID, "RUN 2", min_step=700001, max_step=1440000)
    
    # Parça 3: 1.440.001'den SON ADIMA KADAR (O anki tüm veriyi alır)
    process_and_upload(RUN_3_ID, "RUN 3 (GÜNCEL)", min_step=1440001)

print("\n!!! ESM_UW MODELİ İÇİN 3 RUN BİRLEŞTİRİLDİ !!!", flush=True)'''

'''# 1. Giriş ve API Hazırlığı
wandb.login()
api = wandb.Api(timeout=300) 

# --- AYARLAR ---
ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
# Bu model için özel MASTER_ID
MASTER_RUN_ID = "noesm_uw_512_unified_final"

# Birleştirilecek Run ID'leri
RUN_1_ID = "92b2p3xm" # 0 - 700.000
RUN_2_ID = "jmb0ipn8" # 700.000 - GÜNCEL (Hepsini alacağız)

# 2. Yeni Master Run'ı Başlat
with wandb.init(project=NEW_PROJECT, id=MASTER_RUN_ID, resume="allow", name="NoESM_UW_512_Unified") as master:
    
    def process_and_upload(run_id, label, max_step=None, min_step=None):
        path = f"{ENTITY}/{OLD_PROJECT}/{run_id}"
        print(f"\n>> {label} ({run_id}) verisi işleniyor...", file=sys.stderr, flush=True)
        
        try:
            run = api.run(path)
            count = 0
            # scan_history ile o anki tüm veriyi çekiyoruz
            for row in run.scan_history():
                step = row.get("_step")
                
                # Sınırlandırma Filtreleri
                if max_step and step > max_step: continue
                if min_step and step < min_step: continue
                
                # Metrikleri temizle ve gönder
                metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
                master.log(metrics, step=int(step))
                
                count += 1
                if count % 500 == 0:
                    print(f"   [{run_id}] {count} satır aktarıldı...", file=sys.stderr, flush=True)
            
            print(f">> {run_id} bitti. Toplam {count} satır aktarıldı.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"!! {run_id} aktarılırken hata oluştu: {e}", file=sys.stderr, flush=True)

    # 3. SIRAYLA AKTARIM
    
    # Parça 1: Başlangıçtan 700.000'e kadar
    process_and_upload(RUN_1_ID, "RUN 1", max_step=700000)
    
    # Parça 2: 700.001'den en son nerede kaldıysa oraya kadar (Hepsini alır)
    process_and_upload(RUN_2_ID, "RUN 2 (GÜNCEL)", min_step=700001)

print("\n!!! NoESM_UW_512 MODELİ İÇİN BİRLEŞTİRME TAMAMLANDI !!!", flush=True)'''


'''# 1. Giriş ve API Hazırlığı
wandb.login()
api = wandb.Api(timeout=300) 

# --- AYARLAR ---
ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
# Bu model için benzersiz MASTER_ID
MASTER_RUN_ID = "esm_vg_contrastive_unified_final"

# Birleştirilecek Run ID'leri
RUN_1_ID = "mkn0gjdx" # 0 - 180.000
RUN_2_ID = "ikdke5xm" # 180.000 - 280.000
RUN_3_ID = "sfxlen6x" # 280.000 - GÜNCEL

# 2. Yeni Master Run'ı Başlat
with wandb.init(project=NEW_PROJECT, id=MASTER_RUN_ID, resume="allow", name="ESM_VG_Contrastive_Unified") as master:
    
    def process_and_upload(run_id, label, max_step=None, min_step=None):
        path = f"{ENTITY}/{OLD_PROJECT}/{run_id}"
        print(f"\n>> {label} ({run_id}) verisi işleniyor...", file=sys.stderr, flush=True)
        
        try:
            run = api.run(path)
            count = 0
            # scan_history o an sunucuda ne kadar veri varsa hepsini çeker
            for row in run.scan_history():
                step = row.get("_step")
                
                # Sınırlandırma Filtreleri
                if max_step and step > max_step: continue
                if min_step and step < min_step: continue
                
                # WandB iç sistem verilerini ve NaN değerleri ayıkla
                metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
                master.log(metrics, step=int(step))
                
                count += 1
                if count % 500 == 0:
                    print(f"   [{run_id}] {count} satır aktarıldı...", file=sys.stderr, flush=True)
            
            print(f">> {run_id} bitti. Toplam {count} satır aktarıldı.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"!! {run_id} aktarılırken hata oluştu: {e}", file=sys.stderr, flush=True)

    # 3. SIRAYLA AKTARIM
    
    # Parça 1: Başlangıçtan 180.000'e kadar
    process_and_upload(RUN_1_ID, "RUN 1", max_step=180000)
    
    # Parça 2: 180.001'den 280.000'e kadar
    process_and_upload(RUN_2_ID, "RUN 2", min_step=180001, max_step=280000)
    
    # Parça 3: 280.001'den sonuna kadar (O an ne kadar log varsa hepsini alır)
    process_and_upload(RUN_3_ID, "RUN 3 (GÜNCEL)", min_step=280001)

print("\n!!! CONTRASTIVE MODELİ İÇİN 3 RUN BAŞARIYLA BİRLEŞTİRİLDİ !!!", flush=True)'''


# 1. WandB Girişi ve API Ayarları
wandb.login()
api = wandb.Api(timeout=600) 

# --- AYARLAR ---
ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
MASTER_RUN_ID = "esm_nouw_unified_final" # Sildiğin ID, şimdi sıfırdan oluşacak
SOURCE_RUN_ID = "erig627c"

# 2. Yeni Projede Run'ı Başlat (Gelişmiş ayarlar ile)
# settings kısmı timeout hatalarını engeller
run_settings = wandb.Settings(init_timeout=300)

with wandb.init(
    project=NEW_PROJECT, 
    id=MASTER_RUN_ID, 
    resume="allow", 
    name="ESM_noUW_Unified",
    settings=run_settings
) as master:
    
    print(f"\n>> {SOURCE_RUN_ID} verisi okunuyor ve sıralanıyor...", file=sys.stderr, flush=True)
    
    try:
        source_run = api.run(f"{ENTITY}/{OLD_PROJECT}/{SOURCE_RUN_ID}")
        
        # --- TÜM VERİYİ ÖNCE LİSTEYE TOPLUYORUZ ---
        rows = []
        for row in source_run.scan_history():
            rows.append(row)
        
        # --- VERİLERİ ADIM NUMARASINA GÖRE SIRALIYORUZ ---
        # Bu işlem grafikteki atlamaları ve "Tried to log..." hatalarını önler
        df = pd.DataFrame(rows).sort_values("_step")
        
        total_rows = len(df)
        print(f">> {total_rows} satır hazır. Aktarım başlıyor...", file=sys.stderr, flush=True)
        
        count = 0
        for _, row in df.iterrows():
            # Sistem metriklerini (_ ile başlayanlar) ve boş değerleri temizle
            metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
            
            # Adım numarasını koruyarak gönder
            master.log(metrics, step=int(row["_step"]))
            
            count += 1
            if count % 1000 == 0:
                print(f"   [{SOURCE_RUN_ID}] {count}/{total_rows} satır aktarıldı...", file=sys.stderr, flush=True)
        
        print(f">> İŞLEM BAŞARIYLA BİTTİ. Toplam {count} satır aktarıldı.", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"!! Hata oluştu: {e}", file=sys.stderr, flush=True)

print("\n!!! ESM_noUW MODELİ TERTEMİZ BİR ŞEKİLDE AKTARILDI !!!", flush=True)

# 1. Hazırlık
wandb.login()
api = wandb.Api(timeout=600) 

ENTITY = "h-nurydn-hacettepe-university"
OLD_PROJECT = "PLANET_TRAINING"
NEW_PROJECT = "PLANET_COMBINED" 
MASTER_RUN_ID = "esm_uw_1b11d_unified_final"

RUN_1_ID = "9z78legk"
RUN_2_ID = "zt73fiek"
RUN_3_ID = "k2ukfze3"

# 2. Master Run'ı Başlat
with wandb.init(project=NEW_PROJECT, id=MASTER_RUN_ID, resume="allow", name="ESM_UW_Unified_Run",settings=wandb.Settings(init_timeout=300)) as master:
    
    def process_and_upload(run_id, label, max_step=None, min_step=None):
        path = f"{ENTITY}/{OLD_PROJECT}/{run_id}"
        print(f"\n>> {label} ({run_id}) verisi okunuyor ve sıralanıyor...", file=sys.stderr, flush=True)
        
        run = api.run(path)
        
        # --- KRİTİK DEĞİŞİKLİK: ÖNCE LİSTEYE AL ---
        rows = []
        for row in run.scan_history():
            step = row.get("_step")
            if max_step and step > max_step: continue
            if min_step and step < min_step: continue
            rows.append(row)
        
        # --- KRİTİK DEĞİŞİKLİK: ADIMA GÖRE SIRALA ---
        # Bu satır, verilerin WandB'ye mutlaka 1, 2, 3... sırasında gitmesini sağlar.
        df = pd.DataFrame(rows).sort_values("_step")
        
        print(f">> {label} aktarılıyor...", file=sys.stderr, flush=True)
        count = 0
        for _, row in df.iterrows():
            metrics = {k: v for k, v in row.items() if not k.startswith("_") and pd.notnull(v)}
            master.log(metrics, step=int(row["_step"]))
            
            count += 1
            if count % 1000 == 0:
                print(f"   [{run_id}] {count} satır tamamlandı...", file=sys.stderr, flush=True)
        
        print(f">> {label} bitti. Toplam {count} satır.", file=sys.stderr, flush=True)

    # 3. SIRAYLA AKTARIM (Sıra çok önemli!)
    process_and_upload(RUN_1_ID, "RUN 1", max_step=700000)
    process_and_upload(RUN_2_ID, "RUN 2", min_step=700001, max_step=1440000)
    process_and_upload(RUN_3_ID, "RUN 3 (GÜNCEL)", min_step=1440001)

print("\n!!! İŞLEM HATASIZ TAMAMLANDI !!!", flush=True)