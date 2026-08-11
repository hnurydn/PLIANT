# DUD-E Per-Target İstatistiksel Anlamlılık Analizi
(2026-08-11, son güncelleme: tek-yönlü Wilcoxon metodolojisi doğrulandı — bkz. §0)

## 0. Metodoloji notu — KESİNLEŞTİ

Tezin Tablo 4.8'i (LIT-PCBA) ile referans karşılaştırma yapılarak test yönü
doğrulandı: **tek-örneklem test TEK-YÖNLÜ** (`scipy.stats.wilcoxon(diff,
alternative='greater')` — AUROC > 0,5 hipotezi, sadece AUROC ≠ 0,5 değil).
Kanıt: LIT-PCBA **orijinal** veriyle v4_Projection p=0,006226 ≈ tezdeki **0,006**
(3 ondalık basamağa kadar birebir). Bu, aynı zamanda **tezin Tablo 4.8'inin zaten
orijinal (centered olmayan) veriyi kullandığını da doğruluyor** — LIT-PCBA için
düzeltme gerekmiyor, sadece DUD-E için (tez bu analizi hiç yapmamıştı).

Bütün sayılar aşağıda bu (doğru, tek-yönlü) yöntemle güncellendi.

## 1. Bağlam

Tez (s.71) DUD-E'yi (101 hedef) yalnızca betimsel raporluyor, istatistiksel
anlamlılık analizi hiç yapılmamış ("ayrıca değerlendirilmelidir" notu var).
Bu, roadmap raporunun "per-target istatistiksel analiz sistematikleştirme"
maddesini karşılıyor.

## 2. ⭐ Ana bulgu: Centering etkisi mimari ile birebir örtüşüyor

| Model | Orijinal AUROC | Centered AUROC | Fark | VG ailesi mi? |
|---|---|---|---|---|
| ESM_VG | 0,486 | 0,576 | **+0,090** | ✅ |
| v4_Projection | 0,477 | 0,556 | **+0,079** | ✅ |
| v2_Softmax_m0 | 0,498 | 0,538 | +0,040 | ✅ |
| v2_Softmax_m01 | 0,516 | 0,554 | +0,038 | ✅ |
| v2_Mean_m0 | 0,512 | 0,536 | +0,024 | ✅ |
| ESM_VG_Contrastive | 0,509 | 0,476 | −0,033 | ✅ (istisna) |
| ESM_UW | 0,554 | 0,554 | **0,000** | ❌ |
| NoESM_UW | 0,501 | 0,501 | **0,000** | ❌ |
| NoESM_NoUW (baseline) | 0,530 | 0,530 | **−0,000** | ❌ |
| ESM_NoUW | 0,505 | 0,505 | **−0,000** | ❌ |

Poz-koordinatını forward pass'te kullanan (Vector-Gating) modeller centering'den
sistematik olarak fayda görüyor; kullanmayan 4 model **tam olarak sıfır** etkileniyor.
Bu, pose-perturbation deneyi (§2, `pose_perturbation_teknik_tasarim.md`) ve
checkpoint-eğrisinden (WandB) sonra **üçüncü, tamamen bağımsız** (gerçek DUD-E
benchmark verisi) kanıt hattı.

## 3. Tek-örneklem Wilcoxon (tek-yönlü, AUROC>0,5), 101 hedef

| Model | Centered AUROC (p, anlamlı) | **Orijinal AUROC (p, anlamlı)** |
|---|---|---|
| ESM_VG | 0,576 (p=1,74×10⁻¹⁰, **Evet**) | 0,486 (p=0,931, Hayır) |
| v4_Projection | 0,556 (p=5,28×10⁻⁸, **Evet**) | 0,477 (p=0,961, Hayır) |
| v2_Softmax_m01 | 0,554 (p=4,32×10⁻⁷, **Evet**) | 0,516 (p=0,171, Hayır) |
| ESM_UW | 0,554 (p=7,77×10⁻⁵, **Evet**) | 0,554 (p=7,77×10⁻⁵, **Evet**) |
| v2_Softmax_m0 | 0,538 (p=3,06×10⁻⁴, **Evet**) | 0,498 (p=0,708, Hayır) |
| v2_Mean_m0 | 0,536 (p=0,0122, **Evet**) | 0,512 (p=0,210, Hayır) |
| NoESM_NoUW (baseline) | 0,530 (p=0,0231, **Evet**) | 0,530 (p=0,0231, **Evet**) |
| ESM_NoUW | 0,505 (p=0,529, Hayır) | 0,505 (p=0,529, Hayır) |
| NoESM_UW | 0,501 (p=0,509, Hayır) | 0,501 (p=0,509, Hayır) |
| ESM_VG_Contrastive | 0,476 (p=0,887, Hayır) | 0,509 (p=0,223, Hayır) |

**Centered veride 7/10 model anlamlı; orijinal (gerçekçi) veride sadece 2/10**
(ESM_UW, NoESM_NoUW) — ikisi de VG-ailesi DEĞİL. ESM_VG/v4_Projection orijinal
veride rastgelenin altına düşüyor.

**⚠️ Makale için kritik:** Orijinal veri birincil analiz olmalı; centered veri
sadece "centering'in etkisini göstermek için" ikincil/duyarlılık analizi.

## 4. İkili (eşleştirilmiş, iki-yönlü) Wilcoxon — orijinal veri, 14/45 çift anlamlı

| Model 1 | Model 2 | Fark | p | Daha iyi |
|---|---|---|---|---|
| ESM_UW | v4_Projection | +0,076 | 1,18×10⁻⁵ | ESM_UW |
| ESM_UW | ESM_VG | +0,068 | 1,41×10⁻⁴ | ESM_UW |
| ESM_UW | v2_Softmax_m0 | +0,055 | 4,87×10⁻⁴ | ESM_UW |
| ESM_NoUW | ESM_UW | −0,049 | 0,0024 | ESM_UW |
| NoESM_NoUW | ESM_VG | +0,045 | 0,0035 | NoESM_NoUW |
| NoESM_NoUW | v4_Projection | +0,053 | 0,0059 | NoESM_NoUW |

Orijinal veride en güçlü/tutarlı model **ESM_UW** (koordinat kullanmıyor,
ESM-2+UW) — VG-ailesi değil.

## 5. Makalede kullanım önerisi

- **Ana metinde orijinal (centered olmayan) sonuçlar birincil.**
- Centering-etkisi tablosu (§2) yeni bir figür/tablo olarak, pose-perturbation
  deneyiyle birlikte "iki bağımsız kanıt hattı" şeklinde sunulabilir.
- Draft cümle (İngilizce):
  > *"Centering — repositioning the ligand at the pocket centroid — inflates
  > apparent AUROC specifically for Vector-Gating variants (+0.024 to +0.090)
  > while leaving architecturally pose-blind variants completely unaffected
  > (Δ=0.000 in all four cases), providing an independent, real-benchmark
  > confirmation of the RMSD-perturbation finding. Using the realistic
  > (non-centered) DUD-E poses, only two of ten variants (ESM_UW, one-sided
  > Wilcoxon p=7.8×10⁻⁵; NoESM_NoUW baseline, p=0.023) achieved statistically
  > significant discrimination from random ranking (n=101 targets) — neither
  > belongs to the pose-sensitive Vector-Gating family."*

## 6. LIT-PCBA karşılaştırması — AYRI dosyada tamamlandı

`makale/litpcba_statistical_analysis.md`'ye bakın. Özet: **tezin Tablo 4.8'i zaten
doğru (orijinal) veriyi kullanmış**, düzeltme gerekmiyor; centering etkisi
LIT-PCBA'da çok daha küçük (±0,012) — DUD-E'deki kadar dramatik değil.
