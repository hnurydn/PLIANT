# LIT-PCBA Per-Target İstatistiksel Anlamlılık Analizi — Centered vs Orijinal
(2026-08-11)

## Bağlam

Kullanıcı DUD-E analizinden sonra ("centered/orijinal DUD-E'de değerler baya
değişmişti") aynı kontrolün LIT-PCBA için de yapılmasını istedi. Tezin Tablo 4.8'i
zaten LIT-PCBA'nın 15 seçili (Target, Template) çifti için (v54 metodolojisi,
`planet-thesis-final.md` §v54) tek-örneklem + eşleştirilmiş Wilcoxon testi
içeriyordu — soru şuydu: hangi veri (centered mi orijinal mi) kullanılmıştı?

**Veri kaynakları:**
- Centered: TRUBA `Planet_model/litpcba_results_centered/*.csv` (124 satır/model)
- Orijinal: **Anzu sunucusu** (10.199.2.9, VPN gerekli — TRUBA'da YOK),
  `core6_results/` (6 model) + `missing4_results/` (4 model), 124 satır/model

Her ikisi de 15 seçili (Target, Template) çiftine filtrelendi (`ADRB2/4lde`,
`ALDH1/5l2n`, `ESR1_ago/2qzo`, `ESR1_ant/5ufx`, `FEN1/5fv7`, `GBA/2v3d`,
`IDH1/4umx`, `KAT2A/5mlj`, `MAPK1/4zzn`, `MTORC1/4dri`, `OPRK1/6b73`, `PKM2/3gr4`,
`PPARG/3b1m`, `TP53/3zme`, `VDR/3a2j`).

## Sonuç: Tez metodolojisi zaten DOĞRU — düzeltme gerekmiyor ✅

**Kanıt:** Tek-yönlü Wilcoxon testiyle (`alternative='greater'`, DUD-E analizi
sırasında keşfedildi — bkz. `dude_statistical_analysis.md` §0) v4_Projection için:
- **Orijinal veri: p=0,006226 → yuvarlanınca 0,006 — tezdeki değerle BİREBİR eşleşiyor**
- Centered veri: p=0,005127 → yuvarlanınca 0,005 (farklı)

**Bu, Tablo 4.8'in zaten orijinal (centered olmayan) veriyle hesaplandığını
kanıtlıyor.** DUD-E'nin aksine (tez hiç istatistik yapmamıştı, ben centered
veriyle yanlış başladım), LIT-PCBA'da tez baştan doğru veriyi kullanmış.

## Tam karşılaştırma tablosu — tek-örneklem Wilcoxon (tek-yönlü), 15 hedef

| Model | Centered AUROC (p) | **Orijinal AUROC (p, tez değeri)** |
|---|---|---|
| v4_Projection | 0,584 (p=0,00513) | **0,579 (p=0,00623 ≈ tez: 0,006)** ✅ |
| ESM_VG | 0,568 (p=0,0473, anlamlı!) | 0,556 (p=0,104, anlamsız) |
| v2_Softmax_m0 | 0,550 (p=0,0757) | 0,549 (p=0,0844) |
| NoESM_NoUW (baseline) | 0,537 (p=0,0757) | 0,537 (p=0,0757 — centering=0 etkisi) |
| v2_Softmax_m01 | 0,537 (p=0,104) | 0,529 (p=0,165) |
| NoESM_UW | 0,533 (p=0,151) | 0,533 (p=0,151 — centering=0) |
| ESM_UW | 0,516 (p=0,104) | 0,516 (p=0,104 — centering=0) |
| ESM_NoUW | 0,509 (p=0,319) | 0,509 (p=0,319 — centering=0) |
| v2_Mean_m0 | 0,504 (p=0,424) | 0,505 (p=0,424) |
| ESM_VG_Contrastive | 0,475 (p=0,835) | 0,481 (p=0,773) |

**Sadece v4_Projection her iki veri setinde de anlamlı.** ESM_VG centered veride
sınırda-anlamlı (p=0,047) ama orijinalde değil (p=0,104) — küçük ama DUD-E'deki
örüntüyle tutarlı bir sapma (VG-ailesi model, centering'den hafif faydalanıyor).

## Centering etkisi — DUD-E'den ÇOK daha küçük

| Model | Orijinal | Centered | Fark |
|---|---|---|---|
| ESM_VG | 0,556 | 0,568 | +0,012 |
| v2_Softmax_m01 | 0,529 | 0,537 | +0,008 |
| v4_Projection | 0,579 | 0,584 | +0,005 |
| v2_Softmax_m0 | 0,549 | 0,550 | +0,001 |
| ESM_UW, NoESM_NoUW, ESM_NoUW | — | — | **0,000** |
| NoESM_UW | 0,533 | 0,533 | −0,000 |
| v2_Mean_m0 | 0,505 | 0,504 | −0,001 |
| ESM_VG_Contrastive | 0,481 | 0,475 | −0,006 |

**Maksimum fark ±0,012** (DUD-E'de +0,090'a kadar çıkıyordu) — yön aynı (VG-ailesi
modeller hafifçe faydalanıyor, pose-blind modeller sıfır etkileniyor) ama etki
büyüklüğü çok daha küçük. Muhtemel neden: LIT-PCBA'nın decoy/aktif molekül
hazırlama süreci DUD-E'den farklı, ligandlar zaten cebe DUD-E'ye göre daha yakın
konumlandırılmış olabilir (doğrulanmadı, sadece gözlem).

## İkili (eşleştirilmiş) Wilcoxon — orijinal veri, 3/45 çift anlamlı

| Model 1 | Model 2 | Fark | p | Daha iyi |
|---|---|---|---|---|
| ESM_VG_Contrastive | v4_Projection | −0,098 | 0,0043 | v4_Projection |
| v2_Softmax_m01 | v4_Projection | −0,050 | 0,0125 | v4_Projection |
| ESM_UW | v4_Projection | −0,063 | 0,0256 | v4_Projection |

**v4_Projection'ın üstünlüğü ikili karşılaştırmalarda da tutarlı** — 3 model
karşısında anlamlı şekilde üstün, hiçbir modelden anlamlı şekilde geride değil.
Bu, tek-örneklem sonucunu (v4_Projection tek anlamlı model) doğrudan destekliyor.

## Sonuç — makale için

**LIT-PCBA tarafında ek bir düzeltme YAPILMASINA GEREK YOK** — tez Tablo 4.8
zaten metodolojik olarak sağlam (doğru veri + doğru tek-yönlü test). Makalede
doğrudan kullanılabilir. Tek fark: centering etkisinin DUD-E kadar dramatik
olmadığını gösteren §"Centering etkisi" tablosu, DUD-E'nin yanına küçük bir
karşılaştırma notu olarak eklenebilir ("LIT-PCBA'da centering etkisi ihmal
edilebilir düzeyde küçük, DUD-E'de ise VG-ailesi modellerde AUROC'u 0,09'a kadar
şişiriyor — bu fark muhtemelen iki benchmark'ın decoy/ligand hazırlama
protokollerindeki farklılıktan kaynaklanıyor").
