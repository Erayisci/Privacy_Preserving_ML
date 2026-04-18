# Team Roadmap

## Overall order of operations

```
Phase A: Contract setup (BLOCKING ALL TEAMMATES)
  1. [Egehan]  Step 2: PPT Protocol interface + stubs + cache layer
  2. [Egehan]  Push → teammates can start coding against the interface

Phase B: Parallel implementation (everyone unblocked)
  3. [Egehan]      Step 3–5: attacks, metrics, runner, CLI
  4. [Onur]        DP module        → privacy_ml/ppt/dp.py
  5. [Eray]        BIE module       → privacy_ml/ppt/bie.py
  6. [SMPC owner]  SMPC module      → privacy_ml/ppt/smpc.py
  7. [Recon owner] Reconstruction attack → separate module
  8. [All]         Lit-review paragraphs in conference_101719.tex

Phase C: Integration & experiments
  9.  [Egehan]      Run 8 canonical configs on Colab → results/runs.jsonl
  10. [Recon owner] Run reconstruction evals in parallel
  11. [Egehan]      Generate Results table + figures

Phase D: Paper
  12. [All] Methods + Results + Discussion sections
  13. [All] Update slide 3 wording for final presentation
  14. [All] Proofread + submit
```

## Where we are right now

- ✅ Phase A.1 partially: spec + `models.py` + `data.py` committed on `main` (commits `b6001c1` + `3952ccd`)
- ⏳ Phase A.2 (PPT interface) is the immediate next thing — **teammates are blocked on this**
- Everything else depends on Phase A completing

## What Egehan does right now

**Priority 1 — today**: ship Step 2 (`ppt/base.py` + `stubs.py` + `cache.py`). ~20 min of work. This unblocks all three PPT teammates.

**Priority 2 — this week, in parallel with Step 3+**:

- Start drafting the PRICURE (Paper 3) literature-review paragraph. Doesn't need any code — the paper is ready. ~300–400 words in `conference_101719.tex` under `\section{Literature Review}`.
- Decide with the team: fork the repo to a team-owned org, or keep it on `Erayisci`? (Not blocking; governance call.)

**Priority 3 — later**:

- Run the 8-config experiment matrix in Colab once teammates deliver PPT modules
- Generate figures + fill in Results table

## What teammates do right now

After Step 2 lands, each teammate pulls `main` and:

- **Onur (DP)** — create `privacy_ml/ppt/dp.py` implementing the `PrivacyMechanism` Protocol with `layer="embedding"`; `apply(X)` adds Laplace noise to 128-d vectors, calibrated by an `epsilon` passed at construction.
- **Eray (BIE)** — create `privacy_ml/ppt/bie.py` with `layer="image"`; `apply(X)` tile-shuffles 150×150×1 images with a secret-key-derived permutation.
- **SMPC owner** — create `privacy_ml/ppt/smpc.py` with `layer="embedding"`; `apply(X)` produces an additive 2-share secret split + reconstruction of 128-d vectors.
- **Reconstruction attack owner** — separate module (not under `ppt/`); takes a trained encoder + a set of embeddings and trains a decoder to recover images.
- **All** — start lit-review paragraphs.

## Message to paste in the team chat

> **Ekip bilgilendirme — tasarım dokümanı ve backbone değişikliği**
>
> Repo'da `main` branch'ında tasarım dokümanı ve ilk kod iskeleti var:
>
> - Tasarım: `docs/superpowers/specs/2026-04-18-mia-design.md`
> - Kod: `privacy_ml/` klasörü altında
>
> **Önemli değişiklik — backbone**: sunumun 3. slaytında "pretrained CNN backbone" yazıyor ama biz hız için notebook'taki **custom 3-conv CNN**'i (pretrained değil, sıfırdan eğitiliyor) kullanmaya karar verdik. Sebepler tasarım dokümanının §2'sinde. Slayt metnini güncelleyeceğiz. ResNet18 kullanılmıyor.
>
> **Embedding boyutu**: 128-d (Dense(128, relu) katmanı çıktısı). DP ve SMPC bu 128-d vektör üzerinde çalışacak. BIE ise görüntüde (150×150) tile-shuffle yapacak — reference paper 1'e (Kiya et al.) uygun.
>
> **Sizin için interface**: yakında `privacy_ml/ppt/base.py` içinde `PrivacyMechanism` Protocol'ü olacak. Herkes kendi modülünü (`dp.py` / `bie.py` / `smpc.py`) bu interface'e uyacak şekilde yazacak. Interface push edildiğinde tekrar haber vereceğim.
>
> **Literatür özeti**: paperlarınız için paragrafları `conference_101719.tex` içine yazmaya başlayabilirsiniz. IEEE template kurulu, `\cite{bN}` stilini kullanın.
>
> Sorular varsa yazın.
