<div align="center">

# fetiai-v1-phiusiil-binclf-knn-skl-500k

**Group 16** · IF3070 Foundations of Artificial Intelligence · STEI ITB

<p>
  <img src="https://img.shields.io/badge/course-IF3070-3b5bdb?style=flat-square" alt="IF3070" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/institution-STEI%20ITB-1f2937?style=flat-square" alt="STEI ITB" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/year-2024%2F2025--1-6b7280?style=flat-square" alt="2024/2025-1" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/group-16-3b5bdb?style=flat-square" alt="Group 16" style="display:inline-block;margin:0;vertical-align:text-bottom" />
</p>

<p>
  <img src="https://img.shields.io/badge/task-tabular--classification-3b5bdb?style=flat-square" alt="tabular classification" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/parameters-500k-1f2937?style=flat-square" alt="500,002 parameters" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/dataset-PhiUSIIL-6b7280?style=flat-square" alt="PhiUSIIL" style="display:inline-block;margin:0;vertical-align:text-bottom" />
  <img src="https://img.shields.io/badge/licence-MIT-3b5bdb?style=flat-square" alt="MIT" style="display:inline-block;margin:0;vertical-align:text-bottom" />
</p>

</div>

**KNN (scikit-learn)** for phishing URL classification, served over HTTP as a single-model API.

```console
$ curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
    "features": {"URLLength": 31, "DomainLength": 25, "IsHTTPS": 1, "...": "all 49"}
  }'
{
  "model": "knn_sklearn",
  "label": 1,
  "verdict": "legitimate",
  "phishing_score": 0.0,
  "n_provided": 33,
  "n_imputed": 16,
  "coverage_ratio": 0.6735,
  "low_evidence": false
}
```

> **This is a coursework reimplementation, not a security product.** It is trained on a
> static 2023–24 dataset, has no threat intelligence, no blocklist, and no knowledge of any
> campaign newer than its training data. Do not use it to decide whether a link is safe.

**Algorithm**

`KNN (scikit-learn)` · `Trained on PhiUSIIL` · `SMOTE` · `Feature Engineering`

**Built with**

<p>
  <img src=".github/assets/logos/python.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> Python &nbsp;
  <img src=".github/assets/logos/scikit-learn.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> scikit-learn &nbsp;
  <img src=".github/assets/logos/numpy-dark.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> NumPy &nbsp;
  <img src=".github/assets/logos/pandas-dark.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> pandas &nbsp;
  <img src=".github/assets/logos/scipy.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> SciPy &nbsp;
  <img src=".github/assets/logos/docker.svg" width="18" height="18" align="top" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" /> Docker
</p>

**Links** — [Full application](https://github.com/fetiai/phishing-url-classifier) ·
[Live demo](https://phiusiil.faizath.com) ·
[Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

---

## What it does

It scores **one pre-extracted feature row** with one model and returns a verdict.

That is the whole scope, and the boundary is deliberate. This service has no page fetcher,
no URL feature extractor and no SSRF guard, because duplicating a network-facing security
control into four repositories is how the four copies drift apart. Turning a URL into the
49 features this API expects is the job of the
[full application](https://github.com/fetiai/phishing-url-classifier), which fetches the
page under a guard, extracts the features, and scores them against all four models at once.

So this is a **tabular** classifier, not a text one: the input is a 49-dimensional feature
vector, and the URL string never reaches the model.

| Route | Purpose |
|---|---|
| `POST /predict` | one row → verdict, score, and how much of the row was real |
| `POST /predict/batch` | up to 1000 rows |
| `GET /metadata` | the feature contract, the demoted features, and this model's metrics |
| `GET /healthz` | liveness |
| `GET /readyz` | ready once the golden-row self-test has passed |

## Quickstart

```bash
make install         # venv + pinned dependencies
make selftest        # prove the artifact reproduces its recorded prediction
make serve           # http://127.0.0.1:8000  (docs at /docs)
```

No training step and no download step: **the model is committed** (2.6 MB under
`model/`), so a fresh clone can serve immediately.

### With Docker

```bash
make docker-build && docker run -p 8000:8000 fetiai-v1-phiusiil-binclf-knn-skl-500k:local
```

The image **bakes the model in** rather than mounting it, so the image tag is a complete
description of what the service will predict. The build fails if the artifact does not
reproduce its own golden row.

## The request contract

Send all 49 feature columns. `null` is allowed and is the expected value for a feature the
caller could not determine — 12 of the 49 are permanently null, having failed the
extraction agreement gate in the parent project.

```jsonc
{
  "features": { "URLLength": 31, "DomainLength": 25, "IsHTTPS": 1, /* ...46 more */ },
  "url": "https://example.com/login",   // optional
  "domain": "example.com",              // optional
  "tld": "com",                         // optional
  "title": "Sign in"                    // optional
}
```

`GET /metadata` returns the exact 49 names in order. Three of them carry typos that are
preserved on purpose — `NoOfDegitsInURL`, `DegitRatioInURL`, `SpacialCharRatioInURL` —
because those names are what the training data means.

A **missing** feature is rejected by name; a **null** one is imputed. That distinction is
the point: an extractor that dropped a column and an extractor that honestly could not
determine a value are different faults with different fixes.

### Why the response reports coverage

Imputation fills a missing feature from the training distribution, and that distribution is
92.5% legitimate. A mostly-empty row therefore does not produce a *neutral* prediction — it
produces one biased toward **legitimate**, which is exactly the wrong direction for a
phishing detector.

So every response carries `n_provided`, `n_imputed`, `coverage_ratio` and `low_evidence`.
Without them a verdict drawn from six real values looks precisely as confident as one drawn
from all 49.

The optional `url`, `domain`, `tld` and `title` fields are worth sending when you have
them: 21 of the 49 features are derived from the URL string, and supplying it lets those be
**recomputed** rather than imputed.

## Results

Measured on a 28,081-row validation split. **The held-out file shipped with the dataset has
no labels, so there is no test score and none is claimed.**

| Model | Phishing recall | Phishing precision | Accuracy |
|---|---|---|---|
| **KNN (scikit-learn)** | **0.763** | **0.981** | **0.98073** |

**Read that accuracy against 0.9248.** The corpus is 92.48% legitimate, so answering
"legitimate" to everything scores 0.9248 while catching no phishing whatsoever. Accuracy
alone cannot tell a working detector from a constant; phishing recall can.

Class 0 is phishing and is the positive class throughout.

The scikit-learn and from-scratch implementations of this algorithm disagree on
**0.0356%** of the validation split. That number is the reason both exist: a
reimplementation with nothing to check it against is an assertion, not a result. The
counterpart lives in [`fetiai-v1-phiusiil-binclf-knn-scratch-500k`](../fetiai-v1-phiusiil-binclf-knn-scratch-500k).

`model/metrics.json` also carries the `legacy` profile, flagged `"leaky": true`. It
reconstructs the original notebook's configuration, which standardised each split by its
own mean and standard deviation — information no deployed model can have, since there is no
batch to average over when a single row arrives. It is kept as evidence of what the leak
was worth and is never presented as this model's result.

## What is in this repo

| Path | What it is |
|---|---|
| `model/knn_sklearn.joblib` | the trained model (500,002 stored values) |
| `model/fitted_stats.json` | **not optional** — the scaler, imputation values, clip bounds and mode tables |
| `model/manifest.json` | sha256 of every file above, verified at load |
| `model/golden_row.json` | one record with its expected vector and prediction |
| `phiusiil/` | the scoring path: schema, preprocessing, and this one model class |
| `server/` | loader, prediction, HTTP layer |

`fitted_stats.json` deserves the emphasis. The model alone cannot classify anything: it was
fitted on standardised inputs, and the numbers that produce that standardisation live in
that file. Publishing weights without it would be publishing something unusable.

It is plain JSON rather than a pickled transformer on purpose. A pickled estimator arrives
with a `fit` method attached, and the defect this whole pipeline exists to avoid is someone
calling it at serving time. Numbers that cannot be re-fitted cannot leak.

> **The model file is a pickle.** `joblib.load` executes code, so treat it as you would any
> executable, and note that it was produced by **scikit-learn 1.9.0** — loading it under a
> different version is unsupported. It holds a bare scikit-learn estimator rather than a
> wrapper class, so unpickling depends on scikit-learn alone and on nothing defined in this
> repository. If you want a model that loads without executing anything, the from-scratch
> counterpart in [`fetiai-v1-phiusiil-binclf-knn-scratch-500k`](../fetiai-v1-phiusiil-binclf-knn-scratch-500k) is plain a NumPy `.npz` of the reference matrix and its labels.

## Verification

```bash
make selftest    # golden row, offline
make test        # golden row + HTTP contract + naming
make namecheck   # provenance hygiene
```

The check that carries the weight is the **golden row**: one real record, its expected
49-feature vector, and its expected prediction, pushed through the whole path and compared
**exactly**. The vector and the prediction are asserted separately, because a wrong vector
means the preprocessing drifted while a right vector with a wrong label means the model
artifact did — different faults, different fixes.

Equality is bitwise on float32, never a tolerance. A tolerance-based comparison would pass
while a fitted statistic quietly differed, which is the one thing the test exists to catch.

`origin.json` records where every copied file came from, including the parent bundle's own
hashes, so drift is detectable without the parent repository present.

## Licence and data

MIT, as in the parent project. See `LICENSE`.

This model was trained in part on the PhiUSIIL Phishing URL Dataset (Prasad & Chandra), available from the UCI Machine Learning Repository, licensed under CC BY 4.0.

The dataset is the UCI PhiUSIIL Phishing URL Dataset (ID 967).

**This artifact embeds training data.** k-nearest neighbours has no learned parameters —
fitting is memorising — so `model/knn_sklearn.joblib` *is* the 10,000-row scaled reference set that
the classifier searches at prediction time. The dataset's attribution therefore travels with
this model, not only with the dataset.

---

## Team

<div align="center">

<table>
  <tr>
    <td width="220" align="center" valign="top">
      <a href="https://github.com/thalitazhrr">
        <img src="https://github.com/thalitazhrr.png?size=140" width="120" height="120" alt="Thalita Zahra Sutejo" style="border-radius:50%" />
      </a>
      <br /><br />
      <b>Thalita Zahra Sutejo</b><br />
      18222023
      <br /><br />
      <a href="https://github.com/thalitazhrr">
        <img src=".github/assets/github.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        thalitazhrr
      </a>
      <br />
      <a href="https://www.linkedin.com/in/thalitazahras/">
        <img src=".github/assets/linkedin.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        thalitazahras
      </a>
    </td>
    <td width="220" align="center" valign="top">
      <a href="https://github.com/IrfanMusthofa">
        <img src="https://github.com/IrfanMusthofa.png?size=140" width="120" height="120" alt="Irfan Musthofa" style="border-radius:50%" />
      </a>
      <br /><br />
      <b>Irfan Musthofa</b><br />
      18222056
      <br /><br />
      <a href="https://github.com/IrfanMusthofa">
        <img src=".github/assets/github.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        IrfanMusthofa
      </a>
      <br />
      <a href="https://www.linkedin.com/in/irfanmusthofa/">
        <img src=".github/assets/linkedin.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        irfanmusthofa
      </a>
    </td>
    <td width="220" align="center" valign="top">
      <a href="https://github.com/EleanorCordelia">
        <img src="https://github.com/EleanorCordelia.png?size=140" width="120" height="120" alt="Eleanor Cordelia" style="border-radius:50%" />
      </a>
      <br /><br />
      <b>Eleanor Cordelia</b><br />
      18222059
      <br /><br />
      <a href="https://github.com/EleanorCordelia">
        <img src=".github/assets/github.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        EleanorCordelia
      </a>
      <br />
      <a href="https://www.linkedin.com/in/eleanorcordelia/">
        <img src=".github/assets/linkedin.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        eleanorcordelia
      </a>
    </td>
    <td width="220" align="center" valign="top">
      <a href="https://github.com/faizath">
        <img src="https://github.com/faizath.png?size=140" width="120" height="120" alt="Muhammad Faiz Atharrahman" style="border-radius:50%" />
      </a>
      <br /><br />
      <b>Muhammad Faiz Atharrahman</b><br />
      18222063
      <br /><br />
      <a href="https://github.com/faizath">
        <img src=".github/assets/github.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        faizath
      </a>
      <br />
      <a href="https://www.linkedin.com/in/faizath/">
        <img src=".github/assets/linkedin.svg" width="14" height="14" alt="" style="display:inline-block;margin:0;vertical-align:text-bottom" />
        faizath
      </a>
    </td>
  </tr>
</table>

</div>

---

<div align="center">

IF3070 Foundations of Artificial Intelligence · STEI ITB · 2024/2025-1

More at **[fetiai.github.io](https://fetiai.github.io/)**

</div>
