# Pooled Correlation and Within-Group Agreement in Predicting Facial Attractiveness Ratings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24562444.svg)](https://doi.org/10.5281/zenodo.24562444)

[Published preprint and materials](https://doi.org/10.5281/zenodo.24562444) · [Interactive page](https://matthew-kiriakov.github.io/face-attractiveness-study/page/) · [Download the release ZIP](https://github.com/Matthew-Kiriakov/face-attractiveness-study/releases/latest/download/face_attractiveness_public.zip)

This study examines how scale and group composition affect the evaluation of predictions of facial attractiveness ratings. On 1,196 SCUT photographs, restoring the same predictions and observations to the 1–5 scale changed pooled Pearson r from 0.672 to 0.824 while the mean of six within-group correlations stayed at 0.678.

- [English manuscript and appendices (PDF)](manuscript/en/face_attractiveness_en.pdf) · [English source](manuscript/en/paper_en.md)
- [Russian manuscript and appendices (PDF)](manuscript/ru/face_attractiveness_ru.pdf) · [Russian source](manuscript/ru/paper_ru.md)
- [Interactive page source](page/index.html) · [Page data schema](page/DATA_SCHEMA.md)
- Saved aggregate outputs in `results/` and the manuscript directories.
- Selected analysis and feature-definition code in `scripts/` and `js/`.

## Reproduce the synthetic example

From this directory, run:

```text
python scripts/group_correlation_demo.py
```

The example uses eight artificial observations and Python's standard library. It yields pooled correlation 0.800, zero correlation within both groups, and pooled correlation 0.894 for the group-mean forecast.

Open `page/index.html` directly to explore the interactive summary. It reads bundled aggregate data on disk. The page also runs on a static HTTP host. It uses a generic MakeHuman head mesh and can display a 2D fallback when the external three.js library is unavailable. See [page asset notices](page/LICENSES.md).

## Data and analysis code

Obtain source data directly from the providers under their applicable conditions. SCUT-FBP5500 is available for non-commercial research. The MEBeauty 2.1.0 terms permit non-commercial academic facial-attractiveness research, including aggregate publication, and prohibit redistribution of the dataset. The release includes summary tables, charts and synthetic observations instead of photographs, ratings, row-level metadata, embeddings or predictions.

- [SCUT-FBP5500 provider documentation](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release#5-citation-and-contact)
- [MEBeauty dataset card](https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction)
- [MEBeauty terms for version 2.1.0](https://huggingface.co/datasets/dr-irina-lebedeva/MEBeauty-Facial-Beauty-Prediction/blob/main/LICENSE)
- [10k US Adult Faces Database](https://www.wilmabainbridge.com/facememorability2.html), one of the image sources listed by SCUT. Its provider requires citation and acceptance of a research-use agreement for direct access. The full agreement is behind the access form. This archive contains no material from the 10k collection itself. See Bainbridge, Isola, and Oliva (2013), [The intrinsic memorability of face images](https://wilmabainbridge.com/papers/jepg-2013.pdf).

The analysis scripts document the original calculations and expect locally prepared inputs. The original run guards and registered input checksums are included as provenance. The archive alone reproduces the synthetic example and the static interactive page. Reproducing the empirical runs also requires provider data, the intermediate features, the models used for extraction and the recorded environment. The manuscript's technical appendix documents the main settings and split. `evaluation_diagnostics.py`, `scut_editorial_checks.py`, and `review_sensitivity.py` identify the code for the post hoc comparisons.

The protocol record was internal and had no independent timestamp. The files here include selected aggregate outputs from the historical runs. Some earlier summaries describe comparisons before the editorial diagnostics. Use the manuscript for the final interpretation and its appendices for the data-access chronology.

The publication date of a GitHub release or a Zenodo deposit dates that snapshot. The earlier internal protocol entries retain the dating status described in the manuscript.

## Licenses

The project's original code is available under the [MIT License](LICENSE). The author-created text and figures are available under [CC BY 4.0](LICENSE-CONTENT.md). The underlying datasets and third-party assets retain their own terms. See [page asset notices](page/LICENSES.md).

## Citation

Kiriakov, M. (2026). *Pooled Correlation and Within-Group Agreement in Predicting Facial Attractiveness Ratings* [Preprint]. Zenodo. [https://doi.org/10.5281/zenodo.24562444](https://doi.org/10.5281/zenodo.24562444).

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.
