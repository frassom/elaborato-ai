# Analisi del sentimento su recensioni di farmaci
Lo scopo del progetto è di utilizzare classificatori Naive Bayes per classificare delle recensioni di farmaci come descritto in [Gräßer et al. 2018](https://dl.acm.org/doi/10.1145/3194658.3194677).

## Dataset
I datasets utilizzato è reperibile al link: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29.

## Requisiti
Il progetto è implementato in `python3`, vengono usate le librerie:
- `panda`
- `scikit-learn`

## Riproduzione dei risultati
Per la riproduzione dei risultati basta scaricare il dataset al link sopracitato, estrarlo nella stessa directory di `main.py` ed eseguire lo script che mostrerà i risultati come output da console; in alternativa può essere modificato `main.py`, per specificare quali file usare come training set e test set, nelle righe dove viene chiamata la funzione `load_dataset`.
