# Predict-Spam-Message
This simple program can predict if text is Spam or normal text message 
Il compito di questo progetto è realizzare un modello di Machine Learning in grado di prevedere se un messaggio è SPAM o Ham(“E-mail that is generally desired and isn't considered spam.”).

Il set di dati per questo caso di studio può essere trovato su Kaggle.
I dati Consistono in 5574 messaggi di testo in inglese.

Di seguito la struttura del Notebook:
-Analisi dei dati e pulizia
-Visualizzazioni
-Modellazione Algoritmo
-Conclusione

```python
df=pd.read_csv("C:/Users/Alessandro/Desktop/spam.csv", encoding="latin-1")
# open the file in Desktop named spam.csv, encoding some unreadable characters
```
<img width="237" alt="Capture" src="https://user-images.githubusercontent.com/37181764/112315908-e472f480-8caa-11eb-9add-f74f6a14d8b9.PNG">

```python
df.rename({"v1":"Category", "v2":"Message"}, axis=1, inplace=True)
df.drop_duplicates(inplace = True)
#Checking for duplicates and removing them
```

### Visualization

```python
df["Category"].value_counts().plot(kind="bar")
# Number of message divided by category spam 653 and ham 4516
```
