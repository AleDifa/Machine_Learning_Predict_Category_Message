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
<img width="400" alt="Capture" src="https://user-images.githubusercontent.com/37181764/112315908-e472f480-8caa-11eb-9add-f74f6a14d8b9.PNG">

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
```python
import matplotlib.pyplot as plt
%matplotlib inline
df["Category"].value_counts().plot(kind="pie", figsize=(5, 5), explode=[0,0.1], autopct = '%1.1f%%' )

# the library Matplotlib is used for visualization, let's see with pie chart how % of Spam and Ham in df 
# use value_counts().plot for plot the number in Category column
# autopct = '%1.1f%%'  show the percentage
plt.legend(["ham","spam"])
plt.title("Category Message")
plt.show()
```

<img width="800" alt="Capture2" src="https://user-images.githubusercontent.com/37181764/112318152-053c4980-8cad-11eb-8de4-1c1adaafdf02.PNG">

### Machine Learning Models
Nei modelli di machine learning è prassi comune convertire le variabili categoriche, ad esempio testo, nella loro rappresentazione numerica.
Convertiamo con CountVectorizer il testo di ogni messaggio in una rappresentazione numerica:

1)Create an instance of the CountVectorizer class.
2)Call the fit() function in order to learn a vocabulary from one or more documents.
3)Call the transform() function on one or more documents as needed to encode each as a vector.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
Vect_df= vectorizer.fit_transform(df["Message"])
```

```python
from sklearn.model_selection import train_test_split

# X= Vect_df
y=df["Category"]
#impostiamo i dati di X e y, stiamo cercando di trovare un associazione fra la category
#spam or Ham ed il testo dei messaggi.

# split into 80% training and 30% testing
X_train,X_test,y_train,y_test = train_test_split(Vect_df,y, test_size = 0.3, random_state = 10)
```

### Costruzione di un modello Naive Bayes
```python
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
#istance of our model

classifier.fit(X_train,y_train)
#fit the model to our dataset
#adesso la variabile classifier contiene il nostro modello addestrato. La fase di addestramento del modello prevede il calcolo della funzione di probabilità.
```

### Valutazioni
Usiamo la accuracy_score e la confusion_matrix per valutare il comportamento del nostro modello di Naive Bayes

```python
from sklearn.metrics import classification_report

prediction = classifier.predict(X_train)
print (classification_report(y_train, prediction))
```

<img width="650" alt="Capture3" src="https://user-images.githubusercontent.com/37181764/112319024-f1451780-8cad-11eb-9e49-fbd2d16d461e.PNG">
