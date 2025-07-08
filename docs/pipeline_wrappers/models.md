# models.py

`MODELS` enthält vorbereitete Instanzen verschiedener Klassifikatoren:

```python
    MODELS = {
        "logreg_l2": LogisticRegression(...),
        "rf_small":  RandomForestClassifier(...),
        "svm_rbf":   SVC(kernel="rbf", ...),
        "lda":       LinearDiscriminantAnalysis(),
        ...
    }
```
【F:src/FIT_python/pipeline_sex/models.py†L10-L31】

* **Logistische Regression** dient als leichtgewichtige Basislinie und lässt sich sowohl mit \(\ell_2\)- als auch \(\ell_1\)-Regularisierung betreiben. Sie setzt lineare Zusammenhänge voraus, liefert dafür interpretierbare Koeffizienten.

* **Random Forest** und **Extra Trees** bilden Ensembles von Entscheidungsbäumen. Sie kommen mit Ausreißern zurecht und erfassen nichtlineare Effekte. Der Lernaufwand steigt jedoch mit der Anzahl der Bäume und sehr tiefe Modelle neigen zum Overfitting.

* **K-nearest Neighbors** nutzt die Nachbarschaft im Merkmalsraum und benötigt daher keine explizite Modellanpassung. Gut geeignet ist es für kleinere Datensätze; mit steigender Stichprobe werden Vorhersagen jedoch langsam.

* **Support Vector Machines** mit linearem oder RBF-Kern liefern scharfe Trennungen auch bei wenigen Beobachtungen. Sie erfordern eine sorgfältige Wahl der Hyperparameter und sind weniger transparent hinsichtlich der Featurebedeutung.

* **Linear Discriminant Analysis** ist der Standard in dieser Pipeline. Die Methode geht von multivariaten Normalverteilungen mit identischen Kovarianzmatrizen aus und erzeugt lineare Entscheidungsgrenzen.

* **Gradient-Boosting-Modelle** wie XGBoost, LightGBM und CatBoost können sehr komplexe Zusammenhänge erfassen und liefern häufig beste Genauigkeiten. Sie bringen zahlreiche Parameter mit und erfordern mehr Rechenzeit.

Die Vielfalt der Modelle erleichtert Vergleiche zwischen einfachen linearen und leistungsfähigen nichtlinearen Ansätzen.

### Referenzen
* Breiman, L. (2001). "Random Forests." *Machine Learning*.
* Chen, T., & Guestrin, C. (2016). "XGBoost." *KDD*.
* Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine." *Annals of Statistics*.
* Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*.
