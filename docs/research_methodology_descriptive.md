# Pipeline Methodology Overview (Descriptive)

This document paraphrases the original `research_methodology.md` without mentioning internal function names. It describes the data flow and processing steps in words only.

## Sex Classification Pipeline (English)
The workflow begins by loading the raw CSV tables located in `data/raw`. During this stage, the column names are harmonised and numeric columns are converted so that all tables have a consistent structure. The cleaned versions are saved under `data/cleaned` for later use.

After cleaning, the footprints belonging to each individual are grouped together. These groups are then split by species into separate training and testing files. If required, an additional inference set is created. The splitting procedure ensures that every individual appears in only one of the sets, keeping the proportion of male and female footprints balanced. When cross-validation is enabled, an equal number of folds is drawn so that each fold has a comparable sex ratio.

For model preparation a scikit-learn `Pipeline` is assembled from several optional components. Missing values can be imputed with a random forest based iterative method. Outliers may be handled either by clipping extreme values or by winsorising according to a z-score threshold. Features are scaled using either standard or robust scaling. Dimensionality reduction is supported via PCA, UMAP or t-SNE, and the feature set can be narrowed down through techniques such as forward selection, random-forest importance, variance thresholding, univariate tests or the LASSO. In the final step one of several classifiers, for example logistic regression, support vector machines or gradient boosting, is added to the pipeline.

During training the prepared pipeline is applied to each species separately using five-fold cross-validation. Balanced accuracy serves as the evaluation metric. The processing time of each configuration is measured and both the accuracies and the chosen hyperparameters are recorded in `results/data/raw_results.csv`. Afterwards every configuration is refitted on the entire training data and stored with `joblib` under `results/data/sex_models`. The best model for each species is duplicated in `sex_models_best` so that predictions can be made easily.

Reproducibility is supported by caching intermediate results, saving split files in the Parquet format and relying on a globally fixed random seed. The resulting metrics can be compared with related literature on sex classification using gait characteristics, such as the classical works on PCA and the LASSO.

## Methodik: Sex-Klassifikations-Pipeline (Deutsch)
Der Arbeitsablauf beginnt damit, dass die Rohdaten aus `data/raw` eingelesen werden. Dabei werden die Spalten vereinheitlicht und numerische Felder so umgewandelt, dass alle Tabellen das gleiche Format besitzen. Die bereinigten Dateien werden unter `data/cleaned` abgelegt.

Im Anschluss werden die Spuren jedes Individuums zusammengefasst und pro Tierart in Trainings-, Test- und optional Inferenzsätze aufgeteilt. Dabei wird darauf geachtet, dass ein Individuum nur in einem dieser Sätze vorkommt und dass das Verhältnis der Geschlechter ausgewogen bleibt. Bei Bedarf werden zusätzlich gleichmäßige Folds für eine Kreuzvalidierung erzeugt.

Für das Modell wird eine scikit-learn-Pipeline aus mehreren optionalen Bausteinen zusammengesetzt. Fehlende Werte können mit einer iterativen Methode auf Basis von Random Forests imputiert werden. Ausreißer lassen sich entweder durch Clipping extremer Werte oder mittels Winsorisieren nach einem Z-Score-Kriterium behandeln. Die Merkmale werden wahlweise standardisiert oder robust skaliert. Zur Dimensionsreduktion stehen PCA, UMAP und t-SNE zur Verfügung. Die Auswahl relevanter Merkmale kann über Forward Selection, die Bedeutung im Random Forest, einen Varianzschwellenwert, univariate Tests oder die Lasso-Regression erfolgen. Abschließend wird einer von mehreren Klassifikatoren wie logistische Regression, Support Vector Machines oder Gradient Boosting verwendet.

Beim Training wird diese Pipeline für jede Tierart separat mit einer fünffachen Kreuzvalidierung ausgewertet. Als Kennzahl dient die Balanced Accuracy. Die Laufzeiten der einzelnen Schritte werden erfasst und gemeinsam mit den erzielten Ergebnissen sowie den gewählten Hyperparametern in `results/data/raw_results.csv` gespeichert. Anschließend wird jede Konfiguration auf den vollständigen Trainingsdaten erneut trainiert und per `joblib` unter `results/data/sex_models` gespeichert; das jeweils beste Modell wird zusätzlich in `sex_models_best` kopiert, um Vorhersagen zu erleichtern.

Die Reproduzierbarkeit wird durch Caching von Zwischenergebnissen, das Speichern der Splits im Parquet-Format und einen global festgelegten Zufallssamen unterstützt. Die Messwerte lassen sich mit früheren Arbeiten zur Geschlechtsklassifikation anhand der Gangmerkmale vergleichen, etwa jenen zu PCA oder der Lasso-Regression.

## Pairwise Individual Identification Pipeline (English)
This pipeline evaluates whether two movement trails originate from the same individual. For each animal, non-overlapping trail segments are sampled and paired both within and across individuals. Each pair receives a label indicating whether the trails come from the same individual and is assigned to a fold based on the size of the first trail. Parameters such as chunk size, available trail lengths and the maximum number of trails per animal determine how many pairs are generated.

Multiple distance measures are calculated for every pair, including Euclidean, Manhattan, cosine, Chebyshev, Canberra and Bray–Curtis distances. Optionally the Mahalanobis distance can also be computed if an inverse covariance matrix is provided. These metrics capture different aspects of similarity between the trails.

The core routine processes each pair by applying optional outlier treatment, scaling the features, selecting a subset of features and reducing the dimensionality using either LDA, PCA or UMAP. Once both trails are projected into a common space, distances are calculated between their centroids. For each pair, the mean and median distances across all individual points are summarised.

The computation can be executed in parallel with progress reporting. The resulting records store all parameter settings, the distance metrics, optionally predicted sex probabilities and the coordinates of both trails together with those of the reference set.

## Methodik: Pairwise-Identifikations-Pipeline (Deutsch)
Diese Pipeline prüft, ob zwei Bewegungsbahnen vom selben Tier stammen. Zu diesem Zweck werden pro Tier nicht überlappende Abschnitte ausgewählt und zu Paaren kombiniert, sowohl innerhalb eines Individuums als auch zwischen verschiedenen Individuen. Jedes Paar erhält ein Kennzeichen, ob es sich um dasselbe Tier handelt, sowie eine Fold-Nummer basierend auf der Länge des ersten Trails. Die Parameter zur Abschnittslänge, den möglichen Trailgrößen und der maximalen Trailzahl pro Tier steuern die Menge der erzeugten Paare.

Für jedes Paar werden mehrere Distanzmaße bestimmt: Euklidische Distanz, Manhattan-, Cosinus-, Chebyshev-, Canberra- und Bray–Curtis-Distanz. Wenn eine inverse Kovarianzmatrix vorliegt, kann zusätzlich die Mahalanobis-Distanz berechnet werden. Diese Metriken ermöglichen es, globale und lokale Unterschieden der Trajektorien zu erfassen.

Im Hauptschritt wird für jedes Paar optional eine Ausreißerbehandlung vorgenommen, die Daten werden skaliert, es wird eine Teilmenge relevanter Merkmale ausgewählt und schließlich die Dimension mittels LDA, PCA oder UMAP reduziert. Nach der Projektion in denselben Raum werden die Distanzen zwischen den Zentroiden der beiden Trails berechnet und statistische Kennwerte wie Mittel- und Medianabstand festgehalten.

Die Berechnungen können parallel ausgeführt werden und bieten einen Fortschrittsbalken. Die Ergebnisse enthalten alle Parameter der Durchführung, die ermittelten Distanzen, gegebenenfalls Sex-Wahrscheinlichkeiten sowie die Koordinaten der projizierten Trails mitsamt der Referenzdaten.
