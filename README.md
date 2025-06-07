# Analyseur de Maladies IA

## Aperçu
L'**Analyseur de Maladies IA** est une application développée en Python qui utilise l'intelligence artificielle pour prédire des maladies courantes (grippe, rhume, gastroentérite, migraine) à partir de descriptions textuelles des symptômes. L'application s'appuie sur un modèle de réseau de neurones construit avec TensorFlow, combiné à un prétraitement de texte utilisant TF-IDF et SpaCy pour l'analyse en langue française. Une interface graphique conviviale, développée avec Tkinter, permet aux utilisateurs de saisir des symptômes, visualiser les prédictions en temps réel et entraîner ou réentraîner le modèle avec des données personnalisées.

## Fonctionnalités
- **Prédiction en temps réel** : Saisissez une description des symptômes pour obtenir une prédiction de la maladie avec des probabilités associées, affichées sous forme de graphique à barres.
- **Entraînement personnalisé** : Chargez des fichiers CSV contenant des données d'entraînement (symptômes et étiquettes de maladies) pour entraîner ou réentraîner le modèle.
- **Visualisation des performances** : Affichez la matrice de confusion et les courbes d'apprentissage pour évaluer les performances du modèle.
- **Sauvegarde des résultats** : Enregistrez les prédictions et les modèles entraînés pour une utilisation ultérieure.
- **Prétraitement avancé** : Inclut la lemmatisation, la suppression des mots vides, et l'augmentation des données via des synonymes et des variations textuelles.

## Prérequis
- **Python** : Version 3.8 ou supérieure
- **Bibliothèques Python** :
  - `tensorflow>=2.10`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `spacy`
  - `nltk`
  - `matplotlib`
  - `seaborn`
- **Modèle SpaCy** : `fr_core_news_sm` (installé via `python -m spacy download fr_core_news_sm`)
- **NLTK** : Téléchargez les mots vides en français (`nltk.download('stopwords')`)

## Installation
1. Clonez le dépôt ou téléchargez le code source.
2. Installez les dépendances :
   ```bash
   pip install tensorflow scikit-learn pandas numpy spacy nltk matplotlib seaborn
   python -m spacy download fr_core_news_sm
   ```
3. Exécutez l'application :
   ```bash
   python prediction_maladie.py
   ```

## Utilisation
1. **Lancement de l'application** :
   - Exécutez le script principal (`prediction_maladie.py`) pour lancer l'interface graphique.
2. **Prédiction des symptômes** :
   - Dans l'onglet "Prédiction", saisissez une description des symptômes dans la zone de texte.
   - La prédiction est mise à jour en temps réel avec un graphique des probabilités pour chaque maladie.
   - Cliquez sur "Sauvegarder Prédiction" pour enregistrer la prédiction dans un fichier texte.
3. **Entraînement du modèle** :
   - Dans l'onglet "Entraînement", chargez un fichier CSV avec les colonnes `text` (symptômes) et `disease` (maladie).
   - Cliquez sur "Charger CSV" pour importer les données, puis sur "Réentraîner" pour entraîner le modèle.
   - Visualisez les métriques (précision, matrice de confusion) et les courbes d'apprentissage.
4. **Sauvegarde du modèle** :
   - Cliquez sur "Sauvegarder Modèle" pour enregistrer le modèle entraîné au format `.keras`.

## Exemple de données
### Fichier d'entraînement (train.csv)
```csv
text,disease
"J'ai de la fièvre, une toux sèche et je me sens très fatigué","grippe"
"Je tousse beaucoup et j'ai le nez qui coule depuis plusieurs jours","rhume"
"J'ai des douleurs abdominales, de la diarrhée et des nausées","gastroentérite"
"Je ressens une forte douleur à la tête et une sensibilité à la lumière","migraine"
"Fièvre élevée, frissons et douleurs musculaires intenses","grippe"
```

### Fichier de test (test.csv)
```csv
text,disease
"Je tousse beaucoup et j'ai une fièvre légère","grippe"
"Écoulement nasal clair et éternuements fréquents","rhume"
"Diarrhée, vomissements et douleurs abdominales","gastroentérite"
```

## Performances actuelles
Les performances du modèle, mesurées le **7 juin 2025 à 20:55:07**, sont les suivantes :
- **Précision sur le test** : 1.0000
- **Perte sur le test** : 0.1329
- **Matrice de confusion** :
  ```
  [[3 0 0 0]
   [0 3 0 0]
   [0 0 3 0]
   [0 0 0 4]]
  ```
- **Rapport de classification** :
  ```
                  precision    recall  f1-score   support
  grippe          1.00      1.00      1.00         3
  rhume           1.00      1.00      1.00         3
  gastroentérite  1.00      1.00      1.00         3
  migraine        1.00      1.00      1.00         4
  accuracy                           1.00        13
  macro avg       1.00      1.00      1.00        13
  weighted avg    1.00      1.00      1.00        13
  ```
- **Nombre d'erreurs sur le test** : 0
- **Précision moyenne sur la validation** : 0.89
- **Précision sur le test** : 1.00

## Structure du projet
- `prediction_maladie.py` : Script principal contenant le code de l'application.
- `disease_model.keras` : Fichier du modèle entraîné (généré après l'entraînement).
- `vectorizer.pkl` : Fichier du vectoriseur TF-IDF sauvegardé.
- Dossiers de données (non inclus) : Contiennent les fichiers CSV pour l'entraînement et le test.

## Améliorations futures
- Ajout de nouvelles maladies à la liste des prédictions.
- Intégration de davantage de synonymes pour améliorer l'augmentation des données.
- Support pour d'autres langues.
- Optimisation des hyperparamètres du modèle pour une meilleure généralisation.

## Avertissement
Ce projet est destiné à des fins éducatives et de démonstration. Les prédictions ne doivent pas être utilisées pour un diagnostic médical réel. Consultez un professionnel de santé pour tout problème médical.

## Contact
Pour toute question ou contribution, veuillez ouvrir une issue sur le dépôt GitHub ou contacter l'auteur.