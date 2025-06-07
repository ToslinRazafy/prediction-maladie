import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from tkinter import Label, Frame, Scrollbar, Text
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import logging
import threading
import os
import pickle
import pandas as pd
from datetime import datetime
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
import spacy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import csv
import seaborn as sns

# Désactiver oneDNN pour éviter les messages inutiles
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Télécharger les ressources nécessaires
nltk.download('stopwords', quiet=True)
french_stop_words = stopwords.words('french')
nlp = spacy.load("fr_core_news_sm", disable=['parser', 'ner'])

# Configurer les logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Classes de maladies
DISEASES = ['grippe', 'rhume', 'gastroentérite', 'migraine']
label_encoder = LabelEncoder()
label_encoder.fit(DISEASES)

# Chemins pour sauvegarder le modèle et le vectoriseur
MODEL_PATH = "disease_model.keras"
VECTORIZER_PATH = "vectorizer.pkl"

# Dictionnaire de synonymes pour l'augmentation
synonyms = {
    'fièvre': ['température', 'chaleur', 'hyperthermie', 'fébrile'],
    'toux': ['quinte', 'toux sèche', 'toux grasse', 'irritation pulmonaire'],
    'fatigue': ['épuisement', 'lassitude', 'faiblesse', 'asthénie'],
    'mal_de_tête': ['céphalée', 'migraine', 'douleur crânienne', 'maux de tête'],
    'diarrhée': ['selles liquides', 'troubles intestinaux', 'évacuations fréquentes'],
    'nausée': ['envie de vomir', 'malaise', 'écœurement', 'vomissement']
}

# Nettoyage et prétraitement du texte
def preprocess_text(text, augment=False):
    try:
        if not isinstance(text, str) or not text.strip():
            logger.warning("Texte vide ou invalide.")
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in french_stop_words and len(token.text) > 2]
        
        if augment and len(tokens) > 2:
            if random.random() < 0.3:
                tokens = [random.choice(['très', 'légèrement', '']) + ' ' + w for w in tokens]
            if random.random() < 0.2:
                random.shuffle(tokens)
        text = ' '.join(tokens).strip()
        return text
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement : {e}")
        return ""

# Augmentation des données
def augment_dataset(texts, labels):
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(3):  # Générer 3 versions augmentées
            aug_text = preprocess_text(text, augment=True)
            if aug_text:
                words = aug_text.split()
                for i, word in enumerate(words):
                    for key, syn_list in synonyms.items():
                        if word in syn_list or word == key:
                            words[i] = random.choice([key] + syn_list)
                            break
                aug_text = ' '.join(words)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Chargement des données CSV
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL)
        if 'text' not in df.columns or 'disease' not in df.columns:
            raise ValueError("Le fichier CSV doit contenir les colonnes 'text' et 'disease'.")
        df['text'] = df['text'].apply(preprocess_text)
        df = df[df['text'].str.strip() != ""]
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])
        
        texts = train_df['text'].tolist()
        labels = train_df['disease'].tolist()
        texts, labels = augment_dataset(texts, labels)
        
        test_texts = test_df['text'].tolist()
        test_labels = test_df['disease'].tolist()
        
        invalid_labels = [label for label in labels + test_labels if label not in DISEASES]
        if invalid_labels:
            raise ValueError(f"Étiquettes invalides : {invalid_labels}")
        
        logger.info(f"Données d'entraînement : {len(texts)} exemples, Répartition : {Counter(labels)}")
        logger.info(f"Données de test : {len(test_texts)} exemples, Répartition : {Counter(test_labels)}")
        return texts, labels, test_texts, test_labels
    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV : {e}")
        return None, None, None, None

# Initialisation du vectoriseur
def initialize_vectorizer(texts):
    try:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words=french_stop_words, ngram_range=(1, 3))
        vectorizer.fit(texts)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info("Vectoriseur TF-IDF sauvegardé.")
        return vectorizer
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du vectoriseur : {e}")
        return None

# Extraction des caractéristiques texte
def extract_text_features(text, vectorizer):
    try:
        text = preprocess_text(text)
        if not text:
            logger.warning("Texte vide après prétraitement.")
            return np.zeros((1, len(vectorizer.vocabulary_)))
        features = vectorizer.transform([text]).toarray()
        return features
    except Exception as e:
        logger.error(f"Erreur dans l'extraction des caractéristiques : {e}")
        return np.zeros((1, len(vectorizer.vocabulary_)))

# Analyse des erreurs
def analyze_errors(model, vectorizer, test_texts, test_labels):
    X_test = vectorizer.transform(test_texts).toarray()
    y_test = label_encoder.transform(test_labels)
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    errors = [(test_texts[i], test_labels[i], label_encoder.inverse_transform([predicted_labels[i]])[0])
              for i in range(len(test_texts)) if predicted_labels[i] != y_test[i]]
    for text, true_label, pred_label in errors:
        logger.info(f"Erreur : Texte='{text}', Vrai={true_label}, Prédit={pred_label}")
    return errors

# Création et entraînement du modèle
def create_and_train_model(text_dim, training_texts, training_labels, test_texts, test_labels, progress_callback=None, log_callback=None, app=None):
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            logger.info("Chargement du modèle existant...")
            return load_model(MODEL_PATH), None
        
        vectorizer = initialize_vectorizer(training_texts)
        if vectorizer is None:
            raise ValueError("Échec de l'initialisation du vectoriseur.")
        
        X = vectorizer.transform(training_texts).toarray()
        y = label_encoder.transform(training_labels)
        X_test = vectorizer.transform(test_texts).toarray()
        y_test = label_encoder.transform(test_labels)
        
        # Modèle amélioré
        text_input = Input(shape=(text_dim,), name='text_input')
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(text_input)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        output = Dense(len(DISEASES), activation='softmax')(x)
        
        model = Model(inputs=text_input, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_accuracies = []
        histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-5)
            checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / 30 * 100
                    if progress_callback:
                        progress_callback(progress)
                    if log_callback:
                        log_callback(f"Fold {fold+1} Époque {epoch + 1}/30 - Perte: {logs['loss']:.4f}, "
                                    f"Précision: {logs['accuracy']:.4f}, Val Perte: {logs['val_loss']:.4f}, "
                                    f"Val Précision: {logs['val_accuracy']:.4f}")
            
            history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val),
                               verbose=0, callbacks=[ProgressCallback(), early_stopping, reduce_lr, checkpoint])
            
            val_accuracies.append(history.history['val_accuracy'][-1])
            histories.append(history.history)
            log_callback(f"Fold {fold+1} - Précision validation : {history.history['val_accuracy'][-1]:.4f}")
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        log_callback(f"Précision test : {test_accuracy:.4f}, Perte test : {test_loss:.4f}")
        
        predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test, predictions)
        log_callback(f"Matrice de confusion :\n{cm}")
        log_callback(classification_report(y_test, predictions, target_names=DISEASES))
        
        if app:
            app.root.after(0, lambda: app.plot_confusion_matrix(cm, DISEASES))
        
        errors = analyze_errors(model, vectorizer, test_texts, test_labels)
        log_callback(f"Nombre d'erreurs sur le test : {len(errors)}")
        
        model.save(MODEL_PATH)
        logger.info("Modèle entraîné et sauvegardé.")
        return model, {'val_accuracy': val_accuracies, 'history': histories, 'test_accuracy': test_accuracy}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {e}")
        return None, None

# Prédiction
def predict_disease(model, text, vectorizer):
    try:
        if model is None or vectorizer is None:
            raise ValueError("Modèle ou vectoriseur non initialisé.")
        
        text_features = extract_text_features(text, vectorizer)
        prediction = model.predict(text_features, verbose=0)
        disease_index = np.argmax(prediction)
        disease = label_encoder.inverse_transform([disease_index])[0]
        probabilities = prediction[0]
        logger.debug(f"Prédiction : {disease}, Probabilités : {probabilities}")
        return disease, probabilities
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return None, None

# Interface Tkinter
class DiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyseur de Maladies IA")
        self.root.geometry("1100x900")
        self.root.configure(bg="#e6effa")
        
        self.model = None
        self.vectorizer = None
        self.training_texts = []
        self.training_labels = []
        self.test_texts = []
        self.test_labels = []
        self.history = None
        
        # Charger le modèle et le vectoriseur si existants
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = load_model(MODEL_PATH)
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        # Style de l'interface
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 12), background="#e6effa")
        
        # Frame principal
        self.main_frame = Frame(root, bg="#e6effa")
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Titre
        self.label = Label(self.main_frame, text="Analyseur de Maladies IA", 
                          font=("Helvetica", 24, "bold"), bg="#e6effa", fg="#1a3c5e")
        self.label.pack(pady=(0, 20))
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(pady=10, fill="both", expand=True)
        
        # Onglet Prédiction
        self.prediction_tab = Frame(self.notebook, bg="#e6effa")
        self.notebook.add(self.prediction_tab, text="Prédiction")
        
        # Onglet Entraînement
        self.training_tab = Frame(self.notebook, bg="#e6effa")
        self.notebook.add(self.training_tab, text="Entraînement")
        
        # Frame pour la prédiction
        self.prediction_frame = Frame(self.prediction_tab, bg="#ffffff", bd=2, relief="flat", padx=10, pady=10)
        self.prediction_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        Label(self.prediction_frame, text="Prédiction des Symptômes", font=("Helvetica", 16, "bold"), bg="#ffffff").pack(anchor="w", pady=5)
        
        self.text_label = Label(self.prediction_frame, text="Décrivez vos symptômes :", font=("Helvetica", 12), bg="#ffffff")
        self.text_label.pack(anchor="w", padx=5, pady=5)
        
        self.text_frame = Frame(self.prediction_frame, bg="#ffffff")
        self.text_frame.pack(fill="both", expand=True)
        self.text_entry = Text(self.text_frame, height=6, width=60, font=("Helvetica", 12), bd=1, relief="solid", wrap="word")
        self.text_entry.pack(side="left", fill="both", expand=True)
        text_scrollbar = Scrollbar(self.text_frame, orient="vertical", command=self.text_entry.yview)
        text_scrollbar.pack(side="right", fill="y")
        self.text_entry.config(yscrollcommand=text_scrollbar.set)
        
        self.text_entry.bind('<KeyRelease>', self.real_time_predict)
        
        self.result_label = Label(self.prediction_frame, text="Maladie : En attente...", 
                                font=("Helvetica", 14, "bold"), bg="#ffffff", fg="#1a3c5e")
        self.result_label.pack(pady=10)
        
        # Graphique des probabilités
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.prediction_frame)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # Bouton pour sauvegarder la prédiction
        self.save_pred_button = ttk.Button(self.prediction_frame, text="Sauvegarder Prédiction", command=self.save_prediction)
        self.save_pred_button.pack(pady=5)
        
        # Frame pour l'entraînement
        self.training_frame = Frame(self.training_tab, bg="#ffffff", bd=2, relief="flat", padx=10, pady=10)
        self.training_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        Label(self.training_frame, text="Entraînement du Modèle", font=("Helvetica", 16, "bold"), bg="#ffffff").pack(anchor="w", pady=5)
        
        self.log_text = Text(self.training_frame, height=10, width=80, font=("Helvetica", 10), bd=1, relief="solid")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        log_scrollbar = Scrollbar(self.training_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # Handler pour les logs
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.see(tk.END)
        
        logger.addHandler(TextHandler(self.log_text))
        
        # Contrôles
        self.control_frame = Frame(self.training_frame, bg="#ffffff")
        self.control_frame.pack(pady=10)
        
        self.clear_button = ttk.Button(self.control_frame, text="Effacer", command=self.clear_text)
        self.clear_button.pack(side="left", padx=5)
        
        self.retrain_button = ttk.Button(self.control_frame, text="Réentraîner", command=self.retrain_model, 
                                        state="normal" if self.model else "disabled")
        self.retrain_button.pack(side="left", padx=5)
        
        self.load_csv_button = ttk.Button(self.control_frame, text="Charger CSV", command=self.load_csv)
        self.load_csv_button.pack(side="left", padx=5)
        
        self.save_model_button = ttk.Button(self.control_frame, text="Sauvegarder Modèle", command=self.save_model, 
                                           state="normal" if self.model else "disabled")
        self.save_model_button.pack(side="left", padx=5)
        
        self.metrics_label = Label(self.training_frame, text="Métriques : Non disponible", 
                                 font=("Helvetica", 12), bg="#ffffff", fg="#1a3c5e")
        self.metrics_label.pack(pady=5)
        
        self.progress_frame = Frame(self.training_frame, bg="#ffffff")
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress = ttk.Progressbar(self.progress_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        
        self.status_label = Label(self.main_frame, text="Modèle prêt" if self.model else "Chargez un fichier CSV", 
                                font=("Helvetica", 10), bg="#c3d7e8", fg="#1a3c5e", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=5)
        
        # Configuration des grilles
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.prediction_frame.columnconfigure(0, weight=1)
        self.prediction_frame.rowconfigure(2, weight=1)
        self.training_frame.columnconfigure(0, weight=1)
        self.training_frame.rowconfigure(1, weight=1)
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)
        
        self.plot_probabilities(np.zeros(len(DISEASES)))
    
    def update_progress(self, value):
        self.progress['value'] = value
        self.status_label.config(text=f"Entraînement en cours... {value:.1f}%")
        self.root.update()
    
    def update_log(self, message):
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
    
    def plot_probabilities(self, probabilities, predicted_disease=None):
        self.ax.clear()
        colors = sns.color_palette("husl", len(DISEASES))
        bars = self.ax.bar(DISEASES, probabilities, color=colors, edgecolor="black")
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Probabilités des Maladies", fontweight="bold", fontsize=12)
        self.ax.set_ylabel("Probabilité", fontsize=10)
        self.ax.set_xlabel("Maladies", fontsize=10)
        plt.xticks(rotation=45, fontsize=9)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            if predicted_disease == DISEASES[i]:
                bar.set_color("#e63946")
            self.ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", 
                         ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        self.canvas.draw()
    
    def plot_confusion_matrix(self, cm, classes):
        self.ax.clear()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=self.ax)
        self.ax.set_title("Matrice de Confusion")
        self.ax.set_ylabel('Étiquette réelle')
        self.ax.set_xlabel('Étiquette prédite')
        plt.tight_layout()
        self.canvas.draw()
    
    def plot_training_history(self, history):
        self.ax.clear()
        if isinstance(history, dict) and 'history' in history:
            history = history['history'][0]
        epochs = range(1, len(history['loss']) + 1)
        self.ax.plot(epochs, history['loss'], 'b-', label='Perte entraînement')
        self.ax.plot(epochs, history['val_loss'], 'r-', label='Perte validation')
        self.ax.plot(epochs, history['accuracy'], 'b--', label='Précision entraînement')
        self.ax.plot(epochs, history['val_accuracy'], 'r--', label='Précision validation')
        self.ax.set_title("Courbes d'Apprentissage")
        self.ax.set_xlabel("Époques")
        self.ax.set_ylabel("Valeur")
        self.ax.legend()
        plt.tight_layout()
        self.canvas.draw()
    
    def initialize_model(self):
        if self.model is not None and self.vectorizer is not None:
            self.status_label.config(text="Modèle pré-entraîné chargé")
            self.update_log("Modèle pré-entraîné chargé")
            self.retrain_button.config(state="normal")
            self.save_model_button.config(state="normal")
            return
        
        if not self.training_texts or not self.training_labels:
            self.status_label.config(text="Aucune donnée d'entraînement")
            self.update_log("Erreur : Aucune donnée d'entraînement")
            messagebox.showerror("Erreur", "Aucune donnée d'entraînement. Chargez un fichier CSV.")
            return
        
        try:
            self.progress['value'] = 0
            self.progress.pack()
            text_dim = len(self.vectorizer.vocabulary_)
            self.model, self.history = create_and_train_model(text_dim, self.training_texts, self.training_labels, 
                                                            self.test_texts, self.test_labels, 
                                                            self.update_progress, self.update_log, self)
            if self.model is None:
                messagebox.showerror("Erreur", "Échec de l'initialisation du modèle.")
                self.status_label.config(text="Erreur d'initialisation")
                self.update_log("Erreur : Échec de l'initialisation")
            else:
                val_accuracy = np.mean(self.history['val_accuracy']) if self.history else 0
                test_accuracy = self.history.get('test_accuracy', 0)
                self.metrics_label.config(
                    text=f"Métriques : Précision val. = {val_accuracy:.2f}, Précision test = {test_accuracy:.2f}")
                self.status_label.config(text="Modèle prêt")
                self.retrain_button.config(state="normal")
                self.save_model_button.config(state="normal")
                self.update_log(f"Modèle initialisé - Précision val. : {val_accuracy:.2f}, Précision test : {test_accuracy:.2f}")
                self.plot_training_history(self.history)
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation : {e}")
            messagebox.showerror("Erreur", f"Échec de l'initialisation : {str(e)}")
            self.status_label.config(text="Erreur d'initialisation")
            self.update_log(f"Erreur : Échec de l'initialisation : {str(e)}")
        finally:
            self.progress.pack_forget()
    
    def real_time_predict(self, event=None):
        if hasattr(self, '_predict_timer'):
            self.root.after_cancel(self._predict_timer)
        self._predict_timer = self.root.after(300, self._perform_prediction)
    
    def _perform_prediction(self):
        if self.model is None or self.vectorizer is None:
            self.status_label.config(text="Modèle ou vectoriseur non prêt")
            self.update_log("Erreur : Modèle ou vectoriseur non prêt")
            messagebox.showerror("Erreur", "Modèle ou vectoriseur non initialisé.")
            return
        threading.Thread(target=self._predict_thread, daemon=True).start()
    
    def _predict_thread(self):
        try:
            text = self.text_entry.get("1.0", tk.END).strip()
            if len(text) < 5:
                self.root.after(0, lambda: self.result_label.config(text="Maladie : En attente..."))
                self.root.after(0, lambda: self.plot_probabilities(np.zeros(len(DISEASES))))
                self.root.after(0, lambda: self.status_label.config(text="Texte trop court..."))
                return
            
            self.root.after(0, lambda: self.status_label.config(text="Analyse en cours..."))
            disease, probabilities = predict_disease(self.model, text, self.vectorizer)
            if disease and probabilities is not None:
                self.root.after(0, lambda: self.result_label.config(text=f"Maladie : {disease}"))
                self.root.after(0, lambda: self.plot_probabilities(probabilities, disease))
                self.root.after(0, lambda: self.status_label.config(text=f"Prédiction : {disease}"))
                self.update_log(f"Prédiction : {disease} (Probabilités : {probabilities})")
            else:
                self.root.after(0, lambda: self.result_label.config(text="Maladie : Erreur"))
                self.root.after(0, lambda: self.status_label.config(text="Erreur lors de la prédiction"))
                self.update_log("Erreur : Échec de la prédiction")
        except Exception as e:
            logger.error(f"Erreur dans la prédiction : {e}")
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur lors de la prédiction : {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Erreur lors de la prédiction"))
            self.update_log(f"Erreur : Prédiction échouée : {str(e)}")
    
    def save_prediction(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text or self.result_label.cget("text") == "Maladie : En attente...":
            messagebox.showwarning("Avertissement", "Aucune prédiction à sauvegarder.")
            return
        disease = self.result_label.cget("text").replace("Maladie : ", "")
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Symptômes : {text}\nPrédiction : {disease}\nDate : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.update_log(f"Prédiction sauvegardée à {file_path}")
    
    def clear_text(self):
        self.text_entry.delete("1.0", tk.END)
        self.result_label.config(text="Maladie : En attente...")
        self.plot_probabilities(np.zeros(len(DISEASES)))
        self.status_label.config(text="Texte effacé")
        self.update_log("Texte effacé")
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.update_log(f"Chargement du fichier CSV : {file_path}")
            texts, labels, test_texts, test_labels = load_csv_data(file_path)
            if texts and labels:
                self.training_texts = texts
                self.training_labels = labels
                self.test_texts = test_texts
                self.test_labels = test_labels
                self.vectorizer = initialize_vectorizer(self.training_texts)
                if self.vectorizer is None:
                    messagebox.showerror("Erreur", "Échec de l'initialisation du vectoriseur.")
                    self.status_label.config(text="Erreur lors du chargement du CSV")
                    self.update_log("Erreur : Échec de l'initialisation du vectoriseur")
                    return
                self.status_label.config(text="Données CSV chargées, entraînement en cours...")
                self.retrain_button.config(state="normal")
                self.save_model_button.config(state="normal")
                self.update_log("Données CSV chargées")
                self.initialize_model()
            else:
                messagebox.showerror("Erreur", "Échec du chargement du CSV.")
                self.update_log("Erreur : Échec du chargement du CSV")
    
    def retrain_model(self):
        self.status_label.config(text="Réentraînement du modèle...")
        self.retrain_button.config(state="disabled")
        self.load_csv_button.config(state="disabled")
        self.save_model_button.config(state="disabled")
        self.progress.pack()
        self.progress['value'] = 0
        self.update_log("Démarrage du réentraînement")
        threading.Thread(target=self._retrain_model_thread, daemon=True).start()
    
    def _retrain_model_thread(self):
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(VECTORIZER_PATH):
                os.remove(VECTORIZER_PATH)
            
            self.vectorizer = initialize_vectorizer(self.training_texts)
            if self.vectorizer is None:
                self.root.after(0, lambda: messagebox.showerror("Erreur", "Échec de l'initialisation du vectoriseur."))
                self.root.after(0, lambda: self.status_label.config(text="Erreur de réentraînement"))
                self.update_log("Erreur : Échec de l'initialisation du vectoriseur")
                return
            
            text_dim = len(self.vectorizer.vocabulary_)
            self.model, self.history = create_and_train_model(text_dim, self.training_texts, self.training_labels, 
                                                            self.test_texts, self.test_labels, 
                                                            self.update_progress, self.update_log, self)
            if self.model is None:
                self.root.after(0, lambda: messagebox.showerror("Erreur", "Échec du réentraînement."))
                self.root.after(0, lambda: self.status_label.config(text="Erreur de réentraînement"))
                self.update_log("Erreur : Échec du réentraînement")
            else:
                val_accuracy = np.mean(self.history['val_accuracy']) if self.history else 0
                test_accuracy = self.history.get('test_accuracy', 0)
                self.root.after(0, lambda: self.metrics_label.config(
                    text=f"Métriques : Précision val. = {val_accuracy:.2f}, Précision test = {test_accuracy:.2f}"))
                self.root.after(0, lambda: self.status_label.config(text="Modèle réentraîné"))
                self.update_log(f"Modèle réentraîné - Précision val. : {val_accuracy:.2f}, Précision test : {test_accuracy:.2f}")
                self.root.after(0, lambda: self.plot_training_history(self.history))
        except Exception as e:
            logger.error(f"Erreur lors du réentraînement : {e}")
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Échec du réentraînement : {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Erreur de réentraînement"))
            self.update_log(f"Erreur : Réentraînement échoué : {str(e)}")
        finally:
            self.root.after(0, lambda: self.retrain_button.config(state="normal"))
            self.root.after(0, lambda: self.load_csv_button.config(state="normal"))
            self.root.after(0, lambda: self.save_model_button.config(state="normal"))
            self.root.after(0, lambda: self.progress.pack_forget())
    
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Erreur", "Aucun modèle à sauvegarder.")
            self.update_log("Erreur : Aucun modèle à sauvegarder")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".keras", filetypes=[("Keras Model", "*.keras")])
        if file_path:
            self.model.save(file_path)
            self.status_label.config(text=f"Modèle sauvegardé à {file_path}")
            self.update_log(f"Modèle sauvegardé à {file_path}")
    
    def destroy(self):
        plt.close('all')
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseApp(root)
    root.protocol("WM_DELETE_WINDOW", app.destroy)
    root.mainloop()