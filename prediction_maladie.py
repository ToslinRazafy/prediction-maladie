import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
from contextlib import redirect_stdout
import unicodedata
from fuzzywuzzy import fuzz, process

class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction de Maladie avec TensorFlow")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f4f8")https://github.com/ToslinRazafy/prediction-maladie

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.history = None
        self.feature_names = ['Age', 'Sexe']
        self.symptom_names = []
        self.classes = []
        self.class_colors = {}
        self.default_colors = ['#28a745', '#dc3545', '#fd7e14', '#17a2b8', '#ffc107']

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=8, font=("Helvetica", 11, "bold"), background="#007bff", foreground="white", borderwidth=1, relief="flat")
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("TLabel", font=("Helvetica", 11), background="#f0f4f8")
        style.configure("TEntry", padding=6, font=("Helvetica", 11), relief="flat", background="#ffffff", borderwidth=1)
        style.configure("TCombobox", padding=6, font=("Helvetica", 11), fieldbackground="#ffffff")
        style.configure("TNotebook", background="#f0f4f8")
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[10, 5], background="#d1e7ff")
        style.map("TNotebook.Tab", background=[('selected', '#007bff'), ('active', '#b3d7ff')])

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Prédiction")

        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Entraînement")

        self.setup_prediction_tab()
        self.setup_training_tab()

    def setup_training_tab(self):
        dataset_frame = ttk.LabelFrame(self.training_frame, text="Chargement des Données", padding=10)
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(dataset_frame, text="Charger Dataset (CSV)", command=self.load_dataset).pack(pady=10)

        params_frame = ttk.LabelFrame(self.training_frame, text="Paramètres d'Entraînement", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Nombre d'époques :").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(params_frame, width=10)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Taille du lot :").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.batch_size_entry = ttk.Entry(params_frame, width=10)
        self.batch_size_entry.insert(0, "32")
        self.batch_size_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(params_frame, text="Entraîner Modèle", command=self.train_model).grid(row=2, column=0, columnspan=2, pady=10)

        logs_frame = ttk.LabelFrame(self.training_frame, text="Logs d'Entraînement", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = tk.Text(logs_frame, height=8, width=80, font=("Courier", 10), bg="#ffffff", relief="flat", borderwidth=1)
        self.log_text.pack(pady=5, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        results_frame = ttk.LabelFrame(self.training_frame, text="Résultats", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.result_label = ttk.Label(results_frame, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=5)

        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=results_frame)
        self.canvas.get_tk_widget().pack(pady=5, fill=tk.BOTH, expand=True)

    def setup_prediction_tab(self):
        input_frame = ttk.LabelFrame(self.prediction_frame, text="Entrée des Données", padding=15)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.input_frame = ttk.Frame(input_frame)
        self.input_frame.pack(pady=10)

        self.input_entries = []
        for i, feature in enumerate(self.feature_names):
            ttk.Label(self.input_frame, text=f"{feature} :").grid(row=i, column=0, padx=10, pady=8, sticky="w")
            if feature == 'Sexe':
                entry = ttk.Combobox(self.input_frame, values=["Homme", "Femme"], width=15, state="readonly")
                entry.set("Homme")
            else:
                entry = ttk.Entry(self.input_frame, width=15)
                entry.insert(0, "30")
            entry.grid(row=i, column=1, padx=10, pady=8)
            self.input_entries.append(entry)

        ttk.Label(self.input_frame, text="Symptômes (séparés par des virgules) :").grid(row=len(self.feature_names), column=0, padx=10, pady=8, sticky="w")
        self.symptoms_entry = ttk.Entry(self.input_frame, width=60)
        self.symptoms_entry.insert(0, "")
        self.symptoms_entry.grid(row=len(self.feature_names), column=1, padx=10, pady=8)

        ttk.Button(self.input_frame, text="Prédire", command=self.predict).grid(row=len(self.feature_names)+1, column=0, columnspan=2, pady=10)

        self.symptom_tooltip_label = ttk.Label(self.input_frame, text="Symptômes valides : Chargez un dataset pour voir les symptômes", font=("Helvetica", 9), wraplength=600, foreground="#555")
        self.symptom_tooltip_label.grid(row=len(self.feature_names)+2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        result_frame = ttk.LabelFrame(self.prediction_frame, text="Résultat de la Prédiction", padding=15)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.prediction_label = ttk.Label(result_frame, text="Entrez les données et cliquez sur Prédire", font=("Helvetica", 14, "bold"), wraplength=900, justify="center")
        self.prediction_label.pack(pady=10)

        self.prob_figure, self.prob_ax = plt.subplots(figsize=(6, 2), dpi=100)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_figure, master=result_frame)
        self.prob_canvas.get_tk_widget().pack(pady=10, fill=tk.X)

    def normalize_text(self, text):
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        return text.lower().strip()

    def map_symptoms(self, input_symptoms):
        mapped_symptoms = []
        for symptom in input_symptoms:
            if symptom.lower() == "aucun symptôme":
                return []
            
            normalized_symptom = self.normalize_text(symptom)
            choices = [self.normalize_text(s) for s in self.symptom_names]
            if not choices:
                self.prediction_label.config(text="Erreur : Aucun symptôme disponible. Chargez un dataset.", foreground="#dc3545")
                return None
            
            match, score = process.extractOne(normalized_symptom, choices, scorer=fuzz.token_sort_ratio)
            if score >= 80:  # Seuil de similarité
                mapped_symptoms.append(self.symptom_names[choices.index(match)])
            else:
                self.prediction_label.config(
                    text=f"Symptôme '{symptom}' non reconnu. Essayez : {', '.join(self.symptom_names[:5])}...",
                    foreground="#dc3545"
                )
                return None
        return mapped_symptoms

    def load_dataset(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                expected_columns = ['Age', 'Sexe', 'Symptômes', 'Maladie']
                if not all(col in df.columns for col in expected_columns):
                    messagebox.showerror("Erreur", "Le dataset doit contenir les colonnes : " + ", ".join(expected_columns))
                    return

                df['Symptômes'] = df['Symptômes'].apply(lambda x: x.split(', ') if isinstance(x, str) and x and x != "Aucun symptôme" else [] if x == "Aucun symptôme" else [x])
                self.symptom_names = sorted(set(symptom for sublist in df['Symptômes'] for symptom in sublist if symptom))
                if not self.symptom_names:
                    messagebox.showerror("Erreur", "Aucun symptôme trouvé dans le dataset.")
                    return

                self.symptom_tooltip_label.config(text=f"Symptômes valides : {', '.join(self.symptom_names[:10])}{'...' if len(self.symptom_names) > 10 else ''}")

                self.mlb.fit([self.symptom_names])
                X_symptoms = self.mlb.transform(df['Symptômes'])
                X_base = df[['Age', 'Sexe']].copy()
                X_base['Sexe'] = X_base['Sexe'].map({'Homme': 0, 'Femme': 1})
                X = np.hstack((X_base.values, X_symptoms))

                self.classes = sorted(df['Maladie'].unique())
                self.label_encoder.fit(self.classes)
                y = self.label_encoder.transform(df['Maladie'])

                self.class_colors = {cls: self.default_colors[i % len(self.default_colors)] for i, cls in enumerate(self.classes)}

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)

                messagebox.showinfo("Succès", f"Dataset chargé avec succès ! {len(self.symptom_names)} symptômes et {len(self.classes)} maladies détectés.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement : {str(e)}")

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger un dataset !")
            return

        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_size_entry.get())

            if epochs <= 0 or batch_size <= 0:
                messagebox.showerror("Erreur", "Les époques et la taille du lot doivent être des nombres positifs !")
                return

            self.model = Sequential([
                Dense(32, activation='relu', input_shape=(self.X_train.shape[1],)),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dense(len(self.classes), activation='softmax')
            ])

            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            log_stream = io.StringIO()
            with redirect_stdout(log_stream):
                self.history = self.model.fit(
                    self.X_train, self.y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.X_test, self.y_test),
                    verbose=1
                )
            logs = log_stream.getvalue()
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, logs)

            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.result_label.config(text=f"Précision sur test : {accuracy*100:.2f}%")

            self.ax.clear()
            self.ax.plot(self.history.history['accuracy'], label='Précision entraînement', color='#007bff')
            self.ax.plot(self.history.history['val_accuracy'], label='Précision validation', color='#28a745')
            self.ax.plot(self.history.history['loss'], label='Perte entraînement', linestyle='--', color='#007bff')
            self.ax.plot(self.history.history['val_loss'], label='Perte validation', linestyle='--', color='#28a745')
            self.ax.set_title('Courbes d\'Apprentissage', fontsize=12, fontweight='bold')
            self.ax.set_xlabel('Époque')
            self.ax.set_ylabel('Valeur')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas.draw()

            messagebox.showinfo("Succès", "Modèle entraîné avec succès !")
        except ValueError as e:
            messagebox.showerror("Erreur", "Veuillez vérifier les paramètres d'entraînement (époques et taille du lot doivent être des nombres entiers).")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement : {str(e)}")

    def predict(self):
        if self.model is None:
            self.prediction_label.config(text="Veuillez entraîner le modèle d'abord !", foreground="#555")
            return

        try:
            inputs = []
            for i, entry in enumerate(self.input_entries):
                value = entry.get().strip()
                if self.feature_names[i] == 'Sexe':
                    if value not in ['Homme', 'Femme']:
                        self.prediction_label.config(text="Erreur : Sélectionnez un sexe valide (Homme ou Femme)", foreground="#dc3545")
                        return
                    value = 1 if value == 'Femme' else 0
                else:
                    try:
                        value = float(value)
                        if value < 0:
                            raise ValueError("L'âge doit être positif.")
                    except ValueError:
                        self.prediction_label.config(text="Erreur : L'âge doit être un nombre valide", foreground="#dc3545")
                        return
                inputs.append(value)

            symptoms = [s.strip() for s in self.symptoms_entry.get().split(',') if s.strip()]
            if not symptoms and self.symptoms_entry.get().strip() != "Aucun symptôme":
                self.prediction_label.config(text="Erreur : Entrez au moins un symptôme ou 'Aucun symptôme'", foreground="#dc3545")
                return

            mapped_symptoms = self.map_symptoms(symptoms)
            if mapped_symptoms is None:
                return

            symptoms_encoded = self.mlb.transform([mapped_symptoms])[0]
            inputs = np.hstack((inputs, symptoms_encoded)).reshape(1, -1)
            inputs = self.scaler.transform(inputs)

            prediction = self.model.predict(inputs, verbose=0)
            predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            probabilities = prediction[0]

            result_text = f"Prédiction : {predicted_class}"
            self.prediction_label.config(text=result_text, foreground=self.class_colors.get(predicted_class, '#000000'))

            self.prob_ax.clear()
            bars = self.prob_ax.bar(self.classes, probabilities, color=[self.class_colors.get(cls, '#000000') for cls in self.classes], alpha=0.8)
            self.prob_ax.set_ylim(0, 1)
            self.prob_ax.set_title('Probabilités des Prédictions', fontsize=12, fontweight='bold')
            self.prob_ax.set_ylabel('Probabilité')
            for bar, prob in zip(bars, probabilities):
                self.prob_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prob:.2f}', 
                                 ha='center', va='bottom', fontsize=10)
            self.prob_ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            self.prob_canvas.draw()

        except Exception as e:
            self.prediction_label.config(text=f"Erreur lors de la prédiction : {str(e)}", foreground="#dc3545")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictionApp(root)
    root.mainloop()