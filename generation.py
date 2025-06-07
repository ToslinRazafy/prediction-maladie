import random
import csv

# Listes de symptômes par maladie, ajustées pour plus de réalisme
symptomes_rhume = ["Éternuements", "Nez qui coule", "Mal de gorge", "Toux légère", "Fièvre légère"]
symptomes_grippe = ["Fièvre élevée", "Frissons", "Courbatures", "Fatigue intense", "Toux sèche", "Maux de tête", "Nez qui coule", "Mal de gorge"]
symptomes_sain = ["Aucun symptôme", "Fatigue légère"]  # Corrigé pour correspondre aux probabilités

# Probabilités pour sélectionner les symptômes (inchangées)
probabilites_rhume = {
    "Éternuements": 0.9, "Nez qui coule": 0.9, "Mal de gorge": 0.7,
    "Toux légère": 0.6, "Fièvre légère": 0.3
}
probabilites_grippe = {
    "Fièvre élevée": 0.9, "Frissons": 0.8, "Courbatures": 0.8, "Fatigue intense": 0.7,
    "Toux sèche": 0.6, "Maux de tête": 0.6, "Nez qui coule": 0.3, "Mal de gorge": 0.3
}
probabilites_sain = {
    "Aucun symptôme": 0.9, "Fatigue légère": 0.1
}

def generer_symptomes(maladie):
    if maladie == "Sain":
        return random.choices(
            symptomes_sain,
            weights=[probabilites_sain[s] for s in symptomes_sain],
            k=1
        )[0]
    elif maladie == "Rhume":
        symptomes_choisis = [
            s for s in symptomes_rhume if random.random() < probabilites_rhume[s]
        ]
        if not symptomes_choisis:  
            symptomes_choisis = random.sample(symptomes_rhume, k=1)
        return ", ".join(random.sample(symptomes_choisis, k=min(len(symptomes_choisis), random.randint(2, 4))))
    else:  # Grippe
        # Sélectionner 3 à 5 symptômes, en respectant les probabilités
        symptomes_choisis = [
            s for s in symptomes_grippe if random.random() < probabilites_grippe[s]
        ]
        if not symptomes_choisis:  
            symptomes_choisis = random.sample(symptomes_grippe, k=1)
        return ", ".join(random.sample(symptomes_choisis, k=min(len(symptomes_choisis), random.randint(3, 5))))

data = []
classes = ["Sain", "Grippe", "Rhume"]
n_samples = 1000
samples_per_class = n_samples // len(classes)

for maladie in classes:
    for _ in range(samples_per_class):
        age = random.randint(18, 80)
        sexe = random.choice([0, 1])  # 0 pour Femme, 1 pour Homme
        symptomes = generer_symptomes(maladie)
        data.append([age, sexe, symptomes, maladie])

while len(data) < n_samples:
    maladie = random.choice(classes)
    age = random.randint(18, 80)
    sexe = random.choice([0, 1])
    symptomes = generer_symptomes(maladie)
    data.append([age, sexe, symptomes, maladie])

with open("dataset_maladies.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Age", "Sexe", "Symptômes", "Maladie"])
    writer.writerows(data)

print("Dataset généré : dataset_maladies.csv")