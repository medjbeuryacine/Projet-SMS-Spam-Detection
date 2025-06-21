import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import torch
import pickle
import os

print("🚀 Démarrage du projet SMS Spam Detection - SpamShield")
print("=" * 60)

# Configuration
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
print("\n📊 Chargement du dataset SMS Spam...")
try:
    dataset = load_dataset("sms_spam")
    df = pd.DataFrame(dataset['train'])
except:
    # Si le dataset n'est pas disponible, on crée un dataset de démonstration
    print("⚠️ Dataset SMS Spam non trouvé, création d'un dataset de démonstration...")
    sample_data = {
        'sms': [
            "WINNER!! You have won £1000 cash! Call 09061701461",
            "Hi how are you today?",
            "FREE entry in 2 a weekly comp for a chance to win FA Cup final tkts",
            "Thanks for your message. See you later",
            "URGENT! You have won a 1 week FREE membership",
            "Hey what's up?",
            "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed",
            "Can we meet tomorrow?",
            "You have been selected to receive £350 CASH or a FREE gift",
            "Good morning! How was your night?"
        ] * 500,  # Répéter pour avoir plus de données
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 500  # 1=spam, 0=ham
    }
    df = pd.DataFrame(sample_data)
    
    # Ajouter plus de variété
    ham_messages = [
        "Thanks for the message", "See you tomorrow", "How are you?", 
        "Meeting at 3pm", "Love you", "Good morning", "Have a nice day",
        "Call me later", "Where are you?", "I'm on my way"
    ]
    spam_messages = [
        "WIN £1000 NOW!", "FREE GIFT claim now", "URGENT call this number",
        "You've won a prize", "Click here to win", "FREE entry to win",
        "Congratulations winner", "Cash prize waiting", "Limited time offer"
    ]
    
    # Équilibrer le dataset
    additional_ham = pd.DataFrame({
        'sms': ham_messages * 100,
        'label': [0] * (len(ham_messages) * 100)
    })
    additional_spam = pd.DataFrame({
        'sms': spam_messages * 100,
        'label': [1] * (len(spam_messages) * 100)
    })
    
    df = pd.concat([df, additional_ham, additional_spam], ignore_index=True)

# Convertir les labels en format standard
if 'label' in df.columns:
    if df['label'].dtype == 'object':  # Si les labels sont 'ham'/'spam'
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"📈 Distribution originale des données:")
print(df['label'].value_counts())
print(f"Ham (0): {sum(df['label'] == 0)}")
print(f"Spam (1): {sum(df['label'] == 1)}")

# 2. ÉQUILIBRAGE DES DONNÉES POUR LE FINE-TUNING
print("\n⚖️ Équilibrage des données...")
df_ham = df[df['label'] == 0]
df_spam = df[df['label'] == 1]

# Méthode 1: Sous-échantillonnage de la classe majoritaire
min_samples = min(len(df_ham), len(df_spam))
df_ham_balanced = df_ham.sample(n=min_samples, random_state=42)
df_spam_balanced = df_spam.sample(n=min_samples, random_state=42)

# Méthode 2: Sur-échantillonnage de la classe minoritaire (alternative)
max_samples = max(len(df_ham), len(df_spam))
if len(df_spam) < len(df_ham):
    df_spam_upsampled = resample(df_spam, replace=True, n_samples=max_samples, random_state=42)
    df_balanced_upsampled = pd.concat([df_ham, df_spam_upsampled])
else:
    df_ham_upsampled = resample(df_ham, replace=True, n_samples=max_samples, random_state=42)
    df_balanced_upsampled = pd.concat([df_spam, df_ham_upsampled])

# Utiliser la méthode de sous-échantillonnage pour éviter l'overfitting
df_balanced = pd.concat([df_ham_balanced, df_spam_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"📊 Distribution après équilibrage:")
print(df_balanced['label'].value_counts())

# Division train/test
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])

print(f"📈 Tailles des ensembles:")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# 3. MODÈLE ZERO-SHOT
print("\n🎯 Modèle 1: Zero-Shot Classification")
print("-" * 40)

# Utilisation d'un modèle pré-entraîné pour la classification zero-shot
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

def zero_shot_predict(texts, candidate_labels=["legitimate message", "spam message"]):
    """Prédiction zero-shot"""
    results = []
    for text in texts:
        result = zero_shot_classifier(text, candidate_labels)
        # Convertir en label binaire: spam=1, ham=0
        prediction = 1 if result['labels'][0] == "spam message" else 0
        results.append(prediction)
    return results

# Test sur un petit échantillon
sample_texts = test_df['sms'].head(10).tolist()
zero_shot_preds = zero_shot_predict(sample_texts)
print("✅ Modèle Zero-Shot configuré")

# 4. MODÈLE FEW-SHOT
print("\n🎯 Modèle 2: Few-Shot Learning")
print("-" * 40)

# Préparation des exemples pour few-shot
few_shot_examples = """
Exemples de classification SMS:

Message: "Hi, how are you doing today?" → Classe: ham
Message: "FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005" → Classe: spam
Message: "Thanks for your message. Talk to you later" → Classe: ham
Message: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!" → Classe: spam
Message: "Can we meet for lunch tomorrow?" → Classe: ham

"""

# Utilisation d'un modèle de génération pour few-shot
few_shot_model = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    device=0 if torch.cuda.is_available() else -1
)

def few_shot_predict(texts):
    """Prédiction few-shot avec des exemples"""
    results = []
    for text in texts:
        prompt = f"{few_shot_examples}\nMessage: \"{text}\" → Classe:"
        
        # Approche simplifiée: utiliser des mots-clés
        spam_keywords = ['free', 'win', 'winner', 'prize', 'cash', 'urgent', 'click', 'call now', 
                        'limited time', 'congratulations', 'selected', 'reward', '£', '$']
        
        text_lower = text.lower()
        spam_score = sum(1 for keyword in spam_keywords if keyword in text_lower)
        
        # Prédiction basée sur le score
        prediction = 1 if spam_score >= 2 else 0
        results.append(prediction)
    
    return results

few_shot_preds = few_shot_predict(sample_texts)
print("✅ Modèle Few-Shot configuré")

# 5. MODÈLE FINE-TUNED
print("\n🎯 Modèle 3: Fine-Tuning")
print("-" * 40)

# Configuration du modèle pour fine-tuning
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1}
)

# Tokenisation des données
def tokenize_function(examples):
    return tokenizer(examples['sms'], truncation=True, padding=True, max_length=128)

# Création des datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/fine_tuned_model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir=f"{MODEL_DIR}/logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,
)

# Métrique d'évaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🏋️ Début du fine-tuning...")
trainer.train()

# Sauvegarde du modèle fine-tuné
model.save_pretrained(f"{MODEL_DIR}/fine_tuned_model")
tokenizer.save_pretrained(f"{MODEL_DIR}/fine_tuned_model")
print("💾 Modèle fine-tuné sauvegardé")

# 6. ÉVALUATION DES MODÈLES
print("\n📊 Évaluation des modèles")
print("=" * 60)

# Prédictions sur l'ensemble de test
test_texts = test_df['sms'].tolist()
test_labels = test_df['label'].tolist()

print("🔄 Évaluation Zero-Shot...")
zero_shot_predictions = zero_shot_predict(test_texts[:50])  # Limité pour la démo

print("🔄 Évaluation Few-Shot...")
few_shot_predictions = few_shot_predict(test_texts)

print("🔄 Évaluation Fine-Tuned...")
fine_tuned_classifier = pipeline(
    "text-classification",
    model=f"{MODEL_DIR}/fine_tuned_model",
    tokenizer=f"{MODEL_DIR}/fine_tuned_model",
    device=0 if torch.cuda.is_available() else -1
)

fine_tuned_predictions = []
for text in test_texts:
    result = fine_tuned_classifier(text)
    prediction = 1 if result[0]['label'] == 'LABEL_1' else 0
    fine_tuned_predictions.append(prediction)

# Calcul des métriques
print("\n📈 RÉSULTATS:")
print("-" * 40)

# Few-Shot (sur tous les tests)
few_shot_accuracy = accuracy_score(test_labels, few_shot_predictions)
print(f"Few-Shot Accuracy: {few_shot_accuracy:.4f}")
print("Few-Shot Classification Report:")
print(classification_report(test_labels, few_shot_predictions, target_names=['Ham', 'Spam']))

# Fine-Tuned
fine_tuned_accuracy = accuracy_score(test_labels, fine_tuned_predictions)
print(f"\nFine-Tuned Accuracy: {fine_tuned_accuracy:.4f}")
print("Fine-Tuned Classification Report:")
print(classification_report(test_labels, fine_tuned_predictions, target_names=['Ham', 'Spam']))

# Zero-Shot (échantillon limité)
if len(zero_shot_predictions) > 0:
    zero_shot_accuracy = accuracy_score(test_labels[:len(zero_shot_predictions)], zero_shot_predictions)
    print(f"\nZero-Shot Accuracy (échantillon): {zero_shot_accuracy:.4f}")

# 7. SAUVEGARDE DES MÉTADONNÉES
print("\n💾 Sauvegarde des métadonnées...")

model_info = {
    'zero_shot_model': 'facebook/bart-large-mnli',
    'few_shot_approach': 'keyword-based with examples',
    'fine_tuned_model': f"{MODEL_DIR}/fine_tuned_model",
    'accuracies': {
        'few_shot': few_shot_accuracy,
        'fine_tuned': fine_tuned_accuracy,
        'zero_shot': zero_shot_accuracy if len(zero_shot_predictions) > 0 else 'N/A'
    },
    'dataset_info': {
        'total_samples': len(df_balanced),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'balanced': True
    }
}

with open(f"{MODEL_DIR}/model_info.pkl", 'wb') as f:
    pickle.dump(model_info, f)

print("✅ Entraînement terminé!")
print(f"📁 Modèles sauvegardés dans: {MODEL_DIR}/")
print("\n🎉 Projet SpamShield - Modèles prêts pour le déploiement!")
print("=" * 60)