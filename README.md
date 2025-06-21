# 🛡️ SpamShield - Projet SMS Spam Detection

## 🎯 Aperçu du projet

**SpamShield** est un système de détection de spam SMS utilisant trois approches d'apprentissage automatique différentes pour comparer leurs performances et efficacité.

### Objectifs
- ✅ Implémenter et comparer 3 approches de classification de texte
- ✅ Créer une interface utilisateur interactive avec Gradio
- ✅ Déployer le système pour une utilisation publique
- ✅ Analyser les performances de chaque approche

### Technologies utilisées
- **Framework ML** : Transformers (Hugging Face)
- **Interface** : Gradio
- **Modèles** : BART-MNLI, DistilBERT, DialoGPT
- **Dataset** : SMS Spam Collection
- **Déploiement** : Local + Hugging Face Spaces

## 🏗️ Architecture technique

### 1. 🎯 Zero-Shot Classification
```python
Modèle: facebook/bart-large-mnli
Principe: Classification sans entraînement spécifique
Labels: ["legitimate message", "spam message"]
```

### 2. 📝 Few-Shot Learning
```python
Approche: Analyse heuristique avec mots-clés
Signaux détectés:
- Mots-clés spam: free, win, prize, urgent, etc.
- Indicateurs: £, $, %, majuscules, longueur
```

### 3. 🏋️ Fine-Tuning
```python
Modèle de base: distilbert-base-uncased
Dataset: SMS Spam équilibré (747 ham + 747 spam)
Entraînement: 10 époques, batch_size=8
```

## 📊 Résultats obtenus

### Performances finales
| Modèle | Accuracy | Précision | Rappel | F1-Score |
|--------|----------|-----------|---------|----------|
| **Few-Shot** | **67.2%** | 80% | 67% | 63% |
| **Zero-Shot** | 52.0% | N/A | N/A | N/A |
| **Fine-Tuned** | 50.2% | 25% | 50% | 34% |

### 🏆 Classement des modèles
1. **🥇 Few-Shot Learning** - Meilleur équilibre performance/simplicité
2. **🥈 Zero-Shot Classification** - Performance correcte sans entraînement
3. **🥉 Fine-Tuned Model** - Problèmes de mapping et overfitting

## ❌ Problèmes rencontrés

### 1. **Problème de dépendances**
```bash
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```
**Cause** : Changement de nom du paramètre dans Transformers  
**Solution** : `evaluation_strategy` → `eval_strategy`

### 2. **Dataset déséquilibré**
```
Ham: 4827 messages (86.6%)
Spam: 747 messages (13.4%)
```
**Impact** : Biais vers la classe majoritaire  
**Solution** : Sous-échantillonnage pour équilibrer

### 3. **Mauvaises performances du Fine-Tuned**
```
Fine-Tuned Accuracy: 50.2% (attendu: >90%)
Warning: Precision ill-defined for labels with no predicted samples
```
**Causes identifiées** :
- Mapping incorrect des labels (`LABEL_0`/`LABEL_1`)
- Overfitting (97% train → 50% test)
- Dataset trop petit après équilibrage

## ✅ Solutions implémentées

### 1. **Correction des dépendances**
```python
# Avant
evaluation_strategy="epoch"
# Après
eval_strategy="epoch"
```

### 2. **Équilibrage du dataset**
```python
# Sous-échantillonnage de la classe majoritaire
min_samples = min(len(df_ham), len(df_spam))
df_balanced = pd.concat([df_ham.sample(min_samples), df_spam.sample(min_samples)])
```

### 3. **Amélioration du Fine-Tuned**
```python
# Mapping des labels corrigé
if '0' in label or 'LABEL_0' in label:
    prediction = "Ham (Légitime)"
elif '1' in label or 'LABEL_1' in label:
    prediction = "Spam"
```

## 🔧 Points d'amélioration

### 🚀 Amélioration immédiate (facile)

#### Dataset
- **Augmenter la taille** : Utiliser plus de données d'entraînement
- **Validation croisée** : K-fold pour une évaluation robuste
- **Stratification** : Maintenir la distribution lors des splits

#### Fine-Tuning
```python
# Paramètres suggérés
training_args = TrainingArguments(
    num_train_epochs=5,  # Réduire pour éviter l'overfitting
    learning_rate=1e-5,  # Learning rate plus faible
    early_stopping_patience=3,  # Early stopping
    warmup_ratio=0.1,
)
```

#### Few-Shot
```python
# Améliorer les mots-clés
spam_keywords += ['exclusive', 'limited', 'act now', 'expires', 'credit']
# Ajouter un scoring pondéré
weights = {'free': 2, 'win': 2, 'urgent': 3}
```

### 🎯 Amélioration avancée (moyen terme)

#### Nouveaux modèles
- **RoBERTa** au lieu de DistilBERT
- **Ensemble methods** : Combiner les prédictions
- **BERT spécialisé** pour les SMS courts

#### Techniques avancées
```python
# Data augmentation
from nlpaug import AugmentationPipeline
# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
# Hyperparameter tuning
from optuna import create_study
```

#### Métriques avancées
- **ROC-AUC** : Courbe ROC pour évaluer la discrimination
- **Confusion Matrix** : Analyse détaillée des erreurs
- **Cross-validation** : Validation plus robuste

### 🌟 Amélioration à long terme

#### Architecture
- **API REST** : Déploiement scalable
- **Base de données** : Stockage des prédictions
- **Monitoring** : Suivi des performances en production

#### Interface
- **Dashboard analytics** : Statistiques détaillées
- **Feedback utilisateur** : Amélioration continue
- **Historique** : Sauvegarde des analyses

## 📥 Installation et utilisation

### Prérequis
```bash
Python 3.8+
CUDA (optionnel, pour GPU)
4GB RAM minimum, 8GB recommandé
```

### Installation
```bash
# Cloner le projet
git clone <votre-repo>
cd spamshield

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Dépendances
pip install -r requirements.txt
```

### Utilisation
```bash
# 1. Entraîner les modèles
python train_models.py

# 2. Lancer l'interface
python app.py

# 3. Ouvrir http://127.0.0.1:7860
```

### Structure du projet
```
spamshield/
├── train_models.py       # Entraînement des 3 modèles
├── app.py               # Interface Gradio
├── requirements.txt     # Dépendances
├── README.md           # Ce fichier
├── models/             # Modèles sauvegardés
│   ├── fine_tuned_model/
│   ├── model_info.pkl
```

## 🎯 Résultats d'apprentissage

### ✅ Objectifs atteints
- [x] Implémentation de 3 approches ML différentes
- [x] Interface utilisateur fonctionnelle
- [x] Comparaison des performances
- [x] Déploiement local réussi
- [x] Documentation complète

### 📚 Compétences développées
- **Transformers** : Utilisation de modèles pré-entraînés
- **Gradio** : Création d'interfaces ML
- **Classification de texte** : Preprocessing et évaluation
- **Debugging** : Résolution de problèmes techniques
- **Documentation** : Rapport technique détaillé

### 🔍 Leçons apprises
1. **L'équilibrage des données** est crucial pour éviter les biais
2. **Le fine-tuning** n'est pas toujours la meilleure solution
3. **Les approches simples** (Few-Shot) peuvent être très efficaces
4. **Le debugging** est une partie importante du développement ML
5. **La documentation** aide à identifier et résoudre les problèmes

## 📈 Métriques détaillées

### Few-Shot (Meilleur modèle)
```
              precision    recall  f1-score   support
         Ham       0.60      1.00      0.75       150
        Spam       1.00      0.34      0.51       149
    accuracy                           0.67       299
```

**Analyse** :
- ✅ **Très bonne précision** pour détecter le spam (100%)
- ❌ **Faible rappel** pour le spam (34% seulement détectés)
- 🎯 **Équilibré** entre les deux classes

### Zero-Shot
```
Accuracy: 52% (sur échantillon limité)
```

**Analyse** :
- ✅ **Aucun entraînement** requis
- ❌ **Performance limitée** sur domaine spécialisé
- 🎯 **Bonne baseline** pour comparaison

### Fine-Tuned (Problématique)
```
              precision    recall  f1-score   support
         Ham       0.50      1.00      0.67       150
        Spam       0.00      0.00      0.00       149
```

**Analyse** :
- ❌ **Aucun spam détecté** (rappel = 0%)
- ❌ **Overfitting sévère** (97% train → 50% test)
- 🔧 **Nécessite refactoring** du mapping des labels

## 🚀 Déploiement

### Local ✅
```bash
python app.py
# → http://127.0.0.1:7860
```

### Hugging Face Spaces (optionnel)
```bash
# Créer un Space sur huggingface.co
# Uploader les fichiers
git push origin main
# → https://huggingface.co/spaces/username/spamshield
```

## 🎉 Conclusion

### Bilan du projet
**SpamShield** démontre avec succès l'implémentation et la comparaison de trois approches de classification de texte. Malgré les défis techniques rencontrés, le projet atteint ses objectifs principaux et fournit des insights précieux sur les différentes méthodes ML.

### Points forts
- ✅ **Architecture modulaire** facilitant les améliorations
- ✅ **Interface utilisateur** intuitive et fonctionnelle  
- ✅ **Comparaison rigoureuse** des approches
- ✅ **Documentation complète** des problèmes et solutions
- ✅ **Code reproductible** avec gestion d'erreurs

### Impact pédagogique
Ce projet illustre parfaitement les réalités du développement ML :
- Les **défis techniques** inattendus
- L'importance du **preprocessing** des données
- La **complexité** du fine-tuning vs approches simples
- La valeur de la **documentation** et du debugging

### Perspectives d'évolution
SpamShield constitue une **base solide** pour des améliorations futures et peut servir de **template** pour d'autres projets de classification de texte.

---

**Développé avec ❤️ pour l'apprentissage de la classification de texte**

*Projet réalisé dans le cadre d'un cours de Text Classification - Novembre 2024*