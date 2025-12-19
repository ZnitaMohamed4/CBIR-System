# 📝 Notes - Fine-tuning YOLOv8

## 🎯 Objectif

Expérimentation de fine-tuning YOLOv8n sur 15 classes ImageNet pour comparer dataset déséquilibré vs équilibré.

## 🗂️ Dataset

**15 Classes détectées:**
- Animaux: camel, cow, horse, sheep, bear, elephant, dog, cat, birds
- Véhicules: car, bus, bicycle, airplane, drones
- Personnes: person

### Statistiques du Dataset

**Dataset Initial (Déséquilibré):**
- Train: 5,453 images
- Validation: 1,087 images
- Test: données de validation réutilisées

**Dataset Équilibré:**
- Train: 3,586 images
- Validation: 882 images  
- Test: 415 images
- Cible: 100-600 images par classe

### Équilibrage des Données

Le script `balance_dataset.py` a été utilisé pour:
- Sous-échantillonner les classes sur-représentées (drones, birds, elephant)
- Maintenir les proportions 70/20/10 (train/valid/test)
- Garantir une distribution équilibrée entre les classes

Le script `move_to_test.py` a créé un vrai split de test pour dog et cat.

## 🚀 Entraînement

### Modèle 1 - Dataset Déséquilibré (50 époques)

**Configuration:**
```yaml
Modèle: YOLOv8n
Époques: 50
Batch size: 16
Optimizer: AdamW (auto)
Learning rate: 0.000526 (auto)
Image size: 640x640
```

**Résultats:**
- mAP@50: **0.902** (90.2%)
- mAP@50-95: **0.662** (66.2%)
- Precision: 0.881
- Recall: 0.818

### Modèle 2 - Dataset Équilibré (60 époques) ⭐

**Configuration:**
```yaml
Modèle: YOLOv8n
Époques: 60
Batch size: 16
Optimizer: AdamW
Learning rate: 0.0003 (manuel)
Classification loss weight: 1.5
Augmentations: degrees=10, mixup=0.1, copy_paste=0.05
Image size: 640x640
Patience: 15
```

**Résultats sur données de validation:**
- mAP@50: **0.914** (91.4%)
- mAP@50-95: **0.632** (63.2%)
- Precision: 0.874
- Recall: 0.823

**Résultats sur données de test (jamais vues):**
- mAP@50: **0.897** (89.7%)
- mAP@50-95: **0.627** (62.7%)
- Precision: 0.859
- Recall: 0.830
- **Gap de généralisation: -1.7%** ✅

## 📊 Analyse des Résultats

### Performances par Classe (Modèle 2 - Test)

**Classes Excellentes (>90%):**
- Bear: 98.9% 🐻
- Person: 97.1% 👤
- Birds: 96.4% 🦅
- Camel: 95.5% 🐫
- Cat: 95.4% 🐱
- Drones: 94.3% 🚁
- Horse: 92.8% 🐴

**Classes Bonnes (80-90%):**
- Bus: 88.0%, Dog: 87.8%, Sheep: 87.6%
- Car: 86.5%, Cow: 85.6%, Airplane: 84.8%
- Elephant: 81.1%

**Classe à Améliorer:**
- Bicycle: 73.1% 🚲

### Comparaison des Modèles

| Métrique | Modèle 1 (Déséquilibré) | Modèle 2 (Équilibré) |
|----------|-------------------------|----------------------|
| mAP@50 (val) | 0.902 | 0.914 |
| mAP@50-95 (val) | **0.662** | 0.632 |
| mAP@50 (test) | - | 0.897 |
| Généralisation | - | **Excellente (-1.7%)** |

**Conclusion:** Le Modèle 2 offre une meilleure généralisation et des performances équilibrées entre les classes, bien que le Modèle 1 ait un mAP@50-95 légèrement supérieur.

## 🔍 Problèmes Identifiés

### 1. Faux Positifs sur le Background (67 FP)
- Car: 46 FP (détections dans zones vides)
- Elephant: 30 FP
- Dog: 17 FP

**Solution:** Augmenter le seuil de confiance (0.25-0.35).

### 2. Objets Multiples Non Détectés
- Suppression par NMS lors de chevauchements
- Petits objets difficiles à détecter (bicycles, airplanes)

**Solution:** Ajuster `conf=0.15`, `iou=0.5`, `imgsz=1280`

## 📦 Utilisation

### Télécharger le Modèle

```python
import shutil
from google.colab import files

shutil.copy(
    '/content/runs/detect/imagenet_15classes_balanced_v24/weights/best.pt',
    '/content/yolov8n_15classes_finetuned.pt'
)
files.download('/content/yolov8n_15classes_finetuned.pt')
```

### Inférence

```python
from ultralytics import YOLO

model = YOLO('yolov8n_15classes_finetuned.pt')

# Prédiction sur une image
results = model.predict(
    'image.jpg',
    conf=0.25,
    iou=0.6,
    max_det=300
)
```

## 🎓 Conclusion

**Résultats obtenus:**
- ✅ 89.7% mAP@50 sur test (données jamais vues)
- ✅ Généralisation excellente (-1.7% gap)
- ✅ 7 classes >90%, 13 classes >80%
- ✅ Dataset équilibré améliore la robustesse

**Leçons apprises:**
- L'équilibrage aide la généralisation
- Fine-tuning avec paramètres adaptés améliore les résultats
- NMS et seuils de confiance impactent les détections multiples

---

**Modèle final:** `yolov8n_15classes_finetuned.pt` (6MB)  
**Temps d'entraînement:** 1.4h (60 époques, Tesla T4)
