# BANC CED Dynamic — Visualisation et analyse de spectres

Application PyQt5 pour visualiser et analyser des spectres issus du banc CEDd : gestion des jauges de pression, suivi dDAC, lecteur de séquences vidéo, ajustement de pics, baseline, filtres, FFT et outils d'exploration interactive via PyQtGraph.

## Fonctionnalités principales
- **Spectres** : zoom, mesure dY, correction de baseline, FFT, fit de pics multi-modèles (PseudoVoigt, Moffat, Gaussian, PearsonIV, etc.).
- **dDAC** : affichage P(t), dP/dt(t), σ(t), Δλ, sélection par clic dans les graphes et overlay avec le movie associé.
- **Jauges multiples** : Ruby, Sm, SrFCl, Rhodamine6G, Diamond… avec calculs spécifiques.
- **Kernel Python embarqué** : exécution de code utilisateur directement depuis l'interface.
- **Interface interactive** : PyQt5 + PyQtGraph pour le tracé, Matplotlib pour les overlays lorsque nécessaire.

## Installation
1. Cloner le dépôt.
2. Créer et activer un environnement virtuel (conda ou venv) sous Python 3.10+.
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Lancement
```bash
python main.py
```
(ou `python BANC_CED_Dynamic_vLABO.py` selon votre point d'entrée).

## Organisation du code
- `Bibli_python/CL_FD_Update.py` : cœur des traitements CEDd, gestion des fits, filtres et chargement de données.
- `Bibli_python/Oscilloscope_LeCroy_vLABO.py` : interface oscilloscope/trace.
- `Bibli_python/Data/` : conteneurs de données (spectres, jauges, pics).
- `Bibli_python/Use_Data/` : opérations haut niveau sur les spectres et jauges.
- `BANC_CED_Dynamic_vLABO.py` : interface graphique principale PyQt5/PyQtGraph.

## Dépendances externes spéciales
- **OpenCV (`opencv-python`)** : nécessaire pour la lecture/affichage des fichiers vidéo (optionnel si vous ne traitez pas de séquences).
- **lecroyscope** : utilisé pour communiquer avec l'oscilloscope LeCroy (optionnel si vous chargez uniquement des fichiers).

## Compatibilité
- Systèmes testés : Windows et Linux.
- Version Python recommandée : 3.10 ou supérieure.

## Auteurs et licence
- Auteur : Equipe CEDd.
- Licence : MIT (voir `LICENSE`).
