# CleanChem - Nettoyeur de Chimiothèques

CleanChem est un outil de standardisation et de nettoyage pour vos collections de molécules, conçu spécifiquement pour gérer des bibliothèques chimiques volumineuses (plusieurs millions de molécules et des fichiers de plus de 1 Go) tout en optimisant l'utilisation de la mémoire.

## Fonctionnalités principales

- Vérification et correction des SMILES
- Suppression des sels et solvants
- Traitement des fragments multiples
- Élimination des molécules avec atomes exotiques
- Détection et correction des isotopes
- Détection des doublons optimisée pour les larges collections
- Interface utilisateur intuitive avec visualisation moléculaire
- Traitement par lots pour une gestion optimale de la mémoire
- Support pour des fichiers de plus de 1 Go

## Installation avec Conda

Conda est l'environnement recommandé pour l'installation de CleanChem, car il gère efficacement les dépendances scientifiques comme RDKit.

1. **Installer Miniconda ou Anaconda** (si ce n'est pas déjà fait)
   - Téléchargez et installez depuis [le site officiel de Conda](https://docs.conda.io/en/latest/miniconda.html)

2. **Cloner le dépôt** (ou télécharger le fichier `cleanchem.py`)
   ```bash
   git clone https://github.com/votre-nom/cleanchem.git
   cd cleanchem
   ```

3. **Créer un environnement Conda**
   ```bash
   conda create -n cleanchem python=3.10
   ```

4. **Activer l'environnement**
   ```bash
   conda activate cleanchem
   ```

5. **Installer les dépendances**
   ```bash
   conda install -c conda-forge rdkit
   conda install pandas numpy
   pip install streamlit psutil
   ```

   Ou via un fichier d'environnement (environment.yml):
   ```bash
   conda env create -f environment.yml
   ```

   Contenu de environment.yml:
   ```yaml
   name: cleanchem
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python=3.10
     - rdkit>=2023.3.1
     - pandas>=1.5.0
     - numpy>=1.22.0
     - pip
     - pip:
       - streamlit>=1.22.0
       - psutil>=5.9.0
   ```

## Utilisation

1. **Activer l'environnement** (si ce n'est pas déjà fait)
   ```bash
   conda activate cleanchem
   ```

2. **Lancer l'application**
   ```bash
   streamlit run cleanchem.py --server.maxUploadSize=2024

   ```

3. **Accéder à l'interface web**
   - L'application sera disponible par défaut à l'adresse: http://localhost:8501

## Recommandations pour fichiers volumineux (>1 Go)

Si vous traitez des fichiers de plus de 1 Go, voici quelques recommandations:

1. **Mémoire RAM**
   - Minimum recommandé: 8 Go
   - Optimal: 16 Go ou plus pour les fichiers très volumineux (>2 Go)

2. **Configuration des lots**
   - Ajustez la taille des lots (batch size) dans l'interface selon votre mémoire disponible:
     - Pour mémoire limitée (<8 Go): utilisez des lots de 5 000 molécules
     - Pour mémoire standard (8-16 Go): utilisez des lots de 10 000 à 25 000 molécules
     - Pour mémoire élevée (>16 Go): utilisez des lots de 50 000 à 100 000 molécules

3. **Optimisation Conda**
   - Utilisez Conda pour RDKit car les versions précompilées sont optimisées
   - Définissez la variable d'environnement pour limiter l'utilisation de threads par OpenMP si nécessaire:
     ```bash
     # Pour limiter à 4 threads:
     export OMP_NUM_THREADS=4
     ```

4. **Formats de fichier**
   - Préférez le format CSV avec séparateur virgule ou tabulation
   - Assurez-vous que votre fichier est correctement encodé (UTF-8)

## Processus de nettoyage

Le processus de nettoyage comprend les étapes suivantes:

1. Conversion initiale SMILES → Mol
2. Suppression des sels et solvants
3. Traitement des fragments multiples
4. Suppression des molécules avec atomes exotiques
5. Génération du SMILES canonique et isomérique
6. Détection et correction des atomes spéciaux
7. Vérification finale et suppression des doublons

## Fichiers de sortie

Tous les résultats sont sauvegardés dans un dossier `cleanchem_results`:

- `molecules_finales.csv`: Collection complète des molécules nettoyées
- `molecules_invalides.csv`: Structures qui n'ont pas pu être traitées correctement
- `molecules_atomes_exotiques.csv`: Molécules contenant des atomes non médicinaux
- `molecules_isotopes.csv`: Molécules qui contenaient des isotopes spécifiques
- `molecules_similaires.csv`: Molécules identifiées comme similaires ou en doublon
- `molecules_fragments.csv`: Molécules originales contenant des ions ou fragments multiples

## Dépannage

**Erreur de mémoire insuffisante**
- Réduisez la taille des lots dans les options de l'interface
- Augmentez la mémoire swap (Linux/macOS) ou la mémoire virtuelle (Windows)
- Subdivisez votre fichier d'entrée en plusieurs fichiers plus petits

**Problèmes avec RDKit**
- Assurez-vous d'installer RDKit via Conda et non pip pour éviter les problèmes de compatibilité
- Si vous rencontrez des erreurs avec RDKit, essayez de réinstaller avec:
  ```bash
  conda install -c conda-forge rdkit --force-reinstall
  ```

**Traitement lent**
- Vérifiez l'utilisation CPU et mémoire dans l'interface
- Désactivez l'affichage des images pour accélérer le traitement
- Limitez les threads OpenMP si RDKit utilise trop de ressources

**Problèmes spécifiques à Streamlit sous Conda**
- Si Streamlit ne lance pas correctement sous Conda, essayez:
  ```bash
  pip uninstall streamlit
  pip install streamlit
  ```

## Pour mettre à jour l'environnement

Pour mettre à jour les dépendances ultérieurement:
```bash
conda activate cleanchem
conda update -c conda-forge rdkit pandas numpy
pip install --upgrade streamlit psutil
```
## Auteur

Youcef BAGDAD, 2025
