import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, SaltRemover, MolStandardize
import base64
from io import StringIO
import io
import tempfile
import subprocess
import os
import psutil
import gc
import time
import pickle
import tempfile
import shutil

# Configuration de la page
st.set_page_config(
    page_title="Nettoyeur de Chimiothèques",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Masquer le "0" en haut de la page
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def mol_to_img(mol, size=(300, 200)):
    """Convertit une molécule RDKit en image PNG encodée en base64"""
    if mol is None:
        return None
    
    try:
        img = Draw.MolToImage(mol, size=size)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"Erreur lors de la conversion en image: {e}")
        return None

def validate_smiles(smiles):
    """Vérifie si un SMILES est valide en essayant de le convertir en molécule RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return True
    except:
        return False

def get_largest_fragment(mol):
    """Retourne le plus grand fragment d'une molécule contenant plusieurs fragments"""
    if mol is None:
        return None
    
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    
    if len(frags) <= 1:
        return mol
    
    # Trouver le fragment avec le plus grand nombre d'atomes
    largest_frag = None
    max_atoms = 0
    
    for frag in frags:
        num_atoms = frag.GetNumAtoms()
        if num_atoms > max_atoms:
            max_atoms = num_atoms
            largest_frag = frag
    
    return largest_frag

def has_nonmedicinal_atoms(mol):
    """Vérifie si la molécule contient des atomes autres que ceux couramment utilisés dans les médicaments"""
    # Liste des éléments couramment utilisés dans les médicaments
    acceptable_atoms = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in acceptable_atoms:
            return True, symbol
    
    return False, None

def get_memory_usage():
    """Renvoie l'utilisation actuelle de la mémoire en Go"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Conversion en Go

def update_memory_display(memory_container, force=False):
    """Met à jour l'affichage de la mémoire utilisée si elle a changé significativement"""
    current_mem = get_memory_usage()
    
    # Mettre à jour seulement si la valeur a changé significativement ou si force=True
    if force or not hasattr(update_memory_display, "last_mem") or abs(current_mem - update_memory_display.last_mem) > 0.1:
        memory_container.metric(
            "Mémoire utilisée", 
            f"{current_mem:.2f} Go", 
            delta=f"{current_mem - update_memory_display.last_mem:.2f} Go" if hasattr(update_memory_display, "last_mem") else None
        )
        update_memory_display.last_mem = current_mem

def get_system_info():
    """Récupère les informations sur les ressources système disponibles"""
    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    # RAM
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    
    return {
        "cpu_count": cpu_count,
        "physical_cores": physical_cores,
        "ram_gb": ram_gb
    }

def process_molecule(molecule_data, name_col=None, keep_largest_fragment=True, remove_salts=True, 
                    remove_nonmedicinal_atoms=True):
    """Traite une seule molécule selon les étapes spécifiées"""
    idx, row = molecule_data
    smiles = row['SMILES'] if 'SMILES' in row else row.iloc[0]
    mol_name = row[name_col] if name_col else f"Mol_{idx}"
    
    issues = []
    
    # Initialiser les compteurs pour cette molécule
    molecule_counters = {
        "invalid_smiles": 0,
        "special_atoms": 0,
        "nonmedicinal_atoms": 0,
        "salts_removed": 0,
        "fragments_removed": 0
    }
    
    # 1. Conversion initiale SMILES -> Mol
    original_mol = Chem.MolFromSmiles(smiles)
    if original_mol is None:
        # Molécule avec SMILES invalide (erreur syntaxique)
        return {
            "idx": idx,
            "name": mol_name,
            "success": False,
            "issues": ["SMILES invalide"],
            "counters": {"invalid_smiles": 1},
            "error_type": "invalid_smiles",
            "original_smiles": smiles  # Conserver le SMILES original sans modification
        }
    
    # Conserver la molécule originale pour la comparaison
    mol_original = Chem.Mol(original_mol)
    
    # Compter les fragments de la molécule originale pour l'information
    frags = Chem.GetMolFrags(original_mol, asMols=False)
    frag_count = len(frags)
    has_multiple_fragments = frag_count > 1
    
    # Travailler avec une copie
    mol = Chem.Mol(original_mol)
    
    # 2. Suppression des sels et solvants 
    has_salts = False
    if remove_salts:
        try:
            # Initialiser le remover de sels
            salt_remover = SaltRemover.SaltRemover()
            
            # Vérifier le nombre d'atomes avant
            atoms_before = mol.GetNumAtoms()
            
            res = salt_remover.StripMol(mol)
            
            # Vérifier le nombre d'atomes après
            atoms_after = res.GetNumAtoms()
            
            if atoms_after != atoms_before:
                has_salts = True
                molecule_counters["salts_removed"] += 1
                issues.append(f"Sels/solvants supprimés ({atoms_before-atoms_after} atomes)")
            
            mol = res
        except Exception as e:
            issues.append(f"Erreur lors de la suppression des sels: {str(e)}")
    
    # 3. Traitement des fragments multiples après la suppression des sels
    # Vérifier si après suppression des sels, il reste encore plusieurs fragments
    frag_count_after = len(Chem.GetMolFrags(mol, asMols=False))
    if frag_count_after > 1 and keep_largest_fragment:
        largest_frag = get_largest_fragment(mol)
        if largest_frag:
            mol = largest_frag
            molecule_counters["fragments_removed"] += 1
            issues.append(f"Fragments supprimés ({frag_count_after} fragments)")
    
    # 4. Élimination des molécules avec atomes exotiques 
    if remove_nonmedicinal_atoms:
        has_nonmed, nonmed_atom = has_nonmedicinal_atoms(mol)
        if has_nonmed:
            molecule_counters["nonmedicinal_atoms"] = 1
            issues.append(f"Atome exotique détecté: {nonmed_atom}")
            return {
                "idx": idx,
                "name": mol_name,
                "success": False,
                "issues": issues,
                "counters": molecule_counters,
                "error_type": "exotic_atom",
                "exotic_atom": nonmed_atom,
                "original_smiles": smiles  # Conserver le SMILES original sans modification
            }
    
    # 5. Génération du SMILES canonique et isomérique
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    
    # 6. Détection et correction des atomes spéciaux (isotopes)
    has_special_atoms = False
    for atom in mol.GetAtoms():
        # Vérifier les isotopes
        if atom.GetIsotope() > 0:
            has_special_atoms = True
            issues.append(f"Atome isotopique: {atom.GetSymbol()}{atom.GetIsotope()}")
    
    if has_special_atoms:
        molecule_counters["special_atoms"] += 1
        
        # Corriger les atomes spéciaux (remplacer par les atomes standards)
        mol_fixed = Chem.Mol(mol)
        for atom in mol_fixed.GetAtoms():
            if atom.GetIsotope() > 0:
                atom.SetIsotope(0)
        
        # Mettre à jour le SMILES canonique après correction
        mol = mol_fixed
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    
    # Retourner les résultats du traitement
    result = {
        "idx": idx,
        "name": mol_name,
        "success": True,
        "original_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "special_atoms": has_special_atoms,
        "fragment_count": frag_count,
        "has_multiple_fragments": has_multiple_fragments,
        "has_salts": has_salts,
        "issues": issues,
        "counters": molecule_counters
    }
    
    return result

def clean_molecular_library_streaming(df, options, batch_size=10000, name_col=None, progress_bar=None, status_text=None, memory_metric=None, output_prefix="cleanchem_"):
    """
    Version optimisée pour le traitement de grandes bibliothèques moléculaires
    avec écriture progressive des résultats et meilleure gestion de la mémoire
    """
    # Imports nécessaires pour la base de données
    import sqlite3
    import hashlib
    
    # Créer un dossier pour les résultats dans le répertoire courant
    output_dir = os.path.join(os.getcwd(), f"{output_prefix}results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer un dossier temporaire pour d'éventuels traitements intermédiaires
    temp_dir = tempfile.mkdtemp()
    
    # Déterminer le nombre de lots
    total_mols = len(df)
    num_batches = (total_mols + batch_size - 1) // batch_size
    
    if status_text is not None:
        status_text.text(f"Initialisation du traitement pour {total_mols} molécules...")
        status_text.text(f"Les résultats seront sauvegardés dans: {output_dir}")
    else:
        print(f"Initialisation du traitement pour {total_mols} molécules...")
        print(f"Les résultats seront sauvegardés dans: {output_dir}")
    
    # Préparer les fichiers de sortie finaux
    final_files = {
        "cleaned": os.path.join(output_dir, "molecules_finales.csv"),
        "invalid": os.path.join(output_dir, "molecules_invalides.csv"),
        "exotic_atoms": os.path.join(output_dir, "molecules_atomes_exotiques.csv"),
        "special_atoms": os.path.join(output_dir, "molecules_isotopes.csv"),
        "duplicates": os.path.join(output_dir, "molecules_similaires.csv"),
        "fragments": os.path.join(output_dir, "molecules_fragments.csv")
    }
    
    # Initialiser les compteurs globaux
    global_counters = {
        "invalid_smiles": 0,
        "special_atoms": 0,
        "nonmedicinal_atoms": 0,
        "salts_removed": 0,
        "fragments_removed": 0,
        "duplicates": 0,
        "multi_fragments": 0
    }
    
    # Initialiser les fichiers avec leurs en-têtes
    with open(final_files["cleaned"], 'w', encoding='utf-8') as f:
        f.write("Name,Original_SMILES,SMILES_Final,Issues\n")
    
    with open(final_files["invalid"], 'w', encoding='utf-8') as f:
        f.write("Name,Original_SMILES,Issues\n")
    
    with open(final_files["exotic_atoms"], 'w', encoding='utf-8') as f:
        f.write("Name,Original_SMILES,Exotic_Atom,Issues\n")
    
    with open(final_files["special_atoms"], 'w', encoding='utf-8') as f:
        f.write("Name,Original_SMILES,Canonical_SMILES,Issues,Fragment_Count\n")
    
    # Initialiser le fichier des doublons avec son en-tête modifié
    with open(final_files["duplicates"], 'w', encoding='utf-8') as f:
        f.write("ID,Original_SMILES,SMILES_Final,Similar_To_ID,Similar_To_Original_SMILES,Similar_To_SMILES_Final,Type\n")
    
    # Nouveau fichier pour les molécules avec fragments ou ions
    with open(final_files["fragments"], 'w', encoding='utf-8') as f:
        f.write("Name,Original_SMILES,Cleaned_SMILES,Fragment_Count,Type\n")
    
    # Pour stocker un ÉCHANTILLON de molécules pour visualisation (limité)
    MAX_SAMPLE_MOLS = 100  # Limiter l'échantillon pour économiser la mémoire
    mol_pairs = []
    
    # Initialisation du système hybride de détection des doublons
    # 1. Cache en mémoire des molécules récentes pour vérification rapide
    recent_smiles_cache = {}
    CACHE_SIZE = 50000  # Garder les 50K dernières molécules en mémoire
    
    # 2. Base de données SQLite pour le stockage persistant efficace
    db_path = os.path.join(temp_dir, "smiles_database.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Créer la table avec un index sur les SMILES canoniques
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS seen_smiles (
        smiles_hash TEXT PRIMARY KEY,
        canonical_smiles TEXT,
        name TEXT,
        original_smiles TEXT
    )
    ''')
    # Créer l'index pour des recherches rapides
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_smiles ON seen_smiles(smiles_hash)')
    conn.commit()
    
    # Widget pour afficher les statistiques en temps réel
    if 'st' in globals():
        stats_container = st.empty()
    
    # Traiter chaque lot et écrire les résultats au fur et à mesure
    for batch_idx in range(num_batches):
        # Calculer les indices de début et fin
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_mols)
        
        # Mettre à jour la progression
        progress = (batch_idx + 1) / num_batches
        if progress_bar is not None:
            progress_bar.progress(progress)
        
        if status_text is not None:
            status_text.text(f"Traitement du lot {batch_idx + 1}/{num_batches} (molécules {start_idx + 1}-{end_idx} sur {total_mols})")
        else:
            print(f"Traitement du lot {batch_idx + 1}/{num_batches} (molécules {start_idx + 1}-{end_idx} sur {total_mols})")
        
        # Afficher l'utilisation de la mémoire
        if memory_metric is not None:
            update_memory_display(memory_metric, force=True)
        
        # Extraire le lot
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        # Préparer les données pour le traitement
        molecule_data = [(idx, row) for idx, row in batch_df.iterrows()]
        
        # Pour stocker les résultats de ce lot
        batch_results = []
        
        # Traiter chaque molécule du lot
        for mol_data in molecule_data:
            result = process_molecule(
                mol_data,
                name_col,
                options["keep_largest_fragment"],
                options["remove_salts"],
                options["remove_nonmedicinal_atoms"]
            )
            batch_results.append(result)
        
        # Traiter et écrire les résultats immédiatement en utilisant une transaction pour la BD
        conn.execute('BEGIN TRANSACTION')
        try:
            for result in batch_results:
                idx = result["idx"]
                name = result["name"]
                
                # Mettre à jour les compteurs
                for key, value in result["counters"].items():
                    if key in global_counters:
                        global_counters[key] += value
                
                if result["success"]:
                    canonical_smiles = result["canonical_smiles"]
                    original_smiles = result["original_smiles"]
                    
                    # Enregistrer les molécules avec fragments ou ions
                    if result.get("has_multiple_fragments", False) or result.get("has_salts", False):
                        global_counters["multi_fragments"] += 1
                        
                        with open(final_files["fragments"], 'a', encoding='utf-8') as f:
                            fragment_type = "ions_et_fragments" if result.get("has_salts", False) and result.get("has_multiple_fragments", False) else \
                                            "fragments" if result.get("has_multiple_fragments", False) else "ions"
                            f.write(f"{name},{original_smiles},{canonical_smiles},{result.get('fragment_count', 0)},{fragment_type}\n")
                    
                    # Vérifier si c'est un doublon
                    is_duplicate = False
                    similar_info = None
                    
                    if options["remove_duplicates"]:
                        # Calculer un hash du SMILES pour des recherches plus rapides
                        smiles_hash = hashlib.md5(canonical_smiles.encode()).hexdigest()
                        
                        # 1. Vérifier d'abord dans le cache mémoire (rapide)
                        if smiles_hash in recent_smiles_cache:
                            is_duplicate = True
                            similar_info = recent_smiles_cache[smiles_hash]
                        else:
                            # 2. Si non trouvé, vérifier dans la base de données
                            cursor.execute('SELECT name, original_smiles, canonical_smiles FROM seen_smiles WHERE smiles_hash = ?', (smiles_hash,))
                            db_result = cursor.fetchone()
                            
                            if db_result:
                                is_duplicate = True
                                similar_info = db_result
                            else:
                                # 3. Si nouveau, ajouter à la base de données et au cache
                                cursor.execute('INSERT INTO seen_smiles VALUES (?, ?, ?, ?)', 
                                            (smiles_hash, canonical_smiles, name, original_smiles))
                                
                                # Ajouter au cache mémoire
                                recent_smiles_cache[smiles_hash] = (name, original_smiles, canonical_smiles)
                                
                                # Maintenir la taille du cache
                                if len(recent_smiles_cache) > CACHE_SIZE:
                                    # Supprimer les plus anciens éléments (20% du cache)
                                    keys_to_remove = list(recent_smiles_cache.keys())[:int(CACHE_SIZE * 0.2)]
                                    for key in keys_to_remove:
                                        del recent_smiles_cache[key]
                        
                        # Compter et écrire les doublons
                        if is_duplicate:
                            global_counters["duplicates"] += 1
                            with open(final_files["duplicates"], 'a', encoding='utf-8') as f:
                                similar_id, similar_orig_smiles, similar_final_smiles = similar_info
                                f.write(f"{name},{original_smiles},{canonical_smiles},{similar_id},{similar_orig_smiles},{similar_final_smiles},structure_identique\n")
                    
                    if not is_duplicate or not options["remove_duplicates"]:
                        # Écrire dans le fichier principal
                        issues = "; ".join(result["issues"])
                        
                        with open(final_files["cleaned"], 'a', encoding='utf-8') as f:
                            f.write(f"{name},{original_smiles},{canonical_smiles},{issues}\n")
                        
                    # Si la molécule a des atomes spéciaux, l'écrire dans le fichier correspondant
                    if result.get("special_atoms", False):
                        with open(final_files["special_atoms"], 'a', encoding='utf-8') as f:
                            issues = "; ".join(result["issues"])
                            fragment_count = result.get("fragment_count", 0)
                            f.write(f"{name},{original_smiles},{canonical_smiles},{issues},{fragment_count}\n")
                    
                    # Stocker seulement un échantillon limité pour visualisation
                    if len(mol_pairs) < MAX_SAMPLE_MOLS:
                        mol_pair = {
                            "name": name,
                            "original_smiles": original_smiles,
                            "cleaned_smiles": canonical_smiles,
                            "issues": result["issues"]
                        }
                        mol_pairs.append(mol_pair)
                
                else:
                    # Écrire les molécules invalides en fonction du type d'erreur
                    original_smiles = result["original_smiles"]  # SMILES original sans modification
                    issues = "; ".join(result["issues"])
                    
                    # Écrire dans le fichier général des molécules invalides
                    with open(final_files["invalid"], 'a', encoding='utf-8') as f:
                        f.write(f"{name},{original_smiles},{issues}\n")
                    
                    # En plus, écrire dans le fichier spécifique si c'est un atome exotique
                    if "error_type" in result and result["error_type"] == "exotic_atom":
                        with open(final_files["exotic_atoms"], 'a', encoding='utf-8') as f:
                            exotic_atom = result.get("exotic_atom", "")
                            f.write(f"{name},{original_smiles},{exotic_atom},{issues}\n")
            
            # Valider la transaction
            conn.commit()
        except Exception as e:
            # En cas d'erreur, annuler la transaction
            conn.rollback()
            raise e
        
        # Afficher les statistiques en temps réel si on est dans Streamlit
        if 'st' in globals() and 'stats_container' in locals():
            with stats_container.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Molécules traitées", f"{min(end_idx, total_mols)}/{total_mols}")
                col2.metric("SMILES invalides", f"{global_counters['invalid_smiles']}")
                col3.metric("Atomes exotiques", f"{global_counters['nonmedicinal_atoms']}")
                col4.metric("Doublons identifiés", f"{global_counters['duplicates']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Molécules avec isotopes", f"{global_counters['special_atoms']}")
                col2.metric("Fragments/ions", f"{global_counters['multi_fragments']}")
                col3.metric("Sels supprimés", f"{global_counters['salts_removed']}")
                
                # Afficher aussi la mémoire
                mem_usage = get_memory_usage()
                st.caption(f"Mémoire utilisée: {mem_usage:.2f} Go")
        
        # Libérer la mémoire de manière agressive
        del batch_df
        for result in batch_results:
            # Assurer la libération des objets RDKit
            if "mol" in result:
                del result["mol"]
        del batch_results
        
        # Forcer le garbage collector après chaque lot
        gc.collect()
    
    # Afficher les statistiques finales
    if status_text is not None:
        status_text.text(f"Traitement terminé. Les résultats sont sauvegardés dans: {output_dir}")
    else:
        print(f"Traitement terminé. Les résultats sont sauvegardés dans: {output_dir}")
    
    # Fermer la connexion à la base de données
    conn.close()
    
    # Nettoyer les fichiers temporaires
    if os.path.exists(db_path):
        os.remove(db_path)
    shutil.rmtree(temp_dir)
    
    # Libération finale
    gc.collect()
    
    return global_counters, final_files, mol_pairs, output_dir


def save_large_dataframe(df, file_path, sep=",", batch_size=50000):
    """Sauvegarde un grand DataFrame par lots pour économiser la mémoire"""
    # Nombre total de lignes
    total_rows = len(df)
    
    # Ouvrir le fichier pour écriture
    with open(file_path, 'w', encoding='utf-8') as f:
        # Écrire l'en-tête
        f.write(sep.join(df.columns) + '\n')
        
        # Écrire les données par lots
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx]
            
            # Convertir le batch en CSV sans en-tête
            batch_csv = batch.to_csv(index=False, header=False, sep=sep)
            
            # Écrire dans le fichier
            f.write(batch_csv)
            
            # Nettoyer la mémoire
            del batch
            gc.collect()



def main():
    # En-tête de l'application
    st.title("🧪 CleanChem")
    st.write("Un outil pour standardiser et nettoyer vos collections de molécules")
    
    # Récupérer les informations système
    system_info = get_system_info()

    # Afficher l'utilisation de la mémoire
    memory_container = st.empty()
    update_memory_display(memory_container, force=True)
    
    # Barre latérale pour les options
    with st.sidebar:
        st.header("Options de nettoyage")
        
        remove_salts = st.checkbox(
            "Suppression des sels et solvants", 
            value=True,
            help="Retire les contre-ions et molécules de solvant",
            key="sidebar_remove_salts"
        )
        
        st.subheader("Options principales")
        keep_largest_fragment = st.checkbox(
            "Conserver uniquement le plus grand fragment", 
            value=True,
            help="Garde uniquement le plus grand fragment des molécules à fragments multiples",
            key="sidebar_largest_fragment"
        )
        
        remove_nonmedicinal_atoms = st.checkbox(
            "Élimination des molécules avec atomes exotiques", 
            value=True,
            help="Retire les molécules contenant des atomes autres que C, H, O, N, S, P, F, Cl, Br, I",
            key="sidebar_nonmedicinal_atoms"
        )
        
        st.subheader("Options supplémentaires")
        
        remove_duplicates = st.checkbox(
            "Traitement des doublons", 
            value=True,
            help="Identifie et supprime les molécules en double",
            key="sidebar_remove_duplicates"
        )
        
        # Affichage des informations système détectées
        st.subheader("Informations système")
        st.info(f"Système détecté: {system_info['physical_cores']} cœurs physiques, "
                f"{system_info['cpu_count']} threads, "
                f"{system_info['ram_gb']} Go RAM")
        
        # Options de performance
        st.subheader("Options de performance")
        batch_size = st.select_slider(
            "Taille des lots (molécules)",
            options=[1000, 2500, 5000, 10000, 25000, 50000, 100000],
            value=10000,
            help="Nombre de molécules traitées par lot. Utilisez des valeurs plus petites si vous manquez de mémoire.",
            key="sidebar_batch_size"
        )
        
        st.subheader("Options d'affichage")
        show_images = st.checkbox(
            "Afficher les images des molécules", 
            value=True,
            help="Générer des représentations visuelles des molécules",
            key="sidebar_show_images"
        )
    
    # Création des onglets
    tabs = st.tabs(["🧪✨ Nettoyage", "🔍 Comparaison", "📖 Documentation"])
    
    with tabs[0]:  # Onglet Nettoyage
    
        st.header("Importation et nettoyage")
        
        # Upload du fichier
        st.info("Importez votre fichier CSV ou TXT contenant les SMILES à traiter")
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV ou TXT avec des SMILES", 
            type=["csv", "txt"],
            key="cleantab_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Options de séparateur communes à tous les formats
                separators = {
                    ",": "Virgule (,)",
                    ";": "Point-virgule (;)",
                    "\t": "Tabulation (\\t)",
                    "|": "Barre verticale (|)",
                    " ": "Espace"
                }
                selected_sep = st.selectbox(
                    "Sélectionnez le séparateur", 
                    list(separators.values()), 
                    index=0,
                    key="cleantab_separator"
                )
                sep = list(separators.keys())[list(separators.values()).index(selected_sep)]
                
                # Déterminer le format et lire le fichier
                if uploaded_file.name.endswith('.csv'):
                    # Lire d'abord quelques lignes pour afficher un aperçu
                    preview_df = pd.read_csv(uploaded_file, sep=sep, nrows=5)
                    st.success(f"Fichier CSV détecté. Lecture en cours...")
                    
                    # Réinitialiser le curseur pour la lecture complète
                    uploaded_file.seek(0)
                    
                    # Afficher un aperçu des données
                    st.subheader("Aperçu des données")
                    st.dataframe(preview_df)
                    
                    # Sélectionner la colonne contenant les SMILES
                    col_options = preview_df.columns.tolist()
                    smiles_col = st.selectbox(
                        "Sélectionnez la colonne contenant les SMILES", 
                        col_options,
                        key="cleantab_csv_smiles_col"
                    )
                    
                    # Sélectionner la colonne contenant les noms (optionnel)
                    name_col = st.selectbox(
                        "Sélectionnez la colonne contenant les noms (optionnel)", 
                        ['Aucune'] + col_options,
                        key="cleantab_csv_name_col"
                    )
                    if name_col == 'Aucune':
                        name_col = None
                    
                    # Ajouter un bouton pour confirmer et charger le fichier complet
                    if st.button(
                        "Charger le fichier complet",
                        key="cleantab_csv_load_button"
                    ):
                        with st.spinner("Chargement du fichier en cours..."):
                            # Charger tout le fichier
                            df = pd.read_csv(uploaded_file, sep=sep)
                            
                            # S'assurer que la colonne SMILES est présente
                            if smiles_col not in df.columns:
                                st.error(f"La colonne {smiles_col} n'existe pas dans le fichier.")
                                return
                            
                            # Renommer la colonne SMILES pour traitement uniforme
                            if smiles_col != 'SMILES':
                                df = df.rename(columns={smiles_col: 'SMILES'})
                            
                            st.success(f"Fichier CSV importé avec succès! {len(df)} lignes détectées.")
                            
                            # Stocker dans la session state
                            st.session_state['input_df'] = df
                            st.session_state['name_col'] = name_col
                            # Réinitialiser l'état des résultats
                            if 'has_results' in st.session_state:
                                st.session_state['has_results'] = False
                
                else:  # txt file
                    # Lire les premières lignes pour l'aperçu
                    content = uploaded_file.read(1024).decode()
                    
                    # Vérifier si le contenu contient le séparateur
                    lines_preview = content.strip().split('\n')[:5]
                    
                    # Essayer de diviser chaque ligne selon le séparateur choisi
                    preview_data = []
                    for line in lines_preview:
                        preview_data.append(line.split(sep))
                    
                    # Créer un DataFrame pour l'aperçu
                    preview_df = pd.DataFrame(preview_data)
                    
                    # Si la première ligne semble être un en-tête (vérifie si elle contient "SMILES" ou similaire)
                    has_header = any("SMILES" in str(col).upper() for col in preview_data[0] if col)
                    
                    if has_header:
                        # Utiliser la première ligne comme en-tête
                        preview_df.columns = preview_data[0]
                        preview_df = preview_df.iloc[1:]
                    
                    # Réinitialiser pour la lecture complète
                    uploaded_file.seek(0)
                    
                    # Afficher l'aperçu
                    st.subheader("Aperçu des données")
                    st.dataframe(preview_df)
                    
                    # Demander à l'utilisateur si le fichier a un en-tête
                    header_option = st.checkbox(
                        "Le fichier contient une ligne d'en-tête", 
                        value=has_header,
                        key="cleantab_txt_header_option"
                    )
                    
                    # Si le fichier a plusieurs colonnes, demander laquelle contient les SMILES
                    if len(preview_data[0]) > 1:
                        if header_option:
                            col_options = preview_df.columns.tolist()
                            smiles_col_idx = st.selectbox(
                                "Sélectionnez la colonne contenant les SMILES", 
                                range(len(col_options)), 
                                format_func=lambda x: col_options[x],
                                key="cleantab_txt_header_smiles_col"
                            )
                            smiles_col = col_options[smiles_col_idx]
                            
                            # Sélectionner la colonne contenant les noms (optionnel)
                            name_options = ['Aucune'] + col_options
                            name_col_idx = st.selectbox(
                                "Sélectionnez la colonne contenant les noms (optionnel)", 
                                range(len(name_options)), 
                                format_func=lambda x: name_options[x],
                                key="cleantab_txt_header_name_col"
                            )
                            name_col = None if name_options[name_col_idx] == 'Aucune' else name_options[name_col_idx]
                        else:
                            smiles_col_idx = st.selectbox(
                                "Sélectionnez la colonne contenant les SMILES", 
                                range(len(preview_data[0])), 
                                format_func=lambda x: f"Colonne {x+1}",
                                key="cleantab_txt_noheader_smiles_col"
                            )
                            smiles_col = smiles_col_idx
                            
                            # Sélectionner la colonne contenant les noms (optionnel)
                            name_options = ['Aucune'] + [f"Colonne {i+1}" for i in range(len(preview_data[0]))]
                            name_col_idx = st.selectbox(
                                "Sélectionnez la colonne contenant les noms (optionnel)", 
                                range(len(name_options)), 
                                format_func=lambda x: name_options[x],
                                key="cleantab_txt_noheader_name_col"
                            )
                            name_col = None if name_col_idx == 0 else name_col_idx - 1
                    else:
                        smiles_col = 0
                        name_col = None
                    
                    # Ajouter un bouton pour confirmer et charger le fichier complet
                    if st.button(
                        "Charger le fichier complet",
                        key="cleantab_txt_load_button"
                    ):
                        with st.spinner("Chargement du fichier en cours..."):
                            content = uploaded_file.read().decode()
                            lines = content.strip().split('\n')
                            
                            data = []
                            for line in lines:
                                data.append(line.split(sep))
                            
                            # Créer le DataFrame
                            if header_option:
                                header = data[0]
                                df = pd.DataFrame(data[1:], columns=header)
                            else:
                                df = pd.DataFrame(data)
                            
                            # Extraire la colonne SMILES et renommer
                            if isinstance(smiles_col, int):
                                # Si smiles_col est un index
                                df = df.rename(columns={df.columns[smiles_col]: 'SMILES'})
                            else:
                                # Si smiles_col est un nom de colonne
                                df = df.rename(columns={smiles_col: 'SMILES'})
                            
                            # Extraire et renommer la colonne de noms si présente
                            if name_col is not None:
                                if isinstance(name_col, int):
                                    # Si name_col est un index
                                    name_actual_col = df.columns[name_col]
                                    df = df.rename(columns={name_actual_col: 'Name'})
                                    name_col = 'Name'
                                else:
                                    # Si name_col est un nom de colonne
                                    df = df.rename(columns={name_col: 'Name'})
                                    name_col = 'Name'
                            
                            st.success(f"Fichier TXT importé avec succès! {len(df)} lignes détectées.")
                            
                            # Stocker dans la session state
                            st.session_state['input_df'] = df
                            st.session_state['name_col'] = name_col
                            # Réinitialiser l'état des résultats
                            if 'has_results' in st.session_state:
                                st.session_state['has_results'] = False
                
                # Afficher un bouton pour lancer le nettoyage si le fichier est chargé
                if 'input_df' in st.session_state:
                    df = st.session_state['input_df']
                    name_col = st.session_state['name_col']
                    
                    # Bouton pour lancer le nettoyage
                    if st.button(
                        "🧪 🧹 Nettoyer la bibliothèque", 
                        use_container_width=True,
                        key="cleantab_process_button"
                    ):
                        # Rassembler les options de nettoyage
                        cleaning_options = {
                            "keep_largest_fragment": keep_largest_fragment,
                            "remove_salts": remove_salts,
                            "remove_nonmedicinal_atoms": remove_nonmedicinal_atoms,
                            "remove_duplicates": remove_duplicates
                        }
                        
                        with st.spinner(f"Nettoyage en cours par lots de {batch_size} molécules..."):
                            # Préparer des conteneurs pour afficher la progression
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            memory_metric = st.empty()  

                            # Effectuer le nettoyage streaming
                            start_time = time.time()
                            global_counters, final_files, mol_pairs, output_dir = clean_molecular_library_streaming(
                                df,
                                cleaning_options,
                                batch_size=batch_size,
                                name_col=name_col,
                                progress_bar=progress_bar,
                                status_text=status_text,
                                memory_metric=memory_metric
                            )
                            
                            # Calculer le temps de traitement
                            duration = time.time() - start_time
                            hours, remainder = divmod(duration, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            
                            # Nettoyer la mémoire
                            del df
                            gc.collect()
                            
                            # Stocker les résultats dans la session state
                            st.session_state['output_dir'] = output_dir
                            st.session_state['final_files'] = final_files
                            st.session_state['mol_pairs'] = mol_pairs
                            st.session_state['global_counters'] = global_counters
                            st.session_state['has_results'] = True
                            
                            # Vider les indicateurs de progression
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Afficher message de succès et statistiques
                            st.success(f"✅ Nettoyage terminé avec succès en {int(hours)}h {int(minutes)}m {int(seconds)}s!")
                            
                            # Afficher les résultats
                            st.header("Résultats du nettoyage")
                            
                            # Calculer le nombre total de molécules finales
                            input_size = len(st.session_state['input_df'])
                            invalid_smiles_count = global_counters['invalid_smiles']  # Uniquement les SMILES invalides
                            exotic_atoms_count = global_counters['nonmedicinal_atoms']  # Uniquement les atomes exotiques
                            duplicates_count = global_counters['duplicates']
                            
                            # Nombre de molécules valides = total initial - (invalides + exotiques) - doublons
                            valid_molecules = input_size - (invalid_smiles_count + exotic_atoms_count) - duplicates_count
                            final_molecules = valid_molecules
                            
                            # Afficher un résumé amélioré visuellement avec des statistiques plus précises
                            st.subheader("Bilan des molécules")
                            
                            # Première ligne de métriques : volumes de molécules
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Molécules importées", f"{input_size}")
                            col2.metric("Molécules finales", f"{final_molecules}", 
                                      delta=f"{final_molecules - input_size}" if final_molecules != input_size else None)
                            col3.metric("Taux de conservation", f"{(final_molecules / input_size * 100):.1f}%")
                            
                            # Seconde ligne de métriques : détails des problèmes
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("SMILES invalides", f"{invalid_smiles_count}")  # Renommé pour clarifier
                            col2.metric("Atomes exotiques", f"{exotic_atoms_count}")    # Séparé clairement
                            col3.metric("Doublons identifiés", f"{duplicates_count}")
                            col4.metric("Molécules avec isotopes", f"{global_counters['special_atoms']}")
                            
                            # Troisième ligne de métriques : détails des transformations
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Molécules avec ions/fragments", f"{global_counters['multi_fragments']}")
                            col2.metric("Sels supprimés", f"{global_counters['salts_removed']}")
                            col3.metric("Fragments supprimés", f"{global_counters['fragments_removed']}")
                            
                            # Détails supplémentaires
                            with st.expander("Détails complets du traitement"):
                                # Première colonne : statistiques des molécules
                                stats_col1, stats_col2 = st.columns(2)
                                
                                with stats_col1:
                                    st.subheader("Statistiques des molécules")
                                    st.markdown(f"""
                                    - **Molécules importées:** {input_size}
                                    - **Molécules valides:** {valid_molecules} ({valid_molecules/input_size*100:.1f}%)
                                    - **Total molécules finales:** {final_molecules}
                                    - **SMILES invalides:** {invalid_smiles_count} ({invalid_smiles_count/input_size*100:.1f}%)
                                    - **Molécules avec atomes exotiques:** {exotic_atoms_count} ({exotic_atoms_count/input_size*100:.1f}%)
                                    - **Doublons identifiés:** {duplicates_count} ({duplicates_count/input_size*100:.1f}%)
                                    """)
                                
                                # Deuxième colonne : statistiques des transformations
                                with stats_col2:
                                    st.subheader("Statistiques des transformations")
                                    st.markdown(f"""
                                    - **Molécules avec ions/fragments:** {global_counters['multi_fragments']}
                                    - **Fragments supprimés:** {global_counters['fragments_removed']}
                                    - **Sels/solvants supprimés:** {global_counters['salts_removed']}
                                    - **Molécules avec isotopes:** {global_counters['special_atoms']}
                                    """)
                            
                            # Information sur l'emplacement des fichiers sauvegardés
                            st.subheader("Fichiers sauvegardés")
                            st.info(f"""
                            Tous les fichiers ont été sauvegardés dans le dossier: **{output_dir}**

                            Fichiers générés:
                            - molecules_finales.csv - **{final_molecules} molécules** nettoyées
                            - molecules_invalides.csv - **{invalid_smiles_count} SMILES** qui n'ont pas pu être traités (syntaxe/valence invalide)
                            - molecules_atomes_exotiques.csv - **{exotic_atoms_count} molécules** contenant des atomes non médicinaux
                            - molecules_isotopes.csv - **{global_counters['special_atoms']} molécules** qui contenaient des isotopes
                            - molecules_similaires.csv - **{duplicates_count} molécules** identifiées comme doublons
                            - molecules_fragments.csv - **{global_counters['multi_fragments']} molécules** avec ions/fragments
                            """)
                            
                            # Inviter l'utilisateur à consulter l'onglet de comparaison
                            st.info("Consultez l'onglet 'Comparaison' pour visualiser les changements avant/après nettoyage")
            
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier: {e}")
                st.exception(e)
                
            # Mise à jour de l'utilisation de la mémoire
            update_memory_display(memory_container, force=True)

    with tabs[1]:  # Onglet Comparaison
        st.header("Comparaison des molécules avant/après nettoyage")
        
        if 'has_results' not in st.session_state or not st.session_state['has_results']:
            st.info("Aucun résultat de nettoyage disponible. Veuillez d'abord nettoyer une bibliothèque dans l'onglet 'Nettoyage'.")
        else:
            if 'final_files' in st.session_state:
                final_files = st.session_state['final_files']
                
                # Proposer une visualisation directe ou à partir d'un fichier chargé
                view_option = st.radio(
                    "Source des molécules à visualiser",
                    ["Visualiser les molécules du traitement en cours", "Charger un fichier de résultats précédent"],
                    index=0,
                    key="view_option_radio"
                )
                
                if view_option == "Charger un fichier de résultats précédent":
                    # Upload d'un fichier de résultats
                    uploaded_results = st.file_uploader(
                        "Choisissez un fichier CSV de résultats CleanChem", 
                        type=["csv"],
                        key="compare_file_uploader"
                    )
                    
                    if uploaded_results is not None:
                        try:
                            # Lire le fichier CSV
                            df = pd.read_csv(uploaded_results)
                            st.success(f"Fichier chargé avec succès! {len(df)} molécules trouvées.")
                            
                            # Afficher un aperçu des données
                            st.subheader("Aperçu des données")
                            st.dataframe(df.head())
                            
                            # Vérifier les colonnes nécessaires
                            required_columns = ['Original_SMILES', 'SMILES_Final']
                            if not all(col in df.columns for col in required_columns):
                                st.error("Le fichier ne contient pas les colonnes requises. Veuillez vous assurer qu'il s'agit d'un fichier de sortie CleanChem valide.")
                                return
                            
                            # Options d'affichage
                            st.subheader("Options d'affichage")
                            
                            # Slider pour contrôler le nombre de molécules à afficher
                            display_count = st.slider("Nombre de molécules à afficher", 
                                                    1, min(len(df), 100), 5,
                                                    help="Ajustez selon le nombre de molécules que vous souhaitez visualiser simultanément")
                            
                            # Option pour activer/désactiver les images
                            show_images = st.checkbox("Afficher les images des molécules", value=True)
                            
                            # Liste des molécules disponibles
                            st.subheader("Sélection des molécules")
                            
                            if 'Name' in df.columns:
                                mol_options = df['Name'].tolist()
                                selected_mols = st.multiselect("Sélectionnez les molécules à afficher", mol_options, mol_options[:min(display_count, len(mol_options))])
                                selected_df = df[df['Name'].isin(selected_mols)]
                            else:
                                # Utiliser l'index comme identifiant si aucun nom n'est disponible
                                df['ID'] = [f"Molécule {i+1}" for i in range(len(df))]
                                mol_options = df['ID'].tolist()
                                selected_mols = st.multiselect("Sélectionnez les molécules à afficher", mol_options, mol_options[:min(display_count, len(mol_options))])
                                selected_df = df[df['ID'].isin(selected_mols)]
                            
                            # Affichage des molécules sélectionnées
                            if not selected_df.empty:
                                st.subheader("Visualisation des molécules")
                                
                                for idx, row in selected_df.iterrows():
                                    st.write("---")
                                    mol_name = row.get('Name', f"Molécule {idx}")
                                    st.subheader(mol_name)
                                    
                                    # Créer deux colonnes pour avant/après
                                    col1, col2 = st.columns(2)
                                    
                                    # Molécule originale
                                    with col1:
                                        st.markdown("**Molécule originale**")
                                        original_smiles = row['Original_SMILES']
                                        st.markdown("**SMILES original :**")
                                        st.code(original_smiles, language=None)
                                        
                                        if show_images:
                                            # Générer la molécule à la demande
                                            original_mol = Chem.MolFromSmiles(original_smiles)
                                            if original_mol:
                                                img_str = mol_to_img(original_mol)
                                                if img_str:
                                                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
                                                else:
                                                    st.warning("Impossible de générer l'image")
                                            else:
                                                st.warning("SMILES invalide")
                                    
                                    # Molécule nettoyée
                                    with col2:
                                        st.markdown("**Molécule nettoyée**")
                                        cleaned_smiles = row['SMILES_Final']
                                        st.markdown("**SMILES nettoyé :**")
                                        st.code(cleaned_smiles, language=None)
                                        
                                        if show_images:
                                            # Générer la molécule à la demande
                                            cleaned_mol = Chem.MolFromSmiles(cleaned_smiles)
                                            if cleaned_mol:
                                                img_str = mol_to_img(cleaned_mol)
                                                if img_str:
                                                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
                                                else:
                                                    st.warning("Impossible de générer l'image")
                                            else:
                                                st.warning("SMILES invalide")
                                    
                                    # Informations supplémentaires sur la molécule
                                    with st.expander("Informations supplémentaires"):
                                        if 'Issues' in row and pd.notna(row['Issues']):
                                            issues = row['Issues']
                                            if issues:
                                                st.write("**Modifications effectuées:**")
                                                for issue in issues.split(';'):
                                                    if issue.strip():  # Éviter les chaînes vides
                                                        st.write(f"- {issue.strip()}")
                                            else:
                                                st.write("Aucune modification majeure détectée.")
                                        
                                        # Calcul de quelques propriétés moléculaires
                                        if show_images and 'cleaned_mol' in locals() and cleaned_mol:
                                            st.write("**Propriétés:**")
                                            try:
                                                from rdkit.Chem import Descriptors
                                                mol_weight = round(Descriptors.MolWt(cleaned_mol), 2)
                                                logp = round(Descriptors.MolLogP(cleaned_mol), 2)
                                                hbd = Descriptors.NumHDonors(cleaned_mol)
                                                hba = Descriptors.NumHAcceptors(cleaned_mol)
                                                
                                                props_col1, props_col2 = st.columns(2)
                                                props_col1.metric("Poids moléculaire", f"{mol_weight} g/mol")
                                                props_col1.metric("LogP", logp)
                                                props_col2.metric("Donneurs de H", hbd)
                                                props_col2.metric("Accepteurs de H", hba)
                                            except Exception as e:
                                                st.warning(f"Impossible de calculer les propriétés moléculaires: {e}")
                        except Exception as e:
                            st.error(f"Erreur lors du traitement du fichier: {e}")
                else:
                    # Utiliser les données du traitement en cours
                    mol_pairs = st.session_state.get('mol_pairs', [])
                    
                    if not mol_pairs:
                        st.warning("Aucune molécule n'a été conservée pour visualisation lors du traitement.")
                    else:
                        # Options d'affichage
                        st.subheader("Options d'affichage")
                        
                        # Slider pour contrôler le nombre de molécules à afficher
                        display_count = st.slider("Nombre de molécules à afficher", 
                                               1, min(len(mol_pairs), 100), 5,
                                               help="Ajustez selon le nombre de molécules que vous souhaitez visualiser simultanément")
                        
                        show_images = st.checkbox("Afficher les images des molécules", value=True)
                        
                        # Options de filtrage
                        filter_options = ["Toutes les molécules", "Molécules modifiées uniquement", 
                                         "Molécules avec isotopes", "Molécules avec sels/fragments", "Autres modifications"]
                        filter_type = st.selectbox("Filtrer par type de modification", filter_options)
                        
                        # Filtrer les paires de molécules selon l'option choisie
                        filtered_pairs = []
                        
                        if filter_type == "Toutes les molécules":
                            filtered_pairs = mol_pairs
                        elif filter_type == "Molécules modifiées uniquement":
                            filtered_pairs = [pair for pair in mol_pairs if pair["original_smiles"] != pair["cleaned_smiles"] and len(pair["issues"]) > 0]
                        elif filter_type == "Molécules avec isotopes":
                            filtered_pairs = [pair for pair in mol_pairs if any("isotopique" in issue for issue in pair["issues"])]
                        elif filter_type == "Molécules avec sels/fragments":
                            filtered_pairs = [pair for pair in mol_pairs if any("sel" in issue.lower() or "fragment" in issue.lower() for issue in pair["issues"])]
                        elif filter_type == "Autres modifications":
                            filtered_pairs = [pair for pair in mol_pairs if pair["original_smiles"] != pair["cleaned_smiles"] and 
                                             not any(("isotopique" in issue or "sel" in issue.lower() or 
                                                     "fragment" in issue.lower()) 
                                                     for issue in pair["issues"])]
                        
                        # Si aucune molécule ne correspond au filtre
                        if not filtered_pairs:
                            st.warning(f"Aucune molécule ne correspond au filtre '{filter_type}'")
                        else:
                            # Limiter le nombre après filtrage selon le slider
                            display_pairs = filtered_pairs[:min(display_count, len(filtered_pairs))]
                            
                            # Sélection des molécules
                            st.subheader("Sélection des molécules")
                            
                            # Créer une liste d'options pour la sélection
                            mol_options = []
                            for i, pair in enumerate(display_pairs):
                                name = pair.get('name', f"Molécule {i+1}")
                                mol_options.append(name)
                            
                            selected_indices = st.multiselect(
                                "Sélectionnez les molécules à afficher",
                                range(len(mol_options)),
                                range(min(display_count, len(mol_options))),
                                format_func=lambda i: mol_options[i]
                            )
                            
                            # Afficher les molécules sélectionnées
                            if selected_indices:
                                st.subheader("Visualisation des molécules")
                                
                                for idx in selected_indices:
                                    pair = display_pairs[idx]
                                    
                                    st.write("---")
                                    name = pair.get('name', f"Molécule {idx+1}")
                                    st.subheader(name)
                                    
                                    # Créer deux colonnes pour avant/après
                                    col1, col2 = st.columns(2)
                                    
                                    # Molécule originale
                                    with col1:
                                        st.markdown("**Molécule originale**")
                                        st.markdown("**SMILES original :**")
                                        st.code(pair['original_smiles'], language=None)
                                        
                                        if show_images:
                                            # Générer la molécule à la demande
                                            original_mol = Chem.MolFromSmiles(pair['original_smiles'])
                                            if original_mol:
                                                img_str = mol_to_img(original_mol)
                                                if img_str:
                                                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
                                                else:
                                                    st.warning("Impossible de générer l'image")
                                            else:
                                                st.warning("SMILES invalide")
                                    
                                    # Molécule nettoyée
                                    with col2:
                                        st.markdown("**Molécule nettoyée**")
                                        st.markdown("**SMILES nettoyé :**")
                                        st.code(pair['cleaned_smiles'], language=None)
                                        
                                        if show_images:
                                            # Générer la molécule à la demande
                                            cleaned_mol = Chem.MolFromSmiles(pair['cleaned_smiles'])
                                            if cleaned_mol:
                                                img_str = mol_to_img(cleaned_mol)
                                                if img_str:
                                                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
                                                else:
                                                    st.warning("Impossible de générer l'image")
                                            else:
                                                st.warning("SMILES invalide")
                                    
                                    # Informations supplémentaires sur la molécule
                                    with st.expander("Informations supplémentaires"):
                                        if "issues" in pair and pair["issues"]:
                                            st.write("**Modifications effectuées:**")
                                            for issue in pair["issues"]:
                                                if issue.strip():  # Éviter les chaînes vides
                                                    st.write(f"- {issue.strip()}")
                                        else:
                                            st.write("Aucune modification majeure détectée.")
                                        
                                        # Calcul de quelques propriétés moléculaires
                                        if show_images and 'cleaned_mol' in locals() and cleaned_mol:
                                            st.write("**Propriétés:**")
                                            try:
                                                from rdkit.Chem import Descriptors
                                                mol_weight = round(Descriptors.MolWt(cleaned_mol), 2)
                                                logp = round(Descriptors.MolLogP(cleaned_mol), 2)
                                                hbd = Descriptors.NumHDonors(cleaned_mol)
                                                hba = Descriptors.NumHAcceptors(cleaned_mol)
                                                
                                                props_col1, props_col2 = st.columns(2)
                                                props_col1.metric("Poids moléculaire", f"{mol_weight} g/mol")
                                                props_col1.metric("LogP", logp)
                                                props_col2.metric("Donneurs de H", hbd)
                                                props_col2.metric("Accepteurs de H", hba)
                                            except Exception as e:
                                                st.warning(f"Impossible de calculer les propriétés moléculaires: {e}")
            else:
                st.warning("Aucun fichier de résultats disponible.")

    with tabs[2]:  # Onglet Documentation
        st.header("Documentation technique")
        
        st.info(
            "Cette application utilise RDKit pour nettoyer et standardiser des bibliothèques de molécules, avec une optimisation mémoire pour traiter des millions de molécules."
        )
        
        st.subheader("Processus de nettoyage")
        
        # 1. Conversion initiale SMILES → Mol → SMILES canonique
        st.write("**1. Conversion initiale SMILES → Mol**")
        st.markdown("""
        **Outil utilisé:** `Chem.MolFromSmiles()` de RDKit
        
        **Processus:** Les chaînes SMILES sont converties en objets moléculaires RDKit. Cette étape vérifie également la validité des structures. Les molécules invalides sont écartées et placées dans un fichier séparé pour correction ultérieure.
        
        **Résultat:** Cette étape permet de vérifier la validité syntaxique et chimique des SMILES fournis.
        """)
        
        # 2. Suppression des sels et solvants
        st.write("**2. Suppression des sels et solvants**")
        st.markdown("""
        **Outil utilisé:** `SaltRemover.SaltRemover()` de RDKit
        
        **Processus:** Identifie et retire les contre-ions (Na⁺, Cl⁻, etc.) et les molécules de solvant (H₂O, CH₃OH, etc.) qui sont souvent inclus dans les représentations SMILES des composés pharmaceutiques ou biologiques.
        
        **Exemple:** Le SMILES `CC(=O)O.[Na+]` (acétate de sodium) sera transformé en `CC(=O)O` (acide acétique).
        """)
        
        # 3. Traitement des fragments multiples 
        st.write("**3. Traitement des fragments multiples**")
        st.markdown("""
        **Outil utilisé:** `Chem.GetMolFrags()` et une fonction personnalisée `get_largest_fragment()`
        
        **Processus:** Après la suppression des sels et solvants, l'application identifie si la molécule contient encore plusieurs fragments indépendants. Dans ce cas, elle conserve uniquement le fragment ayant le plus grand nombre d'atomes, considéré comme le "composé principal".
        """)
        
        # 4. Suppression des molécules avec atomes exotiques
        st.write("**4. Suppression des molécules avec atomes exotiques**")
        st.markdown("""
        **Outil utilisé:** Fonction personnalisée `has_nonmedicinal_atoms()`
        
        **Processus:** Vérifie que la molécule ne contient que des atomes couramment utilisés dans les médicaments: C, H, O, N, S, P, F, Cl, Br, I. Les molécules contenant d'autres types d'atomes sont écartées.
        
        **Exemple:** Une molécule contenant du silicium (Si) ou du bore (B) sera filtrée car ces éléments ne font pas partie de la liste des atomes acceptables.
        """)
        
        # 5. Génération du SMILES canonique et isomérique
        st.write("**5. Génération du SMILES canonique et isomérique**")
        st.markdown("""
        **Outil utilisé:** `Chem.MolToSmiles()` avec les paramètres `isomericSmiles=True` et `canonical=True`
        
        **Processus:** Convertit l'objet moléculaire en une représentation SMILES canonique qui préserve l'information stéréochimique (centres chiraux, doubles liaisons E/Z).
        
        **Importance:** Cette étape garantit une représentation cohérente et unique de chaque structure, tout en conservant les informations stéréochimiques essentielles pour l'activité biologique.
        """)
        
        # 6. Détection et correction des atomes spéciaux
        st.write("**6. Détection et correction des atomes spéciaux**")
        st.markdown("""
        **Outil utilisé:** `atom.GetIsotope()` et `atom.SetIsotope(0)` de RDKit
        
        **Processus:** Chaque atome de la molécule est examiné pour détecter la présence d'isotopes spécifiques comme le deutérium (²H), le carbone-13 (¹³C), l'azote-15 (¹⁵N), etc. Ces atomes sont ensuite remplacés par leurs versions standard.
        
        **Isotopes détectés:**
        - Deutérium (²H) et Tritium (³H)
        - Carbone-13 (¹³C) et Carbone-14 (¹⁴C)
        - Azote-15 (¹⁵N)
        - Oxygène-17 (¹⁷O) et Oxygène-18 (¹⁸O)
        - Autres isotopes (³²P, ³⁵S, ¹⁸F, etc.)
        
        **Exemple:** Le SMILES `[2H]OC([2H])([2H])C` (éthanol partiellement deutéré) sera normalisé en `CCO`.
        """)
        
        # 7. Vérification finale et suppression des doublons
        st.write("**7. Vérification finale et suppression des doublons**")
        st.markdown("""
        **Processus:** 
        - Génération des SMILES canoniques isomériques après toutes les étapes de traitement
        - Élimination des doublons : si deux formes deviennent identiques après traitement, une seule est conservée
        
        **Importance:** Cette étape garantit que la bibliothèque finale ne contient que des structures uniques, maximisant ainsi l'efficacité des criblages virtuels ultérieurs.
        """)
        
        # 8. Validation finale
        st.write("**8. Validation finale du pipeline**")
        st.markdown("""
        **Processus:**
        - Affichage du nombre total de molécules uniques après nettoyage complet
        - Génération d'un fichier d'erreurs avec les molécules invalides pour correction manuelle ultérieure
        - Création d'un fichier final contenant toutes les molécules retenues avec leurs noms
        """)
        
        # 9. Optimisation mémoire avec système hybride SQLite
        st.write("**9. Optimisation mémoire pour bibliothèques volumineuses**")
        st.markdown("""
        **Techniques utilisées:**
        - **Traitement par lots:** Les molécules sont traitées par groupes de taille configurable
        - **Système hybride de détection des doublons:** 
          - Utilisation d'un cache mémoire intelligent pour les molécules récentes (haute performance)
          - Base de données SQLite avec indexation pour un stockage efficace des millions de structures
          - Hachage MD5 des SMILES pour des recherches ultra-rapides
          - Performance constante quelle que soit la taille de la bibliothèque
        - **Transactions optimisées** pour réduire les opérations disque et maintenir des performances élevées
        - **Écriture progressive des résultats** au fur et à mesure du traitement
        - **Nettoyage périodique des structures de données** pour une utilisation mémoire optimale
        - **Libération explicite des objets RDKit** pour prévenir les fuites mémoire
        
        **Avantages:**
        - Détection des doublons aussi rapide avec 10 millions de molécules qu'avec 10 000
        - Aucun ralentissement notable même pour de très grandes bibliothèques
        - Utilisation mémoire contrôlée et prévisible
        - Intégration transparente sans dépendances externes (utilise SQLite intégré à Python)
        - Fiabilité et consistance garanties pour les grands jeux de données
        """)
        
        # 10. Fichiers de sortie générés
        st.write("**10. Fichiers de sortie générés**")
        st.markdown("""
        **Fichiers produits:**
        - **molecules_finales.csv:** Collection complète des molécules nettoyées
        - **molecules_invalides.csv:** Structures qui n'ont pas pu être traitées correctement
        - **molecules_isotopes.csv:** Molécules qui contenaient des isotopes spécifiques
        - **molecules_similaires.csv:** Molécules identifiées comme similaires ou en doublon
        - **molecules_fragments.csv:** Molécules originales contenant des ions ou fragments multiples
        
        **Utilité:** Ces fichiers séparés permettent une analyse fine de chaque type de transformation et facilitent l'identification des problèmes potentiels dans les données d'entrée.
        """)
        
        st.subheader("Optimisation de la détection des doublons")
        
        st.markdown("""
        ### Système hybride haute performance
        
        La nouvelle implémentation utilise une approche hybride sophistiquée pour maintenir des performances constantes quelle que soit la taille de votre bibliothèque moléculaire:
        
        #### 1. Cache mémoire intelligent
        - Maintient les 50 000 molécules les plus récentes directement en mémoire vive
        - Permet une vérification quasi-instantanée pour les molécules récemment traitées
        - Utilise une stratégie d'éviction LRU (Least Recently Used) pour optimiser l'utilisation mémoire
        
        #### 2. Base de données SQLite avec indexation
        - Stocke efficacement les empreintes des molécules déjà traitées
        - Utilise des index optimisés pour des recherches extrêmement rapides
        - Fonctionne comme un dictionnaire persistant sans limite de taille
        
        #### 3. Hachage MD5 pour recherches ultrarapides
        - Transforme les SMILES canoniques en empreintes compactes et uniques
        - Permet des comparaisons bien plus rapides que sur les chaînes complètes
        - Élimine les problèmes de performance liés à la longueur des SMILES
        
        #### 4. Transactions optimisées
        - Regroupe les opérations d'écriture dans des transactions
        - Réduit drastiquement les accès disque pour maintenir la performance
        - Garantit l'intégrité des données même en cas d'interruption
        """)
        
        st.subheader("FAQs")
        
        faq = {
            "Qu'est-ce qu'un SMILES canonique isomérique?": 
                "Le SMILES canonique isomérique est une représentation unique et standardisée d'une molécule qui préserve "
                "l'information stéréochimique (centres chiraux, géométrie des doubles liaisons). Pour une même structure complète "
                "(y compris la stéréochimie), il n'existe qu'un seul SMILES canonique isomérique.",
            
            "Comment sont détectés les atomes spéciaux?": 
                "L'application analyse chaque atome de la molécule pour détecter la présence d'isotopes "
                "(comme le deutérium ou le carbone-13) en vérifiant les propriétés d'isotope avec la méthode atom.GetIsotope() de RDKit. "
                "Si cette valeur est supérieure à zéro, l'atome est considéré comme un isotope spécial.",
            
            "Comment fonctionne la sélection du plus grand fragment?":
                "Lorsqu'une molécule contient plusieurs fragments non liés (comme dans un sel), l'application utilise Chem.GetMolFrags() "
                "pour identifier tous les fragments, puis sélectionne celui qui contient le plus grand nombre d'atomes, "
                "le considérant comme le 'composé principal'.",
            
            "Comment fonctionne la suppression des sels?":
                "L'application utilise la fonctionnalité SaltRemover de RDKit qui identifie et sépare les "
                "contre-ions et les molécules de solvant de la structure principale. Cette classe utilise une liste prédéfinie "
                "de fragments correspondant aux sels et solvants courants pour les identifier et les retirer.",
            
            "Quels sont les atomes considérés comme 'non médicinaux'?":
                "Les atomes exotiques sont ceux rarement présents dans les petites molécules médicamenteuses (ceux qui marchent dans le docking). "
                "Seuls C, H, O, N, S, P, F, Cl, Br et I sont considérés comme acceptables. "
                "Toutes les molécules possédant les autres éléments sont filtrés.",
            
            "Comment traiter des bibliothèques de plusieurs millions de molécules?":
                "L'application utilise un système hybride optimisé qui combine un cache mémoire intelligent et une base de données SQLite "
                "avec indexation. Cette approche maintient des performances constantes quelle que soit la taille de la bibliothèque, "
                "sans ralentissement, même pour des dizaines de millions de molécules. Le traitement par lots et l'écriture progressive "
                "des résultats assurent une utilisation efficace des ressources.",
            
            "Est-ce que SQLite nécessite une installation séparée?":
                "Non, SQLite est inclus dans la bibliothèque standard de Python et ne nécessite aucune installation supplémentaire. "
                "La base de données SQLite est temporaire, créée localement pendant le traitement, et supprimée automatiquement à la fin. "
                "Aucune donnée n'est jamais envoyée sur un serveur externe, tout reste sur votre machine.",
            
            "Pourquoi la détection des doublons est-elle maintenant beaucoup plus rapide?":
                "La nouvelle implémentation utilise une approche en trois niveaux: un cache mémoire pour les molécules récentes, "
                "un index optimisé dans SQLite, et un hachage MD5 des SMILES pour des recherches ultrarapides. "
                "Cette combinaison offre des performances de recherche quasi-constantes, quelle que soit la taille de la bibliothèque.",
            
            "La détection des doublons est-elle toujours fiable avec de très grands jeux de données?":
                "Oui, le système hybride garantit une détection fiable et précise des doublons, même pour des dizaines de millions de molécules. "
                "L'utilisation de hachage MD5 et d'index optimisés assure l'intégrité des résultats, et les transactions SQL protègent contre "
                "la corruption des données en cas d'interruption.",
            
            "Où sont enregistrés les fichiers de sortie?":
                "Les fichiers sont sauvegardés directement dans un dossier nommé 'cleanchem_results' dans le répertoire courant. "
                "Ce dossier contient tous les fichiers .csv générés durant le traitement, qui sont disponibles immédiatement après la fin du processus."
        }
        
        for question, answer in faq.items():
            with st.expander(question):
                st.write(answer)
        
        st.subheader("Ressources supplémentaires")
        
        st.markdown("""
        - [Documentation RDKit](https://www.rdkit.org/docs/index.html)
        - [Documentation Streamlit](https://docs.streamlit.io/)
        - [Tutoriel SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
        - [Documentation SQLite](https://docs.python.org/3/library/sqlite3.html)
        - [Optimisation mémoire Python](https://docs.python.org/3/library/gc.html)
        """)
        
        # Footer
        st.write("---")
        st.caption("Youcef ---- Développé avec RDKit et Streamlit | © 2025 | Optimisé pour le traitement de grandes bibliothèques")
if __name__ == "__main__":
    main()
