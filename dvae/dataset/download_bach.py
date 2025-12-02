import music21
import os
import time

def extract_bach_to_midi(output_folder="C:\\code\\dvae_final\\DVAE\\data\\bach_midis", limit=None):
    """
    Extrait le corpus Bach de music21 et le sauvegarde en fichiers MIDI.
    """
    # 1. Créer le dossier de destination
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier créé : {output_folder}")

    # 2. Récupérer la liste des fichiers Bach
    print("Recherche des fichiers Bach dans music21...")
    all_bach_paths = music21.corpus.getComposer('bach')
    
    # On filtre pour ne garder que les Chorals (BWV) qui sont propres
    # Les chorals sont parfaits pour les DVAE car homophoniques (toutes les voix bougent ensemble)
    bach_paths = [p for p in all_bach_paths if 'bwv' in str(p)]
    
    if limit:
        bach_paths = bach_paths[:limit]

    print(f"{len(bach_paths)} partitions trouvées. Début de la conversion en MIDI...")

    count = 0
    start_time = time.time()

    for i, path in enumerate(bach_paths):
        try:
            # Nom de fichier propre
            filename = os.path.basename(str(path))
            filename = os.path.splitext(filename)[0] + ".mid"
            save_path = os.path.join(output_folder, filename)

            # Si le fichier existe déjà, on passe (gain de temps si tu relances)
            if os.path.exists(save_path):
                continue

            # Parsing et Conversion
            # C'est l'étape longue : lire le XML et écrire le MIDI
            score = music21.corpus.parse(path)
            score.write('midi', fp=save_path)
            
            count += 1
            if count % 10 == 0:
                print(f"[{count}] Converti : {filename}")

        except Exception as e:
            print(f"Erreur sur {path}: {e}")
            continue

    print(f"\nTerminé ! {count} fichiers MIDI sauvegardés dans '{output_folder}'.")
    print(f"Temps total : {time.time() - start_time:.2f} secondes")

if __name__ == "__main__":
    # Assure-toi d'avoir fait 'pip install music21' avant
    extract_bach_to_midi()