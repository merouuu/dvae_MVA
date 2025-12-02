from music_dataset import create_midi_dataset_file

# Configuration
SOURCE_FOLDER = "C:\\code\\dvae_final\\DVAE\\data\\bach_midis"      # Le dossier créé à l'étape 1
OUTPUT_FILE = "C:\\code\\dvae_final\\DVAE\\data\\bach_data.npz"     # Le fichier que ton VRNN va charger
FS = 16                             # 16 frames par seconde

# Lancement
if __name__ == "__main__":
    create_midi_dataset_file(SOURCE_FOLDER, OUTPUT_FILE, fs=FS)