import numpy as np
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_episode(file_path, target, select=None):
    """
    Legge un file di log degli episodi, seleziona casualmente un episodio e produce:
      - Un grafico 3D della traiettoria dell'agent (usando un gradiente di colore, con marker per start, end e target)
      - Un grafico con due subplot: distanza dal target e reward in funzione dello step.
      
    :param file_path: path del file di testo con i log degli episodi.
    :param target: lista o array di 3 elementi [x, y, z] che rappresenta il target.
    """
    # Leggi tutte le linee del file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    episodes = {}
    current_episode = None

    # Raggruppa le linee per episodio
    for line in lines:
        line = line.strip()
        if line.startswith("Episode"):
            # Es: "Episode 0" -> estrai il numero
            parts = line.split()
            ep_num = int(parts[1])
            current_episode = ep_num
            episodes[current_episode] = []
        elif line and current_episode is not None:
            # Aspettando righe come:
            # 1) State: [0.0, 0.0, 0.0, 1.0, 0.0] | Action: UP | Q_value: None | Reward: 10.0
            try:
                # Separa il numero dello step
                step_str, rest = line.split(")", 1)
                step_num = int(step_str.strip())
                # Dividi la parte restante in campi usando il separatore "|"
                parts = [p.strip() for p in rest.split("|")]
                step_data = {"step": step_num}
                for part in parts:
                    if part.startswith("State:"):
                        state_str = part[len("State:"):].strip().strip("[]")
                        # Converte in lista di float
                        state_vals = [float(x) for x in state_str.split(",")]
                        step_data["state"] = state_vals
                    elif part.startswith("Action:"):
                        step_data["action"] = part[len("Action:"):].strip()
                    elif part.startswith("Q_value:"):
                        # Manteniamo il valore come stringa (es. "None" o un numero)
                        step_data["q_value"] = part[len("Q_value:"):].strip()
                    elif part.startswith("Reward:"):
                        step_data["reward"] = float(part[len("Reward:"):].strip())
                episodes[current_episode].append(step_data)
            except Exception as e:
                print("Errore nel parsing della riga:", line, e)

    if not episodes:
        print("Nessun episodio trovato nel file.")
        return

    # Scelta episodio
    if select==None :
        selected_ep = random.choice(list(episodes.keys()))
        steps_data = episodes[selected_ep]
        print(f"Episodio randomico selezionato: {selected_ep} (step: {len(steps_data)})")
    else:
        selected_ep = select
        steps_data = episodes[selected_ep]
        print(f"Episodio {selected_ep} selezionato (step: {len(steps_data)})")

    # Estrai i dati: stato, reward, step (per il grafico)
    states = np.array([d["state"] for d in steps_data])  # supponiamo che i primi 3 valori siano x, y, z
    rewards = np.array([d["reward"] for d in steps_data])
    step_numbers = np.array([d["step"] for d in steps_data])
    
    # Calcola la distanza dal target per ogni step (usando i primi 3 valori dello state)
    target = np.array(target)
    distances = np.linalg.norm(states[:, :3] - target, axis=1)

    # --- Grafico 3D della traiettoria ---
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Crea segmenti per una linea 3D colorata
    points = states[:, :3]
    segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
    
    # Utilizza un colormap (qui 'viridis') per il gradiente in base all'indice dello step
    lc = Line3DCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(segments)))
    lc.set_array(np.linspace(0, len(segments), len(segments)))
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    
    # Evidenzia il punto di inizio (verde) e di arrivo (rosso)
    ax1.scatter(points[0, 0], points[0, 1], points[0, 2], color='green', s=50, label='Start')
    ax1.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='red', s=50, label='End')
    # Plot del target (blu, a stella)
    ax1.scatter(target[0], target[1], target[2], color='blue', s=100, marker='*', label='Target')
    
    ax1.set_title(f"Episodio {selected_ep} - Traiettoria 3D")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # --- Grafico dei trend: Distanza e Reward vs Step ---
    ax2 = fig.add_subplot(222)
    ax2.plot(step_numbers, distances, marker='o', label='Distanza', color='magenta')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distanza dal Target")
    ax2.set_title("Distanza vs Step")
    
    ax3 = fig.add_subplot(224)
    ax3.plot(step_numbers, rewards, marker='o', label='Reward', color='orange')
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward vs Step")
    
    plt.tight_layout()
    plt.show()


# TODO: Permetti inserimento da CLI del nome file e di quali episodi ottenere i plot
if __name__ == "__main__":
    file_dir = "logs"
    file_name = "training_log_2025-04-07_14-00-21.txt" # INSERIRE QUI NOME FILE PRIMA DI ESEGUIRE
    file_path = os.path.join(file_dir, file_name)
    target = [10.0, 10.0, 5.0]
    plot_episode(file_path, target)
    plot_episode(file_path, target)
    plot_episode(file_path, target, select=250)
    plot_episode(file_path, target, select=450)