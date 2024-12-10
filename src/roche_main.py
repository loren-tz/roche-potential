import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter


# Funzione per raccogliere input dall'utente con valori di default
def get_input(prompt, default):
    user_input = input(f"{prompt} (default={default}): ").strip()
    return float(user_input) if user_input else default

# Prompt iniziale per scegliere la modalità
debug_mode = input("Vuoi procedere in modalità debug? (y/n, default=n): ").strip().lower() in ['s', 'si', 'y', 'yes']

# Inizializzazione dei parametri
if debug_mode:
    print("Modalità debug attiva. Si utilizzano i parametri di default.")
    parameters = [1.0, 1.0, 3.5, 0.1, 50]  # Default: M1=1.0, M2=1.0, L=3.5, R=0.1, N=50
else:
    print("Modalità normale. Inserire i parametri.")
    parameters = []
    parameters.append(get_input("Inserire la massa M1", 1.0)) # M1
    parameters.append(get_input("Inserire la massa M2", 1.0)) # M2
    parameters.append(get_input("Inserire il valore di L", 3.5))  # L
    parameters.append(get_input("Inserire il valore di R", 0.1))  # R
    parameters.append(get_input("Inserire il valore di N", 50))  # N

cutoff = -parameters[2] * 30  # Potenziale di cutoff

# Costruzione dei parametri da passare al programma C
args = [str(parameters[0]), str(parameters[1]), str(parameters[2]), str(parameters[3]), str(cutoff)]

# Esecuzione dello script C
try:
    result = subprocess.run(
        ["./roche_sim.exe"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    print("Simulazione completata con successo.\nElaborazione dei dati in corso...")
except subprocess.CalledProcessError as e:
    print("Errore durante l'esecuzione di roche_sim.exe:")
    print(e.stderr)
    exit()

# Lettura dei dati generati
file_path = "roche_data.dat"
try:
    data = np.loadtxt(file_path)  # Carica il file dati
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
except Exception as e:
    print(f"Errore durante la lettura del file {file_path}:")
    print(e)
    exit()

# Mappatura su griglia 100x100
x_unique = np.linspace(np.min(x), np.max(x), int(parameters[4]))
y_unique = np.linspace(np.min(y), np.max(y), int(parameters[4]))
X, Y = np.meshgrid(x_unique, y_unique)
Z = griddata((x, y), z, (X, Y), method='cubic')

# Maschera dei valori NaN
nan_mask = np.isnan(Z)

# Sostituzione temporanea dei NaN con 0 per il filtro
Z_temp = np.nan_to_num(Z, nan=0)

# Applicazione del filtro gaussiano
sigma = 3  # Incrementa per coprire zone più ampie
Z_filtered = gaussian_filter(Z_temp, sigma=sigma)

# Calcolo dei pesi usando la maschera invertita
weights = gaussian_filter(np.logical_not(nan_mask).astype(float), sigma=sigma)

# Riempimento dei NaN con il risultato pesato
Z[nan_mask] = Z_filtered[nan_mask] / np.maximum(weights[nan_mask], 1e-10)

# Maschera dei valori validi
valid_mask = ~np.isnan(Z)
x_valid = X[valid_mask]
y_valid = Y[valid_mask]
z_valid = Z[valid_mask]

# Interpolazione con RBF
rbf = Rbf(x_valid, y_valid, z_valid, function='multiquadric', smooth=1)
Z_rbf = rbf(X, Y)

# Sostituzione dei NaN con RBF
nan_mask = np.isnan(Z)
Z[nan_mask] = Z_rbf[nan_mask]

# Applicazione di un filtro gaussiano per smussare le transizioni
sigma_smooth = 3
Z = gaussian_filter(Z, sigma=sigma_smooth)


# Creazione del grafico
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot della superficie
surf = ax.plot_surface(X, Y, Z - np.min(Z) / 4, cmap="YlOrBr_r", alpha=0.8, edgecolor="black", linewidth=0.1)

# Proiezione della superficie sul piano XY
ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='Greys_r', alpha=1)

# Configurazione del grafico
ax.set_xticks([])  # Rimuove i ticks dell'asse X
ax.set_xlabel('')  # Rimuove l'etichetta dell'asse X
ax.set_yticks([])  # Rimuove i ticks dell'asse Y
ax.set_ylabel('')  # Rimuove l'etichetta dell'asse Y
ax.set_zticks([])  # Rimuove i ticks dell'asse Z
ax.set_zlabel('')  # Rimuove l'etichetta dell'asse Z
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Colore trasparente (RGBA)
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Colore trasparente (RGBA)
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Colore trasparente (RGBA)
ax.grid(False)  # Rimuove la griglia
fig.patch.set_facecolor('white')  # Sfondo bianco
ax.set_facecolor('white')  # Sfondo degli assi bianco
ax.xaxis.pane.set_visible(False)  # Nasconde il piano
ax.yaxis.pane.set_visible(False)  # Nasconde il piano
ax.zaxis.pane.set_visible(False)  # Nasconde il piano

# Aggiunta della colorbar
cbar = fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.01)  # Configura dimensioni della colorbar
cbar.set_ticks([])  # Rimuove i numeri
cbar.set_label(r"$V_{\mathrm{R}}(x, y)$", fontsize=12, labelpad=30, rotation=0)  # Label in stile matematico

ax.set_title(r"Roche potential for $M_1$, $M_2$", fontsize=12, pad=0)

# Funzione per aggiornare la vista ad ogni frame
def update(frame):
    ax.view_init(elev=30, azim=frame)  # Cambia l'angolo di azimuth
    return surf,

# Creazione dell'animazione
frames = 360  # Numero di frame (un giro completo = 360 gradi)
fps = 60
ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)


# Mostra il grafico
plt.show()
