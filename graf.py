import json
import matplotlib.pyplot as plt
import os

# Ruta del archivo JSON
json_file = "Training3D/6-models/loss_data.json"

# Leer los datos
with open(json_file, "r") as f:
    data = json.load(f)

epochs = data["epoch_vect"][:4000]
disc_loss_fake = data["disc_losses_f"][:4000]
disc_loss_real = data["disc_losses_r"][:4000]

# Crear la figura
plt.figure(figsize=(6,4))

plt.plot(epochs, disc_loss_fake, linewidth=1, color = 'blue', label="C(G(x, y))")
plt.plot(epochs, disc_loss_real, linewidth=1, color = 'green', label="C(x, y)")

plt.xlabel("Epoch")
plt.ylabel("Predictions")
plt.legend()

plt.tight_layout()

# Crear la carpeta si no existe
os.makedirs("Figuras", exist_ok=True)

# Guardar en PDF (recomendado para Overleaf)
plt.savefig("Figuras/discriminator_loss2.pdf", bbox_inches="tight")

# Opcionalmente también en PNG
plt.savefig("Figuras/discriminator_loss2.png", dpi=600, bbox_inches="tight")

plt.show()