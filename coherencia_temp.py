import numpy as np
from config import num_classes, num_cv

def coherence(data_real, data_fake, image_size, N):

    # 27 evoluciones × 34 redshifts
    data_real = np.reshape(data_real, (num_cv, num_classes, image_size, image_size, image_size))
    data_fake = np.reshape(data_fake, (N, num_classes, image_size, image_size, image_size))

    coherence_mean_real, coherence_mean_fake = [], []
    coherence_std_real, coherence_std_fake = [], []

    for z in range(num_classes -1):

        correlations_real = []
        correlations_fake = []

        for evo in range(num_cv):

            # Cubos de dos redshifts consecutivos
            cube_1_real = data_real[evo, z]
            cube_2_real = data_real[evo, z + 1]
            cube_1_fake = data_fake[evo, z]
            cube_2_fake = data_fake[evo, z + 1]

            # Aplanar los cubos
            x_real = cube_1_real.flatten()
            y_real = cube_2_real.flatten()
            x_fake = cube_1_fake.flatten()
            y_fake = cube_2_fake.flatten()

            # Correlación de Pearson
            corr_real = np.corrcoef(x_real, y_real)[0, 1]
            corr_fake = np.corrcoef(x_fake, y_fake)[0, 1]

            correlations_real.append(corr_real)
            correlations_fake.append(corr_fake)

        # Estadística sobre las 27 evoluciones
        coherence_mean_real.append(np.mean(correlations_real))
        coherence_std_real.append(np.std(correlations_real))
        coherence_mean_fake.append(np.mean(correlations_fake))
        coherence_std_fake.append(np.std(correlations_fake))



    coherence_mean_real = np.array(coherence_mean_real)
    coherence_mean_fake = np.array(coherence_mean_fake)

    eps = 1e-12

    residual_coherence = ((coherence_mean_fake - coherence_mean_real) / (coherence_mean_real + eps))

    error_coherence = np.mean(np.abs(residual_coherence))

    print("Residual coherencia:", residual_coherence)
    print("Error coherencia total:", error_coherence)


