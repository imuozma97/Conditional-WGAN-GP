"""
Aquí pondré funciones de plots útiles. De momento solo tengo la de las gráficas de losses durante el entrenamiento
"""
import matplotlib.pyplot as plt
import os

def plot_loss_graph(epoch_vect, loss_1, loss_2, filename, label1, label2, generated_images_folder):
    fig, ax = plt.subplots()
    ax.plot(epoch_vect, loss_1, 'g-', linewidth=1, markersize=3, label = label1)
        
    if loss_2 is not None:
        ax.plot(epoch_vect, loss_2, 'b-', linewidth=1, markersize=3, label = label2)
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    ax.legend()
    plt.savefig(os.path.join(generated_images_folder, filename), bbox_inches='tight', format='pdf')
    plt.close()   
        