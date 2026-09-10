"""
Funciones principales del entrenamiento
-train step: hace cada batch
"""

import tensorflow as tf
import os
import json
import numpy as np


from power import Power
from architectures.downsampling import Downsample
from psd_utils import psd_loss_log
from loss_plot import plot_loss_graph
from transforms import backward_2
from config import n_bar_64, n_bar_128


class Training_upsc(tf.keras.Model):

    def __init__(self, generator, batch_size, trained_models_folder, generated_images_folder, image_size):
        super().__init__()
        self.generator = generator
        self.batch_size = batch_size
        self.trained_models_folder= trained_models_folder
        self.generated_images_folder = generated_images_folder
        self.current_epoch = 0
        self.image_size = image_size

        self.power = Power(self.image_size)
        self.down = Downsample(n_bar_64, n_bar_128)


    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer


    #@tf.function    
    def train_step(self, data):
            
        real_delta_64, z_values, real_psd_128= data  
        print("real_images", real_delta_64.shape)

        with tf.GradientTape(persistent = True) as gen_tape:
                
            # 1-Genero imágenes de 128 a partir de las de 64; está generado tal cual tenga el de 64
            fake_delta_128 = self.generator([real_delta_64, z_values], training=True)
            print("Generated images", fake_delta_128.shape)

            # 2-Saco delta para poder calular el psd del 128 generado, y a que el psd real de 128 es de delta.
            fake_psd_128 = self.power.compute_all_psd(fake_delta_128) 

            # 3-Loss del psd: comparación del psd_real_128 y psd_gen_128
            loss_psd_128 = psd_loss_log(fake_psd_128, real_psd_128) 

            # 4-Loss de cubos: comparación cubo 64 real con cubo 64 reducido desde 128 fake. Primero calculo el cubo de 64 a partir del de 128
            fake_delta_64 = self.down.downsample_128(fake_delta_128)
            loss_delta_64 = tf.reduce_mean(tf.square(real_delta_64 - fake_delta_64))


            # 5-Loss total
            loss_total = lambda_delta * loss_delta_64 + lambda_psd * loss_psd_128


        grads_gen = gen_tape.gradient(loss_total, self.generator.trainable_variables)
        grads_psd = gen_tape.gradient(loss_psd_128, self.generator.trainable_variables)
        grads_delta = gen_tape.gradient(loss_delta_64, self.generator.trainable_variables)

        norm_gen = tf.linalg.global_norm(grads_gen)   
        norm_psd = tf.linalg.global_norm(grads_psd)
        norm_delta = tf.linalg.global_norm(grads_delta)

        self.g_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))


        return loss_delta_64, loss_psd_128, loss_total, norm_gen, norm_psd, norm_delta
        
    



    def train(self, dataset_train,  epochs):

        # Configuración de checkpoints
        checkpoint_dir = os.path.join(self.trained_models_folder, "checkpoints")
        checkpoint = tf.train.Checkpoint(generator = self.generator, discriminator = self.discriminator, g_optimizer = self.g_optimizer, d_optimizer = self.d_optimizer, epoch = tf.Variable(0))
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep = 5)    
        
        loss_file = os.path.join(self.trained_models_folder, "loss_data.json")

        # Restaurar si existe un checkpoint previo
        if checkpoint_manager.latest_checkpoint:
            print(f"Restaurando desde {checkpoint_manager.latest_checkpoint}")
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            start_epoch = int(checkpoint.epoch.numpy())  # Recuperar la última época guardada

            if os.path.exists(loss_file):
                print("Cargando histórico de pérdidas...")
                with open(loss_file, 'r') as f:
                    data = json.load(f)

                epoch_vect = data.get("epoch_vect", [])
                losses_delta_64 = data.get('losses_delta_64', [])
                losses_psd_128 = data.get('losses_psd_128', [])
                losses_total = data.get('losses_total', [])
                norms_gen = data.get('norms_gen', [])
                norms_psd = data.get('norms_psd', [])
                norms_delta = data.get('norms_delta', [])

                best_epoch = data.get('best_epoch', [])
                best_psd = data.get('best_psd', [])
                best_epoch_psd = data.get('best_epoch_psd', [])

                best_psd_metric = best_psd[-1]


        else:
            print("No se encontraron checkpoints previos, iniciando desde cero.")

            start_epoch = 0
            epoch_vect = []
            losses_total, losses_psd, losses_delta = [], [], []
            norms_gen, norms_psd, norms_delta= [], [], []

            best_psd_metric = float("inf")
            best_psd, best_epoch_psd = [], []
        
            
        for epoch in range(start_epoch, epochs):
                
            self.current_epoch = epoch
            batch_count = 0
            total_loss, psd_loss, delta_loss = 0, 0, 0
            norm_gen, norm_psd, norm_delta = 0, 0, 0
            
            print('Currently training on epoch {} (out of {}).'.format(epoch, epochs))

            for image_batch in dataset_train:
                losses = self.train_step(image_batch)

                delta_loss += -losses[0]
                psd_loss += losses[1]
                total_loss += losses[2]
                norm_gen += losses[3]
                norm_psd += losses[4]
                norm_delta += losses[5]
                    
                batch_count += 1
                    

            delta_loss /= batch_count
            psd_loss /= batch_count
            total_loss /= batch_count
            norm_gen /= batch_count
            norm_psd /= batch_count
            norm_delta /= batch_count
                

            if epoch > 1 and psd_loss < best_psd_metric:
                best_psd_metric = psd_loss

                gen_path = os.path.join(self.trained_models_folder, "best_psd_generator", f"epoch_{epoch:05d}")
                os.makedirs(gen_path, exist_ok=True)
                self.generator.save(gen_path)

                best_psd.append(float(psd_loss.numpy()))
                best_epoch_psd.append(epoch)

                #np.savez(os.path.join(gen_path, f"psd_data_{epoch:05d}.npz"),
                 #   psd_gen = psd_gen_batch.numpy(),
                  #  psd_min = psd_min_batch.numpy(),
                  #  psd_max = psd_max_batch.numpy(),
                  #  percent = float(percent_batch.numpy())
                #)

                print(f"Best psd Guardado en época {epoch}")

            
        
            if epoch % 150 == 0:
                print(f"Guardando modelo por estabilización después de 150 épocas.")

                gen_path = os.path.join(self.trained_models_folder, "generator_stable", f"epoch_{epoch:05d}")
                os.makedirs(gen_path, exist_ok=True)
                self.generator.save(gen_path)
                    


            losses_total.append(float(total_loss.numpy()))
            losses_psd.append(float(psd_loss.numpy()))
            losses_delta.append(float(delta_loss.numpy()))
                
            norms_gen.append(float(norm_gen.numpy()))
            norms_psd.append(float(norm_psd.numpy()))
            norms_delta.append(float(norm_delta.numpy()))

            epoch_vect.append(epoch)

            checkpoint.epoch.assign(epoch)
            checkpoint_manager.save()


            # Guardamos pérdidas en archivo
            tmp_file = loss_file + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump({
                        'epoch_vect' : epoch_vect,
                        'gen_losses': losses_total,
                        'psd_losses': losses_psd,
                        'delta_losses' : losses_delta,
                        'best_psd' : best_psd, 
                        'best_epoch_psd' : best_epoch_psd,
                        'norm_gen' : norms_gen, 
                        'norm_psd' : norms_psd, 
                        'norm_delta' : norms_delta
                    }, f)
            os.replace(tmp_file, loss_file)


            plot_loss_graph(epoch_vect, gen_losses, psd_losses, delta_losses, "Generator-Loss.pdf", "Total Loss", "Psd Loss", "Delta Loss",  self.generated_images_folder)
                
