"""
Funciones principales del entrenamiento
-train step: hace cada batch
"""

import tensorflow as tf
import os
import json
import numpy as np


from config import latent_dim
from grad_pen import gradient_penalty
from power import Power
from psd_utils import psd_out_of_band_fraction, lambda_psd_schedule, psd_loss
from loss_plot import plot_loss_graph


power = Power()


class Training(tf.keras.Model):

    def __init__(self, data_class, discriminator, generator, batch_size, ncritic, trained_models_folder, generated_images_folder):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = batch_size
        self.trained_models_folder= trained_models_folder
        self.generated_images_folder = generated_images_folder
        self.current_epoch = 1
        self.ncritic = ncritic
        #self.maximo = maximo
        #self.minimo = minimo
        self.data_class = data_class

        #self.use_backward = backward is not None
        #self.backward = backward
        self.disc_psd = True


    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer


    @tf.function    
    def train_step(self, data):
            
        real_images, z_values, psd_max, psd_min,  psd_mean, sigma_log = data  

        for _ in range(self.ncritic):
            noise = tf.random.normal([self.batch_size, latent_dim]) 
                
            with tf.GradientTape() as disc_tape:
                    
                generated_images = self.generator([noise, z_values], training=True)
                psd_gen = power.compute_all_psd(generated_images)

                #Si el D tiene psd o no:
                if self.disc_psd:
                    fake_predictions = self.discriminator([generated_images, z_values, psd_gen], training=True)
                    real_predictions = self.discriminator([real_images, z_values, psd_mean], training=True)
                else:
                    fake_predictions = self.discriminator([generated_images, z_values], training=True)
                    real_predictions = self.discriminator([real_images, z_values], training=True)
                    
                disc_loss_fake = tf.reduce_mean(fake_predictions)
                disc_loss_real = tf.reduce_mean(real_predictions)
                wass_loss = disc_loss_fake - disc_loss_real

                gp, grads_norm_mean = gradient_penalty(real_images, generated_images, z_values, self.discriminator, self.batch_size, 10, psd_mean)

                disc_loss = wass_loss + gp

            grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            norm_disc = tf.linalg.global_norm(grads_disc)
            self.d_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

            # Generador
        noise = tf.random.normal([self.batch_size, latent_dim]) 
        with tf.GradientTape(persistent = True) as gen_tape:
                
            generated_images = self.generator([noise, z_values], training=True)
            psd_gen = power.compute_all_psd(generated_images)

            #Si el D tiene psd o no:
            if self.disc_psd:
                fake_predictions = self.discriminator([generated_images, z_values, psd_gen], training=True)
            else:
                fake_predictions = self.discriminator([generated_images, z_values], training=True)
                    
            loss_adv = -tf.reduce_mean(fake_predictions)
            percent = psd_out_of_band_fraction(psd_gen, psd_min, psd_max)

            loss_psd = psd_loss(psd_gen, psd_mean, sigma_log) 

            #lambda_psd = lambda_psd_schedule(self.current_epoch)

            gen_loss = loss_adv #+ lambda_psd*loss_psd
                

        grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_psd = gen_tape.gradient(loss_psd, self.generator.trainable_variables)
        grads_adv = gen_tape.gradient(loss_adv, self.generator.trainable_variables)

        norm_gen = tf.linalg.global_norm(grads_gen)
        norm_psd = tf.linalg.global_norm(grads_psd)
        norm_adv = tf.linalg.global_norm(grads_adv)
    
            
        self.g_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
            
        ratio1 = norm_disc / (norm_adv + 1e-8)
        ratio2 = norm_adv / (norm_psd + 1e-8)
        ratio3 = norm_disc / (norm_gen + 1e-8)

        return wass_loss, disc_loss_real, disc_loss_fake, loss_adv, loss_psd, percent, grads_norm_mean, ratio1, ratio2, ratio3, psd_gen, psd_max, psd_min
        
    



    def train(self, dataset_train,  epochs):
        
        epoch_vect = []
        wass_losses = []
        disc_losses_r, disc_losses_f  = [],  []
            
        adv_losses = []
        psd_losses = []
            
        grad_pen = []
        percents = []
            
        ratios1, ratios2, ratios3 = [], [], []

        best_metric = float("inf")
        best_psd, best_epoch, best_percent = [], [], []
        

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
        else:
            print("No se encontraron checkpoints previos, iniciando desde cero.")
            start_epoch = 0

                
            
        for epoch in range(start_epoch, epochs):
                
            self.current_epoch = epoch
            batch_count = 0
            wass_loss, disc_loss_r, disc_loss_f  = 0, 0, 0
            adv_loss = 0
            psd_loss = 0
            gp = 0
                
            percent = 0
                
            ratio1, ratio2, ratio3 = 0, 0, 0
        
            
            print('Currently training on epoch {} (out of {}).'.format(epoch, epochs))

            for image_batch in dataset_train:
                    
                losses = self.train_step(image_batch)
                wass_loss += -losses[0]
                disc_loss_r += losses[1]
                disc_loss_f += losses[2]

                adv_loss += losses[3]
                psd_loss += losses[4]
                percent += losses[5]
                    
                gp += losses[6]
                    
                ratio1 += losses[7]
                ratio2 += losses[8]
                ratio3 += losses[9]
                
                psd_gen_batch = losses[10]
                psd_max_batch = losses[11]
                psd_min_batch = losses[12]
                percent_batch = losses[5]
                    
                batch_count += 1
                    

            wass_loss /= batch_count
            disc_loss_r /= batch_count
            disc_loss_f /= batch_count
                
            adv_loss /= batch_count
            psd_loss /= batch_count
            percent /= batch_count

            gp /= batch_count
                
            ratio1 /= batch_count
            ratio2 /= batch_count
            ratio3 /= batch_count
                
            

            
            #Aquí guardará el modelo únicamente si mejora
            #if epoch > 150 and percent < best_metric:
            if epoch == 1:
                best_metric = percent

                gen_path = os.path.join(self.trained_models_folder, "best_generator", f"epoch_{epoch:05d}")
                os.makedirs(gen_path, exist_ok=True)
                self.generator.save(gen_path)

                best_psd.append(float(psd_loss.numpy()))
                best_percent.append(float(percent.numpy()))
                best_epoch.append(epoch)

                np.savez(os.path.join(gen_path, "psd_data.npz"),
                    psd_gen = psd_gen_batch.numpy(),
                    psd_min=psd_min_batch.numpy(),
                    psd_max=psd_max_batch.numpy(),
                    percent = float(percent_batch.numpy())
                    )

                checkpoint.epoch.assign(epoch)
                checkpoint_manager.save()

                print(f"Checkpoint guardado en época {epoch}")
                    
                    
        
            if self.current_epoch % 150 == 0:
                print(f"Guardando modelo por estabilización después de 150 épocas.")

                gen_dir = os.path.join(self.trained_models_folder, "generator_stable")
                os.makedirs(gen_dir, exist_ok=True)
                gen_path = os.path.join(gen_dir, f"generator_epoch_{epoch:05d}.weights.h5")

                self.generator.save_weights(gen_path)
                    
                    

            wass_losses.append(float(wass_loss.numpy()))
            disc_losses_f.append(float(disc_loss_f.numpy()))
            disc_losses_r.append(float(disc_loss_r.numpy()))
                
            adv_losses.append(float(adv_loss.numpy()))
            psd_losses.append(float(psd_loss.numpy()))
            percents.append(float(percent.numpy()))
                
            grad_pen.append(float(gp.numpy()))
                
            ratios1.append(float(ratio1.numpy()))
            ratios2.append(float(ratio2.numpy()))
            ratios3.append(float(ratio3.numpy()))
                
            
                
            epoch_vect.append(epoch)


                # Guardamos pérdidas en archivo
            with open(loss_file, 'w') as f:
                json.dump({
                        'wass_losses': wass_losses,
                        'adv_losses': adv_losses,
                        'psd_losses': psd_losses,
                        'grad_pen' : grad_pen,
                        'best_epoch' : best_epoch,
                        'best_psd' : best_psd, 
                        'percents' : percents,
                        'ratio1' : ratios1, 
                        'ratio2': ratios2,
                        'ratio3' : ratios3
                    }, f)
    


            plot_loss_graph(epoch_vect, wass_losses, adv_losses, "Wasserstein-Loss.pdf", "Wasserstein Loss", "Adv Loss")
            plot_loss_graph(epoch_vect, disc_losses_f, disc_losses_r, "Distance-Loss.pdf", "Disc Loss Fake", "Disc Loss Real")
            plot_loss_graph(epoch_vect, grad_pen, None,  "Gradient-penalty.pdf", "Gradient Penalty", "GP")
                
