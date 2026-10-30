"""
Funciones principales del entrenamiento
-train step: hace cada batch
"""

import tensorflow as tf
import os
import json
import numpy as np


from grad_pen_sr import gradient_penalty
from power_sr import Power
from psd_utils import psd_loss_log_sr
from loss_plot import plot_loss_graph
from transforms_sr import backward
from config_sr import lado_recorte_HR


class Training(tf.keras.Model):

    def __init__(self, data_class, discriminator, generator, batch_size, ncritic, trained_models_folder, generated_images_folder,
                 lambda_term, margen, boxsize, use_psd_loss = True,
                 grad_ratio = 1.0, lambda_beta = 0.99, lambda_max = 1000.0, warmup_epochs = 0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = batch_size
        self.trained_models_folder= trained_models_folder
        self.generated_images_folder = generated_images_folder
        self.current_epoch = 0
        self.ncritic = ncritic
        self.use_psd_loss = use_psd_loss
        self.data_class = data_class
        self.lambda_term = lambda_term
        self.margen = margen
        self.boxsize = boxsize

        self.power = Power(image_size=lado_recorte_HR, boxsize=self.boxsize)

        # lambda_t = r * ||grad L_adv|| / ||grad L_psd||, suavizado con una media móvil (EMA)
        self.grad_ratio_value = grad_ratio
        self.warmup_epochs = warmup_epochs
        self.lambda_beta = lambda_beta
        self.lambda_max = lambda_max
        self.grad_ratio = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="grad_ratio")
        self.lambda_psd = tf.Variable(-1.0, trainable=False, dtype=tf.float32, name="lambda_psd")  # -1 = EMA sin iniciar



    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer


    @tf.function    
    def train_step(self, data):
            
        real_LR, real_HR, z_values = data  

        # Recortes de los cubos para evitar las celdas que tienen información falsa
        m = self.margen
        real_LR_c = real_LR[:, m:-m, m:-m, m:-m, :]                  # (B,56,56,56,1)
        real_HR_c = real_HR[:, 2*m:-2*m, 2*m:-2*m, 2*m:-2*m, :]

        # Espectro del HR real (objetivo del PSD; no necesita gradiente)
        psd_real = self.power.compute_all_psd(backward(real_HR_c) - 1)

        # Discriminador (sin ruido, el generador da la misma salida en las ncritic iteraciones)
        fake_HR = self.generator([real_LR, z_values], training=True)
        fake_HR_c = fake_HR[:, 2*m:-2*m, 2*m:-2*m, 2*m:-2*m, :]

        for _ in range(self.ncritic):
            with tf.GradientTape() as disc_tape:
                
                fake_predictions = self.discriminator([fake_HR_c, real_LR_c, z_values], training=True)
                real_predictions = self.discriminator([real_HR_c, real_LR_c, z_values], training=True)
                
                gp, grads_norm_mean = gradient_penalty(real_HR_c, fake_HR_c, real_LR_c, z_values, self.discriminator, self.batch_size)
                
                disc_loss_fake = tf.reduce_mean(fake_predictions)
                disc_loss_real = tf.reduce_mean(real_predictions)
                wass_loss = disc_loss_fake - disc_loss_real

                disc_loss = wass_loss + self.lambda_term * gp

            grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            norm_disc = tf.linalg.global_norm(grads_disc)
            self.d_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

            # Generador
        with tf.GradientTape(persistent=True) as gen_tape:

            fake_HR = self.generator([real_LR, z_values], training=True)
            fake_HR_c = fake_HR[:, 2*m:-2*m, 2*m:-2*m, 2*m:-2*m, :]

            psd_fake = self.power.compute_all_psd(backward(fake_HR_c) - 1)
            
            fake_predictions = self.discriminator([fake_HR_c, real_LR_c, z_values], training=True)

            loss_adv = -tf.reduce_mean(fake_predictions)
            #percent = psd_out_of_band_fraction(psd_fake, psd_min_HR, psd_max_HR)
            loss_psd = psd_loss_log_sr(psd_fake, psd_real)

        percent = tf.constant(0.0)
        g_vars = self.generator.trainable_variables
        grads_adv = gen_tape.gradient(loss_adv, g_vars)
        grads_psd = gen_tape.gradient(loss_psd, g_vars)
        del gen_tape
        grads_psd = [tf.zeros_like(v) if g is None else g for g, v in zip(grads_psd, g_vars)]

        norm_adv = tf.linalg.global_norm(grads_adv)
        norm_psd = tf.linalg.global_norm(grads_psd)
        dot = tf.add_n([tf.reduce_sum(g_a * g_p) for g_a, g_p in zip(grads_adv, grads_psd)])
        cos_adv_psd = dot / (norm_adv * norm_psd + 1e-12)   # < 0: los dos términos tiran en sentidos opuestos

        if self.use_psd_loss:
            lambda_t = self.grad_ratio * norm_adv / (norm_psd + 1e-12)
            lambda_t = tf.clip_by_value(lambda_t, 0.0, self.lambda_max)
            lam = tf.where(self.lambda_psd < 0.0, lambda_t,
                           self.lambda_beta * self.lambda_psd + (1.0 - self.lambda_beta) * lambda_t)
            self.lambda_psd.assign(lam)
            grads_gen = [g_a + lam * g_p for g_a, g_p in zip(grads_adv, grads_psd)]
            gen_loss = loss_adv + lam * loss_psd
        else:
            lam = tf.constant(0.0)
            grads_gen = grads_adv
            gen_loss = loss_adv

        norm_gen = tf.linalg.global_norm(grads_gen)
        self.g_optimizer.apply_gradients(zip(grads_gen, g_vars))

        ratio1 = norm_disc / (norm_gen + 1e-8)

        return wass_loss, disc_loss_real, disc_loss_fake, gen_loss, loss_psd, percent, grads_norm_mean, ratio1, norm_disc, norm_gen, lam, norm_adv, norm_psd, cos_adv_psd


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
            start_epoch = int(checkpoint.epoch.numpy())  + 1 # Recuperar la última época guardada

            if os.path.exists(loss_file):
                print("Cargando histórico de pérdidas...")
                with open(loss_file, 'r') as f:
                    data = json.load(f)

                epoch_vect = data.get("epoch_vect", [])
                wass_losses = data.get('wass_losses', [])
                disc_losses_f = data.get('disc_losses_f', [])
                disc_losses_r = data.get('disc_losses_r', [])
                adv_losses = data.get('adv_losses', [])
                psd_losses = data.get('psd_losses', [])
                grad_pen = data.get('grad_pen', [])
                percents = data.get('percents', [])
                ratios1 = data.get('ratio1', [])
                norms_disc = data.get('norms_disc', [])
                norms_gen = data.get('norms_gen', [])

                best_epoch = data.get('best_epoch', [])
                best_psd = data.get('best_psd', [])
                best_percent = data.get('best_percent', [])
                best_epoch_psd = data.get('best_epoch_psd', [])
                best_epoch_percent = data.get('best_epoch_percent', [])

                lambdas = data.get('lambda_psd', [])
                norms_adv = data.get('norm_adv', [])
                norms_psd = data.get('norm_psd', [])
                cosines = data.get('cos_adv_psd', [])
                if lambdas:
                    self.lambda_psd.assign(lambdas[-1])

                best_percent_metric = best_percent[-1]
                best_psd_metric = best_psd[-1]



        else:
            print("No se encontraron checkpoints previos, iniciando desde cero.")
            start_epoch = 0
        
            epoch_vect = []
            wass_losses = []
            disc_losses_r, disc_losses_f  = [],  []
                
            adv_losses = []
            psd_losses = []
                
            grad_pen = []
            percents = []

            ratios1 = []
                
            norms_disc, norms_gen= [], []

            best_percent_metric = best_percent[-1] if best_percent else float("inf")
            best_psd_metric = best_psd[-1] if best_psd else float("inf")
            best_psd, best_epoch_psd, best_percent, best_epoch_percent = [], [], [], []

            lambdas, norms_adv, norms_psd, cosines = [], [], [], []
        
            
        for epoch in range(start_epoch, epochs):
            self.grad_ratio.assign(self.grad_ratio_value if epoch >= self.warmup_epochs else 0.0)
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                self.lambda_psd.assign(-1.0)   # reinicia la EMA: al acabar el warm-up arranca en lambda_t
            lam_ep, nadv_ep, npsd_ep, cos_ep = 0, 0, 0, 0
            self.current_epoch = epoch
            batch_count = 0
            wass_loss, disc_loss_r, disc_loss_f  = 0, 0, 0
            adv_loss = 0
            psd_loss = 0
            gp = 0
            percent = 0
            ratio1 = 0

            norm_disc, norm_gen = 0 , 0
            
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
                norm_disc += losses[8]
                norm_gen += losses[9]

                lam_ep += losses[10]
                nadv_ep += losses[11]
                npsd_ep += losses[12]
                cos_ep += losses[13]
                
                #psd_gen_batch = losses[8]
                #psd_max_batch = losses[9]
                #psd_min_batch = losses[10]
                #percent_batch = losses[5]

                
                    
                batch_count += 1
                    

            wass_loss /= batch_count
            disc_loss_r /= batch_count
            disc_loss_f /= batch_count
                
            adv_loss /= batch_count
            psd_loss /= batch_count
            percent /= batch_count

            gp /= batch_count
                
            ratio1 /= batch_count

            norm_disc /= batch_count
            norm_gen /= batch_count
                

            if epoch > 150 and psd_loss < best_psd_metric:
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
                    

            wass_losses.append(float(wass_loss.numpy()))
            disc_losses_f.append(float(disc_loss_f.numpy()))
            disc_losses_r.append(float(disc_loss_r.numpy()))
                
            adv_losses.append(float(adv_loss.numpy()))
            psd_losses.append(float(psd_loss.numpy()))
            percents.append(float(percent.numpy()))
                
            grad_pen.append(float(gp.numpy()))
            ratios1.append(float(ratio1.numpy()))

            norms_disc.append(float(norm_disc.numpy()))
            norms_gen.append(float(norm_gen.numpy()))

            epoch_vect.append(epoch)

            lambdas.append(float((lam_ep / batch_count).numpy()))
            norms_adv.append(float((nadv_ep / batch_count).numpy()))
            norms_psd.append(float((npsd_ep / batch_count).numpy()))
            cosines.append(float((cos_ep / batch_count).numpy()))
            print(f'  lambda medio = {lambdas[-1]:.4g}, |g_adv| = {norms_adv[-1]:.3g}, |g_psd| = {norms_psd[-1]:.3g}, cos = {cosines[-1]:.3f}')

            checkpoint.epoch.assign(epoch)
            checkpoint_manager.save()


            # Guardamos pérdidas en archivo
            tmp_file = loss_file + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump({
                        'epoch_vect' : epoch_vect,
                        'disc_losses_f' : disc_losses_f,
                        'disc_losses_r' : disc_losses_r,
                        'wass_losses': wass_losses,
                        'adv_losses': adv_losses,
                        'grad_pen' : grad_pen,
                        'psd_losses': psd_losses,
                        'percents' : percents,
                        'best_psd' : best_psd, 
                        'best_epoch_psd' : best_epoch_psd,
                        'best_percent' : best_percent,
                        'best_epoch_percent' : best_epoch_percent,
                        'ratio1' : ratios1, 
                        'norm_disc' : norms_disc, 
                        'norm_gen' : norms_gen,
                        'lambda_psd' : lambdas,
                        'norm_adv' : norms_adv,
                        'norm_psd' : norms_psd,
                        'cos_adv_psd' : cosines,
                    }, f)
            os.replace(tmp_file, loss_file)


            plot_loss_graph(epoch_vect, wass_losses, adv_losses, "Wasserstein-Loss.pdf", "Wasserstein Loss", "Adv Loss", self.generated_images_folder)
            plot_loss_graph(epoch_vect, disc_losses_f, disc_losses_r, "Distance-Loss.pdf", "Disc Loss Fake", "Disc Loss Real", self.generated_images_folder)
            plot_loss_graph(epoch_vect, grad_pen, None,  "Gradient-penalty.pdf", "Gradient Penalty", "GP" ,self.generated_images_folder)
                
