caja_quijote = 1000.0                 # Mpc/h
malla_HR = 1024
malla_LR = 512                        # celdas por lado de la caja HR (1024 en la otra opción)
celda_HR = caja_quijote / malla_HR    # Mpc/h por celda
lado_recorte_HR = 112                 # celdas que ve el discriminador / la pérdida
boxsize = lado_recorte_HR * celda_HR
embedding_dim = 32


batch_size = 4

ncritic1 = 1
ncritic2 = 2
ncritic3 = 3
ncritic4 = 4
ncritic5 = 5    

shift = 0.0001
c = 10000
