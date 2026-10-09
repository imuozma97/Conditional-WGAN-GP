import requests
import numpy as np
import os
import h5py
#import hdf5plugin
from functools import partial
#from scipy.interpolate import make_interp_spline



def datos(cv, snap):
    #1- Cargamos los datos de cada caja
    snapshot = 'Sim_hidrodinamicas/CV_{}/snapshot_{}.hdf5'.format(cv,snap)
        
    f = h5py.File(snapshot, 'r')

    pos_g = f['PartType0/Coordinates'][:]/1e3
    redshift = f['Header'].attrs[u'Redshift']
    mass = f['PartType0/Masses'][:]

    f.close()
    
    return pos_g, redshift, mass

snap = np.array([14, 18, 24, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90])
cv = np.linspace(1, 26, 26, dtype="int")



for i in cv:
    pos = []
    redshift = []
    masses = []

    for j in snap:
        nombre_carpeta = 'Sim_hidrodinamicas/CV_{}'.format(i)
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)

        url = 'https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_{}/snapshot_0{}.hdf5'.format(i, j)
        r = requests.get(url, allow_redirects=True)

        open('Sim_hidrodinamicas/CV_{}/snapshot_0{}.hdf5'.format(i, j), 'wb').write(r.content)
        print('Se ha descargado la snapshot ', i, ' de la CV ', j)


        snapshot = 'Sim_hidrodinamicas/CV_{}/snapshot_0{}'.format(i, j)
        f = h5py.File(snapshot, 'r')
        pos_g, redshift, mass =  datos(cv, snap)
        f.close()

        f = h5py.File("Sim_hidrodinamicas/Gas_positions_0.hdf5", 'w')
        f.create_dataset('positions', data = pos_g)
        f.create_dataset('train_labels', data = redshift)
        f.create_dataset('masses', data = mass)
        f.close()

        for j in snap:
        file_path = 'Sim_hidrodinamicas/CV_{}/snapshot_0{}.hdf5'.format(i, j)

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'El archivo {file_path} ha sido eliminado.')


"""





def position(cv, snap):
    #1- Cargamos los datos de cada caja
    snapshot = 'Camels_data/CV_{}/snapshot_{}.hdf5'.format(cv,snap)
        
    f = h5py.File(snapshot, 'r')

    redshift = f['Header'].attrs[u'Redshift']

    pos_dm = f['PartType1/Coordinates'][:]/1e3  #positions in Mpc/h
    
    f.close()
    
    #2- Sacamos cada coordenada y obtenemos el número de partículas por vóxel   
    #x = pos_dm[:,0]
    #y = pos_dm[:,1]
    #z = pos_dm[:,2]
    
    #h, edges = np.histogramdd((x, y), bins = 64)
    #h, edges = np.histogramdd((x, z), bins = 64)
    #h, edges = np.histogramdd((y, z), bins = 64)
    
    #H.append([h])
               
    #h = np.array(h)
    return pos_dm, redshift




snap = np.array([14, 18, 24, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90])
cajas = np.linspace(1, 26, 26, dtype="int")




for i in cajas:   
    pos = []
    redshift = []
     
    for j in snap:

        nombre_carpeta = 'Sim_hidrodinamicas/CV_{}'.format(i)
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)

        url = 'https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_{}/snapshot_0{}.hdf5'.format(i, j)
        r = requests.get(url, allow_redirects=True)

        open('Camels_data/CV_{}/snapshot_{}.hdf5'.format(i, j), 'wb').write(r.content)
        print('Se ha descargado la snapshot ', j, 'de la CV ', i)

        #Aquí ya está descargada y llamamos a la función de posiciones

        #print('Añadimos al histograma la snap: ', j)
        pos.append(position(i, j)[0])
        redshift.append(position(i, j)[1])


    pos = np.array(pos)
    redshift = np.array(redshift)

    #H = forward(histo)
    R = np.reshape(redshift, (len(snap), 1) )
  

    f = h5py.File("Camels_data/Positions_{}.hdf5".format(i), 'w')
    f.create_dataset('positions', data = pos)
    f.create_dataset('train_labels', data = R)
    f.close()
    
    for j in snap:
        file_path = 'Camels_data/CV_{}/snapshot_{}.hdf5'.format(i, j)

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'El archivo {file_path} ha sido eliminado.')
    

        """