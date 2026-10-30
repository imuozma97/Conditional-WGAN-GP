import h5py
import hdf5plugin



print("Media resolution: 512^3")

with h5py.File("fiducial/0/snapdir_000/snap_000.4.hdf5", "r") as f:
    h = f["Header"].attrs
    print(dict(h))


