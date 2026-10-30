"""
Estadísticas de x = n / n̄ (= δ + 1) en un archivo de mallas (nivel 1), para elegir el parámetro c
de la transformación forward(x) = f0(x + shift) - f0(shift).

Uso:
    python estadisticas_transformacion.py mallas/fiducial/0000.hdf5 --snap 4
    python estadisticas_transformacion.py mallas/fiducial/0000.hdf5 --snap 4 --agrupar 2   # malla 2x más gruesa

--agrupar 2 suma bloques de 2x2x2 celdas (p. ej. 512^3 -> 256^3, n̄ ×8). Se puede repetir (4 -> 4x4x4).
Requisitos: numpy, h5py, hdf5plugin.
"""
import argparse
import numpy as np
import h5py
import hdf5plugin  # noqa: F401

SHIFT = 1e-4
CANDIDATOS_C = [100, 300, 1000, 3000, 10000]


def f0(y, c):
    """Misma fórmula que stat_forward_0 de transforms.py, en numpy."""
    y = np.asarray(y, dtype=np.float64)
    return np.where(y <= c, np.log(y + 1), np.log(c + 1) + (y / c - 1)) * (3 / np.log(c + 1))


def forward(x, c, shift=SHIFT):
    return f0(x + shift, c) - f0(shift, c)


def agrupar(n, f):
    g = n.shape[0] // f
    return n.reshape(g, f, g, f, g, f).sum(axis=(1, 3, 5), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--snap", type=int, default=4, help="0=z3 ... 4=z0")
    ap.add_argument("--agrupar", type=int, default=1, help="factor de agrupación por lado (1, 2, 4...)")
    a = ap.parse_args()

    with h5py.File(a.archivo, "r") as f:
        if not f["completado"][a.snap]:
            raise SystemExit(f"El snapshot {a.snap} no está procesado en {a.archivo}")
        n = f["maps"][a.snap].astype(np.int64)
        z = float(f["redshift"][a.snap])
        conj = f.attrs["conjunto"]
    if a.agrupar > 1:
        n = agrupar(n, a.agrupar)

    g = n.shape[0]
    nbar = n.sum() / n.size
    x = (n / nbar).ravel()

    print(f"{conj}  z = {z:.2f}  malla {g}^3  n̄ = {nbar:.3g} partículas/celda")
    print(f"celdas vacías: {np.mean(n == 0) * 100:.2f} %")
    print("\nx = n/n̄ (= δ+1):")
    for q in [50, 90, 99, 99.9, 99.99, 99.999]:
        print(f"  percentil {q:>7}: {np.percentile(x, q):10.2f}")
    print(f"  máximo          : {x.max():10.2f}")

    print("\n     c   celdas con x > c   salida máx.   salida en percentil 99.9")
    p999 = np.percentile(x, 99.9)
    for c in CANDIDATOS_C:
        frac = np.mean(x > c) * 100
        print(f"{c:6d}   {frac:13.5f} %   {float(forward(x.max(), c)):9.2f}   {float(forward(p999, c)):9.2f}")


if __name__ == "__main__":
    main()