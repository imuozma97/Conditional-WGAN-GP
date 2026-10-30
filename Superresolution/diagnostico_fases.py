"""
Diagnóstico de un par LR/HR de Quijote: ¿comparten fases? ¿a qué escala difieren?

Para cada simulación:
  1. lee las posiciones de los 8 archivos del snapshot (por trozos, para no agotar memoria),
  2. construye la densidad en una malla común (NGP),
  3. calcula P_LR(k), P_HR(k) y el espectro cruzado P_x(k).

Salida:
  - r(k) = P_x / sqrt(P_LR * P_HR)   (≈ 1 a k bajo si las fases coinciden; ≈ 0 si no)
  - P_HR(k) / P_LR(k)                (≈ 1 a k bajo; dónde se separa indica dónde difieren)
  sin corregir el ruido de disparo (ver nota en main); r_corr es solo orientativa.

Uso:
  python diagnostico_fases.py RUTA_LR/snapdir_004 RUTA_HR/snapdir_004 --grid 256

Requisitos: numpy, h5py, hdf5plugin, matplotlib.
"""
import argparse
import glob
import os
import sys

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (registra el filtro de compresión de Quijote)

TROZO = 10_000_000   # partículas leídas de una vez


def archivos_snapshot(carpeta):
    fs = sorted(glob.glob(os.path.join(carpeta, "*.hdf5")),
                key=lambda p: int(p.split(".")[-2]) if p.split(".")[-2].isdigit() else 0)
    if not fs:
        sys.exit(f"No hay archivos .hdf5 en {carpeta}")
    return fs


def cabecera(f0):
    with h5py.File(f0, "r") as f:
        h = f["Header"].attrs
        return {
            "BoxSize_Mpc_h": float(h["BoxSize"]) / 1e3,
            "N": int(h["NumPart_Total"][1]) + (int(h["NumPart_Total_HighWord"][1]) << 32),
            "z": float(h["Redshift"]),
            "Omega0": float(h["Omega0"]),
            "h": float(h["HubbleParam"]),
            "nfiles": int(h["NumFilesPerSnapshot"]),
        }


def malla_ngp(carpeta, grid, L):
    """Recuentos de partículas por celda (NGP), leyendo por trozos."""
    cuentas = np.zeros(grid**3, dtype=np.int64)
    total = 0
    for fn in archivos_snapshot(carpeta):
        with h5py.File(fn, "r") as f:
            pos = f["PartType1/Coordinates"]
            n = pos.shape[0]
            for i in range(0, n, TROZO):
                p = pos[i:i + TROZO].astype(np.float64) / 1e3          # kpc/h -> Mpc/h
                idx = np.floor(p / L * grid).astype(np.int64) % grid    # periódico
                plano = (idx[:, 0] * grid + idx[:, 1]) * grid + idx[:, 2]
                cuentas += np.bincount(plano, minlength=grid**3)
                total += len(p)
        print(f"   leído {os.path.basename(fn)}  (acumuladas {total:,} partículas)")
    return cuentas.reshape(grid, grid, grid), total


def espectros(d1, d2, L, nbins_por_kf=1):
    """P1, P2 y cruzado en bins de k (anchura = k_F * nbins_por_kf)."""
    g = d1.shape[0]
    f1 = np.fft.rfftn(d1)
    f2 = np.fft.rfftn(d2)
    kF = 2 * np.pi / L
    kx = np.fft.fftfreq(g, d=1.0 / g) * kF
    kz = np.fft.rfftfreq(g, d=1.0 / g) * kF
    kmod = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    norm = L**3 / g**6
    p1 = (np.abs(f1)**2).ravel() * norm
    p2 = (np.abs(f2)**2).ravel() * norm
    px = (f1 * np.conj(f2)).real.ravel() * norm
    k = kmod.ravel()
    kny = np.pi * g / L
    ancho = kF * nbins_por_kf
    b = np.floor(k / ancho + 0.5).astype(np.int64)      # bins centrados en múltiplos de k_F
    ok = (k > 0) & (k <= kny)
    b, k, p1, p2, px = b[ok], k[ok], p1[ok], p2[ok], px[ok]
    nb = b.max() + 1
    cnt = np.bincount(b, minlength=nb)
    m = cnt > 0
    med = lambda w: np.bincount(b, weights=w, minlength=nb)[m] / cnt[m]
    return med(k), med(p1), med(p2), med(px), cnt[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lr", help="carpeta snapdir de la simulación de baja resolución")
    ap.add_argument("hr", help="carpeta snapdir de la simulación de alta resolución")
    ap.add_argument("--grid", type=int, default=256, help="celdas por lado de la malla común")
    ap.add_argument("--salida", default="diagnostico_fases", help="prefijo de los archivos de salida")
    args = ap.parse_args()

    hl = cabecera(archivos_snapshot(args.lr)[0])
    hh = cabecera(archivos_snapshot(args.hr)[0])
    print("Cabeceras:")
    for k in hl:
        print(f"  {k:14s} LR = {hl[k]:<16}  HR = {hh[k]}")
    for k in ("BoxSize_Mpc_h", "z", "Omega0", "h"):
        if abs(hl[k] - hh[k]) > 1e-6 * max(1, abs(hl[k])):
            print(f"  ¡AVISO! {k} no coincide entre LR y HR")
    L = hl["BoxSize_Mpc_h"]

    print(f"\nMalla LR ({args.grid}^3)...")
    nl, Nl = malla_ngp(args.lr, args.grid, L)
    print(f"Malla HR ({args.grid}^3)...")
    nh, Nh = malla_ngp(args.hr, args.grid, L)

    dl = nl / nl.mean() - 1.0
    dh = nh / nh.mean() - 1.0
    del nl, nh

    k, pl, ph, px, cnt = espectros(dl, dh, L)
    # Ruido de disparo de Poisson V/N. OJO: las partículas N-body parten de una retícula y su
    # ruido real suele ser MENOR que el de Poisson; restarlo puede sobrecorregir (espectros
    # negativos, r > 1). Por eso los resultados principales son SIN corregir; la columna
    # corregida es solo orientativa.
    snl, snh = L**3 / Nl, L**3 / Nh
    with np.errstate(invalid="ignore", divide="ignore"):
        r = px / np.sqrt(pl * ph)                 # sin corregir: cota inferior de la correlación
        cociente = ph / pl
        r_corr = px / np.sqrt((pl - snl) * (ph - snh))

    kny = np.pi * args.grid / L
    print(f"\nRuido de Poisson V/N: LR = {snl:.3g}  HR = {snh:.3g}  (Mpc/h)^3"
          f"   (n̄ por celda: LR = {Nl/args.grid**3:.3g}, HR = {Nh/args.grid**3:.3g})")
    if Nl / args.grid**3 < 4:
        print("  AVISO: pocas partículas LR por celda; el ruido domina a k alto. "
              "Prueba una malla más gruesa (--grid menor).")
    print(f"k_F = {2*np.pi/L:.4f} h/Mpc,  k_Nyquist de la malla = {kny:.3f} h/Mpc\n")
    print(f"{'k [h/Mpc]':>10} {'modos':>7} {'P_LR':>11} {'P_HR':>11} {'P_HR/P_LR':>10} {'r(k)':>7} {'r_corr':>7}")
    paso = max(1, len(k) // 30)
    for i in sorted(set(list(range(0, len(k), paso)) + [len(k) - 1])):
        print(f"{k[i]:10.4f} {cnt[i]:7d} {pl[i]:11.4g} {ph[i]:11.4g} {cociente[i]:10.4f} "
              f"{r[i]:7.4f} {r_corr[i]:7.3f}")

    np.savetxt(f"{args.salida}.txt",
               np.column_stack([k, cnt, pl, ph, px, cociente, r, r_corr]),
               header=f"k[h/Mpc] modos P_LR P_HR P_cruzado P_HR/P_LR r(k) r_corr(k)   "
                      f"[V/N: LR={snl:.4g} HR={snh:.4g}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].semilogx(k, r, ".-")
        ax[0].axhline(1, color="gray", lw=0.8)
        ax[0].set(xlabel="k [h/Mpc]", ylabel="r(k)", title="Correlación cruzada LR–HR", ylim=(-0.1, 1.1))
        ax[1].semilogx(k, cociente, ".-")
        ax[1].axhline(1, color="gray", lw=0.8)
        ax[1].set(xlabel="k [h/Mpc]", ylabel="P_HR / P_LR", title="Razón de espectros (sin corregir ruido)")
        for a in ax:
            a.axvline(kny, color="r", ls="--", lw=0.8)
        fig.suptitle(f"z = {hl['z']:.2f}   N_LR = {Nl:,}   N_HR = {Nh:,}   malla {args.grid}^3")
        fig.tight_layout()
        fig.savefig(f"{args.salida}.png", dpi=130)
        print(f"\nGuardado: {args.salida}.png y {args.salida}.txt")
    except ImportError:
        print(f"\nmatplotlib no disponible; guardado solo {args.salida}.txt")


if __name__ == "__main__":
    main()