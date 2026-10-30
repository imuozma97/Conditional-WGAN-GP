"""
Comprueba qué índices de simulación existen en cada conjunto fiducial de Quijote
y los compara con el rango esperado, sin mirar nada a ojo.

Uso:
    python comprobar_indices.py ID_COLECCION
    python comprobar_indices.py ID_COLECCION --snapdirs 5   # además revisa 5 índices al azar

Requisitos: globus-cli instalado y `globus login` hecho.
"""
import argparse
import json
import random
import subprocess
import sys

# conjunto -> rango esperado de índices que nos interesa
CONJUNTOS = {
    "fiducial_HR": range(0, 100),
    "fiducial_LR": range(0, 1000),
    "fiducial":    range(0, 1000),   # solo comprobamos los que pueden emparejarse
}
SNAPDIRS = [f"snapdir_{n:03d}" for n in range(5)]


def globus_ls(coleccion, ruta):
    """Devuelve la lista de entradas (dicts con 'name' y 'type') de una carpeta."""
    cmd = ["globus", "ls", f"{coleccion}:{ruta}", "-F", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Error al listar {ruta}:\n{res.stderr.strip()}")
    return json.loads(res.stdout).get("DATA", [])


def indices_numericos(entradas):
    """Nombres de carpeta que son números enteros (descarta NCV_X, CAMB_TABLES, etc.)."""
    return sorted(int(e["name"]) for e in entradas
                  if e.get("type") == "dir" and e["name"].isdigit())


def resumir(nombre, encontrados, esperado):
    esperado = set(esperado)
    enc = set(encontrados)
    faltan = sorted(esperado - enc)
    print(f"\n=== {nombre} ===")
    print(f"  carpetas numéricas encontradas: {len(enc)}"
          + (f" (de {min(enc)} a {max(enc)})" if enc else ""))
    print(f"  esperadas en el rango {min(esperado)}–{max(esperado)}: {len(esperado)}")
    if faltan:
        muestra = faltan[:20]
        print(f"  FALTAN {len(faltan)}: {muestra}{' ...' if len(faltan) > 20 else ''}")
    else:
        print("  OK: están todos los índices del rango esperado")
    fuera = sorted(i for i in enc if i not in esperado)
    if fuera:
        print(f"  además hay {len(fuera)} índices fuera del rango "
              f"(de {fuera[0]} a {fuera[-1]})")
    return enc & esperado


def revisar_snapdirs(coleccion, nombre, indices, n):
    muestra = random.sample(sorted(indices), min(n, len(indices)))
    print(f"  revisando snapdirs en {len(muestra)} índices al azar: {sorted(muestra)}")
    for i in sorted(muestra):
        nombres = {e["name"] for e in globus_ls(coleccion, f"/Snapshots/{nombre}/{i}/")}
        falta = [s for s in SNAPDIRS + ["ICs"] if s not in nombres]
        print(f"    {i}: {'OK' if not falta else 'faltan ' + ', '.join(falta)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coleccion", help="ID de la colección de Quijote en Globus")
    ap.add_argument("--snapdirs", type=int, default=0,
                    help="número de índices al azar en los que revisar snapdir_000-004 e ICs")
    args = ap.parse_args()

    validos = {}
    for nombre, esperado in CONJUNTOS.items():
        try:
            entradas = globus_ls(args.coleccion, f"/Snapshots/{nombre}/")
        except RuntimeError as e:
            print(f"\n=== {nombre} ===\n  {e}")
            continue
        validos[nombre] = resumir(nombre, indices_numericos(entradas), esperado)
        if args.snapdirs and validos[nombre]:
            revisar_snapdirs(args.coleccion, nombre, validos[nombre], args.snapdirs)

    # pares posibles (índices presentes en los dos conjuntos)
    print("\n=== Pares posibles ===")
    for a, b in [("fiducial", "fiducial_HR"), ("fiducial_LR", "fiducial")]:
        if a in validos and b in validos:
            comunes = sorted(validos[a] & validos[b])
            print(f"  {a} → {b}: {len(comunes)} índices comunes"
                  + (f" (de {comunes[0]} a {comunes[-1]})" if comunes else ""))


if __name__ == "__main__":
    sys.exit(main())