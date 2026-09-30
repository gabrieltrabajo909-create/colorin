"""Convierte una ilustración en un dibujo "pintar por números" para la app.

Uso: .venv/bin/python convert.py imagen.png salida_id [--colors 140] [--size 1200] [--min-area 450]

Genera en ../app/catalog/<salida_id>/:
  regions.png  mapa de zonas (id = R*256 + G)
  data.json    paleta y, por zona: color, posición del número y radio libre
  thumb.jpg    miniatura terminada · line.png  miniatura solo contornos
"""
import argparse, colorsys, json, os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('name')
ap.add_argument('--colors', type=int, default=70)
ap.add_argument('--size', type=int, default=1200)
ap.add_argument('--min-area', type=int, default=250)   # zonas más chicas que esto no llevan número
ap.add_argument('--min-radius', type=float, default=3.0)
ap.add_argument('--similar', type=float, default=45)     # distancia de color para unir una zona chica a su vecina
a = ap.parse_args()

img = Image.open(a.src).convert('RGB')
s = a.size / max(img.size)
img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).filter(ImageFilter.MedianFilter(3))
arr = np.asarray(img).astype(np.int16)
H, W = arr.shape[:2]

# 1) sacar los contornos oscuros finos del dibujo: sus píxeles toman el color de la zona más cercana
lum = arr @ np.array([.299, .587, .114])
dark = lum < 75
line = dark & ~ndi.binary_opening(dark, structure=np.ones((5, 5)))   # contornos finos del dibujo original: quedan fijos
iy, ix = ndi.distance_transform_edt(line, return_distances=False, return_indices=True)
clean = arr[iy, ix].astype(np.uint8)

# 2) reducir a N colores y limpiar puntitos sueltos
q = Image.fromarray(clean).quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT, kmeans=4, dither=Image.Dither.NONE)
pal = np.array(q.getpalette()[:a.colors * 3]).reshape(-1, 3)
cidx = np.asarray(Image.fromarray(np.asarray(q), 'L').filter(ImageFilter.ModeFilter(5)))

# 3) zonas = partes conectadas del mismo color
R = np.zeros((H, W), np.int32); color_of = [0]; nxt = 1
for c in np.unique(cidx):
    lab, n = ndi.label((cidx == c) & ~line)
    m = lab > 0
    R[m] = lab[m] + nxt - 1
    color_of += [int(c)] * n; nxt += n
color_of = np.array(color_of)
print(f'{W}x{H}, zonas iniciales: {nxt - 1}')

# 4) unir zonas chicas o muy finas a la vecina con la que más borde comparten
auto = set()   # zonas diminutas con color propio: vienen ya pintadas, sin número
def merge_pass(bad_ids):
    slices = ndi.find_objects(R)
    merged = 0
    for r in bad_ids:
        if r in auto: continue
        sl = slices[r - 1]
        if sl is None: continue
        y0, y1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)
        x0, x1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)
        sub = R[y0:y1, x0:x1]
        m = sub == r
        if not m.any(): continue
        ring = ndi.binary_dilation(m) & ~m
        nb = np.unique(sub[ring]); nb = nb[(nb > 0) & ~np.isin(nb, list(auto))] if auto else nb[nb > 0]
        if nb.size == 0: auto.add(int(r)); continue
        dists = np.linalg.norm(pal[color_of[nb]] - pal[color_of[r]], axis=1)
        k = dists.argmin()
        if dists[k] <= a.similar: sub[m] = nb[k]; merged += 1
        else: auto.add(int(r))
    return merged

def radii():
    edge = np.zeros((H, W), bool)
    edge[:, 1:] |= R[:, 1:] != R[:, :-1]; edge[:, :-1] |= R[:, 1:] != R[:, :-1]
    edge[1:, :] |= R[1:, :] != R[:-1, :]; edge[:-1, :] |= R[1:, :] != R[:-1, :]
    return ndi.distance_transform_edt(~edge)

for it in range(6):
    sizes = np.bincount(R.ravel(), minlength=nxt)
    ids = np.nonzero(sizes)[0]; ids = ids[ids > 0]
    dist = radii()
    rad = np.zeros(nxt); rad[ids] = ndi.maximum(dist, R, ids)
    bad = ids[((sizes[ids] < a.min_area) | (rad[ids] < a.min_radius)) & ~np.isin(ids, list(auto))]
    if not len(bad): break
    bad = bad[np.argsort(sizes[bad])]
    print(f'pasada {it + 1}: uniendo {len(bad)} zonas chicas')
    if merge_pass(bad) == 0: break

# 5) renumerar zonas y colores usados
ids = np.unique(R); ids = ids[ids > 0]
remap = np.zeros(nxt, np.int32); remap[ids] = np.arange(1, len(ids) + 1)
R = remap[R]
reg_color = color_of[ids]
used = np.unique(reg_color)
def sort_key(c):
    r, g, b = pal[c] / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (0, l) if s < .12 else (1, round(h * 12) % 12, l)   # grises primero, después por tono y luz
paintable = {int(color_of[r]) for r in ids if int(r) not in auto}
used = sorted(used, key=lambda c: (int(c) not in paintable, sort_key(c)))   # números 1..n solo para colores que se pintan
cnum = {int(c): i for i, c in enumerate(used)}
palette = ['#%02x%02x%02x' % tuple(pal[c]) for c in used]

dist = radii()
N = len(ids); rid = np.arange(1, N + 1)
pos = ndi.maximum_position(dist, R, rid); rr = ndi.maximum(dist, R, rid)
regions = [[cnum[int(reg_color[i])], int(pos[i][1]), int(pos[i][0]), round(float(rr[i]), 1)] + ([1] if int(ids[i]) in auto else []) for i in range(N)]
print(f'zonas con número: {N - sum(1 for r in regions if len(r) > 4)}, ya pintadas: {sum(1 for r in regions if len(r) > 4)}')
print(f'zonas finales: {N}, colores: {len(palette)}')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app', 'catalog', a.name)
os.makedirs(out, exist_ok=True)
enc = np.zeros((H, W, 3), np.uint8); enc[..., 0] = R >> 8; enc[..., 1] = R & 255
Image.fromarray(enc).save(os.path.join(out, 'regions.png'), optimize=True)
json.dump({'w': W, 'h': H, 'line': '#3a3037', 'palette': palette, 'regions': regions}, open(os.path.join(out, 'data.json'), 'w'), separators=(',', ':'))

# miniaturas: terminada y solo contornos
lut = np.zeros((N + 1, 3), np.uint8); lut[0] = [58, 48, 55]
for i, (c, *_ ) in enumerate(regions): lut[i + 1] = [int(palette[c][k:k + 2], 16) for k in (1, 3, 5)]
final = lut[R]
edge = np.zeros((H, W), bool); edge[:, 1:] |= R[:, 1:] != R[:, :-1]; edge[1:, :] |= R[1:, :] != R[:-1, :]
t = 360 / max(W, H); ts = (round(W * t), round(H * t))
Image.fromarray(final).resize(ts, Image.LANCZOS).save(os.path.join(out, 'thumb.jpg'), quality=85)
lineimg = np.full((H, W), 255, np.uint8); lineimg[edge] = 150; lineimg[R == 0] = 60
Image.fromarray(lineimg).resize(ts, Image.LANCZOS).save(os.path.join(out, 'line.png'), optimize=True)
print('listo:', out)
