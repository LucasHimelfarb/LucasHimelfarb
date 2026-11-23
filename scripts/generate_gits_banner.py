#!/usr/bin/env python3
"""Genera assets/lucas-himelfarb-banner-v3.svg: una escena pixel art cyberpunk (Ghost in the
Shell / Blade Runner) con tu avatar integrado como personaje en primer plano.

Uso:
    pip install pillow numpy
    python3 scripts/generate_gits_banner.py

Entrada: assets/avatar-cutout.png (avatar con el fondo removido, RGBA).
Las capas estáticas se rasterizan en numpy y se incrustan como PNG; las
animaciones (llamaradas, reflectores, autos voladores, niebla, lluvia, carteles,
luz sobre el personaje) se hacen con SVG + CSS.
"""
import base64
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H = 320, 128           # resolución "pixel art"
OUT_W, OUT_H = 800, 320   # tamaño mostrado
UP = 2                    # factor de escalado (nearest) de los PNG incrustados
SEED = 2019
rng = np.random.default_rng(SEED)
random.seed(SEED)

css, body, defs = ["image{image-rendering:pixelated}"], [], []
uid = [0]


def nid(prefix):
    uid[0] += 1
    return f"{prefix}{uid[0]}"


# ============================================================ utilidades raster
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16) - .5


def bayer(h, w):
    return np.tile(BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w]


def noise(w, h, gw, gh, seed, tile=False):
    r = np.random.default_rng(seed).random((gh, gw))
    if tile:
        r = np.tile(r, (1, 3))
        im = Image.fromarray((r * 255).astype(np.uint8)).resize((w * 3, h), Image.BICUBIC)
        return np.asarray(im, dtype=float)[:, w:2 * w] / 255
    return np.asarray(Image.fromarray((r * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=float) / 255


def fbm(w, h, seed, base=6, octaves=4, tile=False):
    out, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        gw, gh = base * 2 ** o, max(2, base * 2 ** o * h // w)
        out += amp * noise(w, h, gw, gh, seed + o, tile)
        tot += amp
        amp *= .5
    return out / tot


def lerp(a, b, t):
    return a + (b - a) * t


def ramp(t, stops):
    """t: array 0..1, stops: [(pos,(r,g,b)), ...] -> array (...,3)."""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (t >= p0) & (t <= p1)
        k = ((t - p0) / (p1 - p0))[..., None]
        out[m] = (np.array(c0) + (np.array(c1) - np.array(c0)) * k)[m]
    return out


def posterize(rgb, colors=48, strength=14):
    """Dithering ordenado + paleta adaptativa => look pixel art coherente."""
    h, w, _ = rgb.shape
    x = np.clip(rgb + bayer(h, w)[..., None] * strength, 0, 255).astype(np.uint8)
    q = Image.fromarray(x, "RGB").quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return np.asarray(q.convert("RGB"), dtype=float)


def image(rgba, x=0, y=0, attrs=""):
    """Incrusta un array RGBA (uint8) recortado a su bbox como <image>."""
    a = np.asarray(rgba, dtype=np.uint8)
    ys, xs = np.nonzero(a[..., 3])
    if len(xs) == 0:
        return ""
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = Image.fromarray(a[y0:y1, x0:x1], "RGBA")
    crop = crop.resize((crop.width * UP, crop.height * UP), Image.NEAREST)
    buf = io.BytesIO()
    crop.save(buf, "PNG", optimize=True)
    uri = base64.b64encode(buf.getvalue()).decode()
    return (f'<image x="{x + x0}" y="{y + y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'preserveAspectRatio="none" image-rendering="optimizeSpeed" '
            f'href="data:image/png;base64,{uri}"{attrs}/>')


def rgba(rgb, alpha):
    return np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 255)]).astype(np.uint8)


def rect(x, y, w, h, fill, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{extra}/>'


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


# fuente 3x5 para textos chicos (HUD, cartel)
MINI = {
 "A": ["010", "101", "111", "101", "101"], "B": ["110", "101", "110", "101", "110"],
 "C": ["011", "100", "100", "100", "011"], "D": ["110", "101", "101", "101", "110"],
 "E": ["111", "100", "110", "100", "111"], "F": ["111", "100", "110", "100", "100"],
 "G": ["011", "100", "101", "101", "011"], "H": ["101", "101", "111", "101", "101"],
 "I": ["111", "010", "010", "010", "111"], "J": ["001", "001", "001", "101", "010"],
 "K": ["101", "101", "110", "101", "101"], "L": ["100", "100", "100", "100", "111"],
 "M": ["101", "111", "111", "101", "101"], "N": ["110", "101", "101", "101", "101"],
 "O": ["010", "101", "101", "101", "010"], "P": ["110", "101", "110", "100", "100"],
 "Q": ["010", "101", "101", "110", "011"], "R": ["110", "101", "110", "101", "101"],
 "S": ["011", "100", "010", "001", "110"], "T": ["111", "010", "010", "010", "010"],
 "U": ["101", "101", "101", "101", "111"], "V": ["101", "101", "101", "101", "010"],
 "W": ["101", "101", "111", "111", "101"], "X": ["101", "101", "010", "101", "101"],
 "Y": ["101", "101", "010", "010", "010"], "Z": ["111", "001", "010", "100", "111"],
 "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
 "2": ["110", "001", "010", "100", "111"], "3": ["110", "001", "010", "001", "110"],
 "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "110", "001", "110"],
 "6": ["011", "100", "111", "101", "111"], "7": ["111", "001", "010", "010", "010"],
 "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "110"],
 " ": ["000"] * 5, ".": ["000", "000", "000", "000", "010"], ",": ["000", "000", "000", "010", "100"],
 ":": ["000", "010", "000", "010", "000"], "/": ["001", "001", "010", "100", "100"],
 ">": ["100", "010", "001", "010", "100"], "+": ["000", "010", "111", "010", "000"],
 "-": ["000", "000", "111", "000", "000"], "(": ["010", "100", "100", "100", "010"],
 ")": ["010", "001", "001", "001", "010"], "_": ["000", "000", "000", "000", "111"],
 "'": ["010", "010", "000", "000", "000"], "!": ["010", "010", "010", "000", "010"],
}


def mini_px(text, x0, y0):
    return {(x0 + i * 4 + x, y0 + y) for i, ch in enumerate(text)
            for y, r in enumerate(MINI[ch]) for x, b in enumerate(r) if b == "1"}


yy, xx = np.mgrid[0:H, 0:W]
HORIZON = 98

# ============================================================ 1. cielo + smog
sky = ramp(yy / H, [(0, (5, 6, 14)), (.28, (14, 13, 30)), (.5, (44, 24, 44)),
                    (.68, (128, 58, 46)), (.78, (196, 98, 52)), (1, (120, 52, 40))])
clouds = fbm(W, H, 11, base=5)
band = np.exp(-((yy - 52) / 26) ** 2)
sky += (clouds[..., None] - .45) * band[..., None] * np.array([120, 60, 50])
sky += np.clip(clouds - .55, 0, 1)[..., None] * (yy < 40)[..., None] * np.array([40, 44, 70])


def haze_col(y):
    return ramp(np.clip((y - 30) / (H - 30), 0, 1), [(0, (40, 26, 44)), (.6, (150, 72, 48)), (1, (110, 48, 40))])


# ============================================================ siluetas futuristas
def silhouette(kind, x, bw, top, flip=False):
    """Máscara (H,W) de un edificio con forma no rectangular."""
    m = np.zeros((H, W), bool)
    bh = H - top
    for r in range(bh):
        y = top + r
        L, R = 0, bw
        if kind == "taper":            # se afina hacia arriba (trapecio) + aguja
            k = max(0, 1 - r / max(1, bh * .55))
            ins = round(bw * .32 * k)
            L, R = ins, bw - ins
        elif kind == "stepped":        # escalonado en 3 niveles
            tier = min(3, r // max(2, bh // 7))
            ins = round((3 - tier) * bw / 8)
            L, R = ins, bw - ins
        elif kind == "slant":          # techo inclinado de un lado
            s_ = max(2, bw * 2 // 3)
            if r < s_:
                R = max(2, round(bw * (r + 1) / s_))
        elif kind == "chamfer":        # esquinas cortadas a 45°
            c = max(2, bw // 3)
            if r < c:
                L, R = c - r, bw - (c - r)
        elif kind == "needle":         # torre fina con cuerpo superior más angosto
            if r < bh * .4:
                ins = max(1, bw // 2 - 2)
                L, R = ins, bw - ins
        if flip:
            L, R = bw - R, bw - L
        xa, xb = max(0, x + L), min(W, x + R)
        if xb > xa:
            m[y, xa:xb] = True
    if kind in ("taper", "needle"):  # aguja
        cx = x + bw // 2
        if 0 <= cx < W:
            m[max(0, top - random.randint(4, 9)):top, cx] = True
    return m


KINDS = ["taper", "stepped", "slant", "chamfer", "needle"]

# ============================================================ 2. megaestructuras lejanas
far = np.zeros((H, W)); far_lights = np.zeros((H, W))
stacks = []
x = -4
while x < W:
    bw = random.randint(6, 16)
    top = random.randint(42, 86)
    kind = random.choice(KINDS)
    if random.random() < .14:
        top, kind, bw = random.randint(12, 28), random.choice(["taper", "needle", "stepped"]), random.randint(10, 18)
    m = silhouette(kind, x, bw, top, flip=random.random() < .5)
    m[HORIZON + 30:] = False
    far[m] = 1
    lit = m & (np.roll(m, 1, 1)) & (np.roll(m, -1, 1)) & (yy % 2 == 0) & (yy < HORIZON)
    far_lights[lit & (rng.random((H, W)) < .06)] = rng.uniform(.4, 1)
    x += bw + random.randint(-3, 2)
for sx in [24, 66, 118]:
    far[HORIZON - 14:HORIZON + 30, sx:sx + 2] = 1  # chimeneas de las llamaradas
    stacks.append(sx)
stacks = sorted(set([24, 66, 118]))

far_col = lerp(np.array([30, 20, 34]), haze_col(yy), .55)
scene = np.where(far[..., None] > 0, far_col, sky)
lights_col = np.array([255, 170, 90])
scene = np.where(far_lights[..., None] > 0, lerp(far_col, lights_col, far_lights[..., None] * .7), scene)

# resplandor del horizonte sobre todo
glow = np.exp(-((yy - HORIZON) / 18) ** 2)[..., None] * np.array([70, 30, 12])
scene = scene + glow * .6
layer_bg = posterize(scene, 40)
body.append(image(rgba(layer_bg, np.full((H, W), 255))))

# ============================================================ 3. llamaradas (animadas)
css.append("@keyframes flare{0%,100%{transform:scaleY(.2);opacity:.2}8%{transform:scaleY(1);opacity:1}"
           "18%{transform:scaleY(.7);opacity:.9}30%{transform:scaleY(.1);opacity:0}}"
           "@keyframes fglow{0%,100%{opacity:.05}8%{opacity:.45}25%{opacity:.1}}")
defs.append('<radialGradient id="fg"><stop offset="0" stop-color="#ffb060" stop-opacity=".9"/>'
            '<stop offset="1" stop-color="#ff6020" stop-opacity="0"/></radialGradient>')
for i, sx in enumerate(stacks):
    dur, dl = random.uniform(5, 8), random.uniform(0, 5)
    top = HORIZON - 14
    body.append(f'<ellipse cx="{sx + 1}" cy="{top - 4}" rx="16" ry="12" fill="url(#fg)" '
                f'style="animation:fglow {dur:.1f}s {dl:.1f}s infinite"/>')
    flame = (rect(sx, top - 3, 2, 3, "#ffe7a0") + rect(sx - 1, top - 7, 4, 4, "#ffb050")
             + rect(sx, top - 11, 2, 4, "#ff7a30") + rect(sx + 1, top - 13, 1, 2, "#e04a20"))
    body.append(f'<g style="transform-origin:{sx + 1}px {top}px;animation:flare {dur:.1f}s {dl:.1f}s infinite">{flame}</g>')

# ============================================================ 4. reflectores desde la ciudad
defs.append('<linearGradient id="sl" x1="0" y1="1" x2="0" y2="0">'
            '<stop offset="0" stop-color="#d8e8ff" stop-opacity=".22"/>'
            '<stop offset="1" stop-color="#d8e8ff" stop-opacity="0"/></linearGradient>')
for i, (ox, d, a0, a1) in enumerate([(52, 11, -28, 10), (138, 14, 18, -16), (206, 9, -10, 22)]):
    k = nid("sw")
    css.append(f"@keyframes {k}{{0%,100%{{transform:rotate({a0}deg)}}50%{{transform:rotate({a1}deg)}}}}")
    body.append(f'<g style="transform-origin:{ox}px 110px;animation:{k} {d}s infinite ease-in-out">'
                f'<polygon points="{ox - 1},110 {ox + 1},110 {ox + 12},-20 {ox - 12},-20" fill="url(#sl)"/></g>')

# ============================================================ autos voladores policiales
CAR = [
    "..............rrbb..........",
    "..........hhhhhhhhh.........",
    "........hhgggggggghhh.......",
    ".....hhhbggggggggggbbbhh....",
    "..hhbbbbbbbbbbbbbbbbbbbbbhh.",
    "lbbbbwwwwwwwwwwwwwwwwwbbbbbH",
    ".bbbbbbbbbbbbbbbbbbbbbbbbbb.",
    "...dddd..............dddd...",
]
CPAL = {"b": "#1a1d26", "h": "#434b5c", "w": "#7a8496", "g": "#3f8ea3", "d": "#0b0c11"}
HW = len(CAR[0])
css.append("@keyframes red{0%,49%{opacity:1}50%,100%{opacity:.08}}@keyframes blu{0%,49%{opacity:.08}50%,100%{opacity:1}}"
           "@keyframes thr{0%,100%{opacity:.95}30%{opacity:.55}60%{opacity:1}80%{opacity:.7}}"
           "@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(1.5px)}}")
defs.append('<linearGradient id="hl" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0" stop-color="#f4f8ff" stop-opacity=".4"/>'
            '<stop offset="1" stop-color="#f4f8ff" stop-opacity="0"/></linearGradient>'
            '<radialGradient id="tg"><stop offset="0" stop-color="#ff9a50" stop-opacity=".8"/>'
            '<stop offset="1" stop-color="#ff5a20" stop-opacity="0"/></radialGradient>')


def runs(px):
    rows = {}
    for x, y in px:
        rows.setdefault(y, []).append(x)
    out = []
    for y, xs in sorted(rows.items()):
        xs.sort(); s = p = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == p + 1:
                p = x; continue
            out.append(f'<rect x="{s}" y="{y}" width="{p - s + 1}" height="1"/>')
            if x is not None: s = p = x
    return "".join(out)


def flying_car():
    s = '<polygon points="27,5 28,6 70,22 70,-6" fill="url(#hl)"/>'          # faros
    for px in (3, 22):                                                    # brillo de propulsores
        s += (f'<g style="animation:thr .35s {px / 50:.2f}s infinite">'
              f'<ellipse cx="{px + 2}" cy="9" rx="5" ry="3" fill="url(#tg)"/>'
              + rect(px, 8, 4, 1, "#ffd28a") + rect(px + 1, 9, 2, 1, "#ff8a3a") + "</g>")
    s += "".join(f'<g fill="{c}">{runs({(x, y) for y, r in enumerate(CAR) for x, ch in enumerate(r) if ch == k})}</g>'
                 for k, c in CPAL.items())
    s += rect(27, 5, 1, 1, "#f4f8ff")                                    # faro
    s += rect(0, 5, 1, 1, "#ff3048")                                     # luz trasera
    s += rect(14, 0, 2, 1, "#ff3048", ' style="animation:red .4s infinite steps(1)"')
    s += rect(16, 0, 2, 1, "#3d6bff", ' style="animation:blu .4s infinite steps(1)"')
    return s


def flight(y, scale, dur, delay, ltr, op):
    k = nid("fly")
    a, b = (-HW * scale - 10, W + 10) if ltr else (W + 10, -HW * scale - 10)
    css.append(f"@keyframes {k}{{from{{transform:translate({a}px,{y}px)}}to{{transform:translate({b}px,{y}px)}}}}")
    flip = "" if ltr else f"translate({HW * scale} 0) scale(-1 1) "
    return (f'<g style="transform:translate({a}px,{y}px);animation:{k} {dur}s {delay}s infinite linear">'
            f'<g style="animation:bob 2s infinite ease-in-out"><g transform="{flip}scale({scale})" opacity="{op}">'
            f'{flying_car()}</g></g></g>')


# tráfico aéreo lejano: pares de luces cruzando en carriles
for lane, (ly, n, dur) in enumerate([(30, 5, 34), (44, 6, 28), (58, 4, 22)]):
    for j in range(n):
        ltr = (lane + j) % 2 == 0
        k = nid("tr")
        a, b = (-4, W + 4) if ltr else (W + 4, -4)
        dy = random.randint(-2, 2)
        css.append(f"@keyframes {k}{{from{{transform:translate({a}px,{ly + dy}px)}}to{{transform:translate({b}px,{ly + dy}px)}}}}")
        col = "#ffe2b0" if ltr else "#ff4a5a"
        body.append(f'<g opacity="{.35 + lane * .15:.2f}" style="transform:translate({a}px,{ly}px);'
                    f'animation:{k} {dur * random.uniform(.8, 1.3):.1f}s {-random.uniform(0, dur):.1f}s infinite linear">'
                    f'{rect(0, 0, 2, 1, col)}</g>')
body.append(flight(36, .35, 40, 0, True, .7))    # auto volador muy lejano

# ============================================================ 5. niebla lejana (deriva)
def fog_layer(y0, y1, seed, color, maxa, dur, rev=False):
    n = fbm(W, H, seed, base=4, octaves=3, tile=True)
    prof = np.clip(1 - np.abs((yy - (y0 + y1) / 2) / ((y1 - y0) / 2)), 0, 1) ** 1.5
    a = np.clip((n - .35) * 2.2, 0, 1) * prof
    a = np.round(a * 4) / 4 * maxa  # alfa escalonado (look pixel)
    a = np.where(bayer(H, W) + .5 < (a / maxa) * 1.2, a, a * .5)
    layer = rgba(np.broadcast_to(np.array(color, float), (H, W, 3)), a * 255)
    k = nid("fog")
    frm, to = (-W, 0) if rev else (0, -W)
    css.append(f"@keyframes {k}{{from{{transform:translateX({frm}px)}}to{{transform:translateX({to}px)}}}}")
    img = image(layer)
    img2 = image(layer, x=W)
    return f'<g style="animation:{k} {dur}s infinite linear">{img}{img2}</g>'


body.append(fog_layer(70, 112, 31, (170, 92, 70), .35, 90))

# ============================================================ 6. edificios medios
mid = np.zeros((H, W)); mid_rgb = np.zeros((H, W, 3)); flick = np.zeros((H, W, 4))
WIN_WARM, WIN_CYAN = np.array([230, 150, 80]), np.array([120, 210, 230])


def paint_building(m, base, light_dir_split=None):
    """Pinta una silueta: sombreado desde la izquierda, bordes iluminados y ventanas."""
    rows = np.nonzero(m.any(1))[0]
    for y in rows:
        xs = np.nonzero(m[y])[0]
        x0, x1 = xs.min(), xs.max()
        if light_dir_split is None:
            t = (xs - x0) / max(1, x1 - x0)
            shade = 1.25 - .45 * t
        else:  # pirámide: cara izquierda iluminada, derecha en sombra
            shade = np.where(xs < light_dir_split, 1.3, .72)
        mid_rgb[y, xs] = base * shade[:, None]
        if y % 3 == 2:
            mid_rgb[y, xs] *= .8                                         # líneas de pisos
    up = m & ~np.roll(m, 1, 0)
    left = m & ~np.roll(m, 1, 1)
    diag = m & ~np.roll(np.roll(m, 1, 0), 1, 1) & ~up
    mid_rgb[left] = base + np.array([28, 16, 18])
    mid_rgb[up | diag] = base + np.array([72, 36, 30])                   # bordes con luz del horizonte
    mid[m] = 1


x = -3
PYR_C, PYR_TOP, PYR_HALF0 = 146, 30, 4
while x < 250:
    bw = random.randint(9, 22)
    top = random.randint(52, 96)
    if x + bw > PYR_C - 26 and x < PYR_C + 26:   # delante de la pirámide: solo edificios bajos
        top = max(top, 88)
    kind = random.choice(KINDS)
    base = np.array(random.choice([(14, 12, 20), (18, 15, 24), (12, 11, 17)]), float)
    m = silhouette(kind, x, bw, top, flip=random.random() < .5)
    paint_building(m, base)
    ys, xs = np.nonzero(m)
    if random.random() < .35 and len(xs):                                # tira de luz vertical
        sx = int(np.median(xs))
        col = random.choice([(70, 170, 190), (190, 70, 140), (200, 140, 70)])
        col_rows = np.nonzero(m[:, sx])[0]
        sel = col_rows[(col_rows > col_rows.min() + 3) & (col_rows < H - 6)]
        mid_rgb[sel, sx] = lerp(base, np.array(col), .55)
    inner = m & np.roll(m, 1, 1) & np.roll(m, -1, 1) & np.roll(m, 1, 0) & np.roll(m, -2, 0)
    for wy, wx in zip(*np.nonzero(inner & (yy % 3 == 0) & (xx % 2 == 0))):
        r = random.random()
        if r < .11:
            mid_rgb[wy, wx] = lerp(base, WIN_WARM, random.uniform(.35, .85))
        elif r < .125:
            mid_rgb[wy, wx] = lerp(base, WIN_CYAN, .6)
        elif r < .14:
            flick[wy, wx] = (235, 170, 100, 255)
    x += bw + random.randint(0, 3)

# --- arcología piramidal (detrás de los edificios bajos, delante del resto)
pyr = np.zeros((H, W), bool)
for y in range(PYR_TOP, H):
    step = (y - PYR_TOP) // 5
    half = PYR_HALF0 + step * 3 + (y - PYR_TOP) % 5 // 3
    pyr[y, max(0, PYR_C - half):min(W, PYR_C + half)] = True
pyr[PYR_TOP - 12:PYR_TOP, PYR_C - 1:PYR_C + 1] = True                  # aguja
pyr[PYR_TOP - 4:PYR_TOP, PYR_C - 2:PYR_C + 2] = True
front = mid.astype(bool) & (yy >= 88)                                   # lo que queda delante
pbase = np.array([20, 16, 26], float)
saved = mid_rgb.copy(); saved_mid = mid.copy()
paint_building(pyr, pbase, light_dir_split=PYR_C)
for y in range(PYR_TOP, H):                                             # terrazas iluminadas
    if (y - PYR_TOP) % 5 == 0:
        xs = np.nonzero(pyr[y])[0]
        mid_rgb[y, xs] = lerp(mid_rgb[y, xs], np.array([235, 140, 70]), np.where(xs < PYR_C, .55, .3)[:, None])
    elif (y - PYR_TOP) % 5 == 2:                                        # bandas de ventanas
        for wx in np.nonzero(pyr[y])[0][1:-1:2]:
            r = random.random()
            if r < .3:
                mid_rgb[y, wx] = lerp(mid_rgb[y, wx], WIN_WARM, random.uniform(.4, .8))
            elif r < .34:
                flick[y, wx] = (235, 170, 100, 255)
mid_rgb[PYR_TOP:H, PYR_C] = lerp(mid_rgb[PYR_TOP:H, PYR_C], np.array([120, 70, 60]), .5)  # arista central
mid_rgb[front] = saved[front]; mid[front] = saved_mid[front]
flick[front] = 0
BILL = (PYR_C - 9, 52, 18, 30)
bx_, by_ = BILL[0], BILL[1]
flick[by_ - 1:by_ + BILL[3] + 1, bx_ - 1:bx_ + BILL[2] + 1] = 0

# bruma sobre la parte baja de los edificios
hz = np.clip((yy - 88) / 40, 0, 1)[..., None]
mid_rgb = lerp(mid_rgb, np.array([90, 46, 44]), hz * .55)
layer_mid = posterize(mid_rgb, 32, strength=8)
body.append(image(rgba(layer_mid, mid * 255)))
css.append("@keyframes wfl{0%,100%{opacity:1}40%{opacity:1}41%{opacity:.1}63%{opacity:.1}64%{opacity:1}}")
body.append(image(flick.astype(np.uint8), attrs=' style="animation:wfl 11s infinite steps(1)"'))

# ============================================================ 7. cartel holográfico (3 pantallas)
CYC = 20  # 4 pantallas x 5 s
SCREENS = [((255, 70, 170), (60, 10, 50)), ((70, 220, 255), (8, 30, 50)), ((255, 170, 70), (60, 24, 8)),
           ((170, 240, 60), (14, 40, 10))]
NS = len(SCREENS); SP = 100 / NS
bx, by, bw_, bh_ = BILL
for i, (hi, lo) in enumerate(SCREENS):
    g = np.zeros((bh_, bw_, 3))
    t = np.linspace(0, 1, bh_)[:, None]
    g[:] = lerp(np.array(lo), np.array(hi), (.25 + .5 * t * (i != 1) + .5 * (1 - t) * (i == 1))[..., None] * .7)
    rr = np.random.default_rng(100 + i)
    if i == 0:   # columnas de glifos
        for cx in range(2, bw_ - 2, 4):
            for cy in range(2, bh_ - 2, 3):
                if rr.random() < .7:
                    w_ = rr.integers(1, 3)
                    g[cy:cy + 2, cx:cx + w_] = (255, 220, 240)
    elif i == 1:  # ojo / iris
        cy, cx = bh_ // 2, bw_ // 2
        m = ((np.mgrid[0:bh_, 0:bw_][0] - cy) / 7) ** 2 + ((np.mgrid[0:bh_, 0:bw_][1] - cx) / 8) ** 2
        g[m < 1] = (200, 250, 255); g[m < .45] = (20, 90, 120); g[m < .12] = (5, 10, 20)
    elif i == 2:  # franjas + bloque de texto
        g[::4] = lerp(g[::4], np.array([255, 230, 180]), .5)
        g[bh_ // 2 - 4:bh_ // 2 + 4, 3:bw_ - 3] = (255, 236, 200)
        for cx in range(4, bw_ - 4, 3):
            g[bh_ // 2 - 2:bh_ // 2 + 2, cx] = (90, 30, 10)
    else:        # Limao: rodaja de lima + "IA"
        gy, gx = np.mgrid[0:bh_, 0:bw_]
        cy, cx = 11, bw_ / 2 - .5
        d = np.hypot(gy - cy, gx - cx)
        ang = np.arctan2(gy - cy, gx - cx)
        g[d < 8] = (60, 140, 30); g[d < 7] = (190, 245, 90); g[d < 6] = (225, 255, 150)
        seg = (np.abs(np.sin(ang * 3.5)) < .22) & (d < 6) & (d > 1.2)
        g[seg] = (190, 245, 90)
        g[d < 1.2] = (240, 255, 200)
        for ch, ox in (("I", 5), ("A", 9)):
            for ry, row in enumerate(MINI[ch]):
                for rx, b in enumerate(row):
                    if b == "1":
                        g[22 + ry, ox + rx] = (225, 255, 150)
    g[::2] *= .82  # scanlines
    full = np.zeros((H, W, 4))
    full[by:by + bh_, bx:bx + bw_, :3] = posterize(g, 12, 10)
    full[by:by + bh_, bx:bx + bw_, 3] = 255
    k = nid("scr")
    p0, p1 = i * SP, (i + 1) * SP
    css.append(f"@keyframes {k}{{0%,100%{{opacity:0}}{max(p0 - .5, 0):.2f}%{{opacity:0}}{p0 + .01:.2f}%{{opacity:1}}"
               f"{p0 + 1:.2f}%{{opacity:.3}}{p0 + 1.6:.2f}%{{opacity:1}}{p1 - .01:.2f}%{{opacity:1}}{p1:.2f}%{{opacity:0}}}}"
               if i else
               f"@keyframes {k}{{0%{{opacity:1}}{SP - .01:.2f}%{{opacity:1}}{SP:.2f}%,99.99%{{opacity:0}}100%{{opacity:1}}}}")
    # halo de luz del cartel
    defs.append(f'<radialGradient id="bh{i}"><stop offset="0" stop-color="{hexc(hi)}" stop-opacity=".38"/>'
                f'<stop offset="1" stop-color="{hexc(hi)}" stop-opacity="0"/></radialGradient>')
    body.append(f'<g style="animation:{k} {CYC}s infinite steps(1)">'
                f'<ellipse cx="{bx + bw_ / 2}" cy="{by + bh_ / 2}" rx="38" ry="34" fill="url(#bh{i})"/>'
                f'{image(full.astype(np.uint8))}</g>')
body.append(rect(bx - 1, by - 1, bw_ + 2, 1, "#0c0b12") + rect(bx - 1, by + bh_, bw_ + 2, 1, "#0c0b12"))

body.append(rect(PYR_C - 1, PYR_TOP - 13, 2, 1, "#ff3048", ' style="animation:red 2.2s infinite steps(1)"'))
body.append(flight(16, .55, 26, 4, False, .9))   # auto volador medio
body.append(fog_layer(96, 128, 47, (120, 64, 64), .3, 60, rev=True))

body.append(flight(12, .8, 19, 10, True, 1))      # auto volador cercano (pasa detrás del personaje)

# ============================================================ 8. azotea en primer plano
near = np.zeros((H, W)); near_rgb = np.zeros((H, W, 3))
ROOF = 112
near[ROOF:, :] = 1
near_rgb[ROOF:, :] = (9, 9, 14)
near_rgb[ROOF, :] = (74, 40, 40)            # borde con luz del horizonte
near_rgb[ROOF + 1, :] = (30, 20, 26)
for px in range(4, 236, 14):                # baranda
    near[ROOF - 9:ROOF, px] = 1; near_rgb[ROOF - 9:ROOF, px] = (16, 14, 20)
near[ROOF - 9, 0:236] = 1; near_rgb[ROOF - 9, 0:236] = (52, 32, 36)
near[ROOF - 5, 0:236] = 1; near_rgb[ROOF - 5, 0:236] = (20, 16, 22)
# equipos de aire y antena en la azotea
near[ROOF - 7:ROOF, 196:212] = 1; near_rgb[ROOF - 7:ROOF, 196:212] = (14, 13, 19)
near_rgb[ROOF - 7, 196:212] = (60, 34, 36)
near[ROOF - 26:ROOF, 34] = 1; near_rgb[ROOF - 26:ROOF, 34] = (14, 13, 19)
for ry in range(ROOF + 4, H, 3):            # charcos que reflejan
    for rx in range(0, 240, 5):
        if random.random() < .18:
            near_rgb[ry, rx:rx + random.randint(2, 6)] = random.choice([(50, 26, 40), (24, 46, 56), (70, 40, 30)])
body.append(image(rgba(near_rgb, near * 255)))
body.append(rect(34, ROOF - 27, 1, 1, "#ff3048", ' style="animation:red 1.8s infinite steps(1)"'))

# --- mate sobre el equipo de aire, con vapor
MATE = [
    ".....m",
    "....m.",
    ".yyym.",
    "ckkkkc",
    "cbbbbc",
    "bbbbbb",
    ".bbbb.",
    "..ss..",
]
MPAL = {"m": "#b9c2cc", "y": "#7aa63a", "c": "#8a5a3a", "k": "#3a2418", "b": "#5c3a26", "s": "#2a1a12"}
MX, MY = 200, ROOF - 7 - len(MATE)
body.append("".join(f'<g fill="{c}">{runs({(MX + x, MY + y) for y, r in enumerate(MATE) for x, ch in enumerate(r) if ch == k})}</g>'
                    for k, c in MPAL.items()))
body.append(rect(MX, MY + 3, 1, 1, "#c08060"))                       # brillo de la ciudad en la calabaza
css.append("@keyframes steam{0%{transform:translate(0,0);opacity:0}15%{opacity:.55}"
           "100%{transform:translate(-3px,-12px);opacity:0}}")
for j, (dx, dl) in enumerate([(1, 0), (3, 1.1), (2, 2.2)]):
    body.append(f'<rect x="{MX + dx}" y="{MY + 1}" width="1" height="2" fill="#e8e4f0" opacity="0" '
                f'style="animation:steam 3.3s {dl}s infinite steps(8)"/>')

# ============================================================ 9. el personaje (tu avatar)
cut = Image.open(ROOT / "assets" / "avatar-cutout.png").convert("RGBA")
cut = cut.crop(cut.getchannel("A").getbbox())                  # 287x460
CH = 122
CW = round(cut.width * CH / cut.height)
small = np.asarray(cut.resize((CW, CH), Image.LANCZOS), dtype=float)
CX, CY = W - CW, H - CH
alpha = (small[..., 3] > 120).astype(float)
rgb = small[..., :3]
lum = (rgb @ np.array([.3, .59, .11])) / 255
ly, lx = np.mgrid[0:CH, 0:CW]
# luz principal desde la izquierda (la ciudad) y caída hacia abajo / derecha
key = np.clip(1.18 - lx / CW * .7 - np.clip((ly - 70) / 60, 0, 1) * .55, .2, 1.1)
graded = ramp(lum * key, [(0, (6, 7, 12)), (.18, (22, 18, 30)), (.38, (70, 44, 56)),
                          (.6, (160, 96, 84)), (.85, (236, 178, 138)), (1, (255, 214, 176))])
graded = lerp(graded, rgb * key[..., None], .22)
graded = posterize(graded, 28, strength=10)
body.append(image(rgba(graded, alpha * 255), CX, CY))

# luz de borde (rim) y relleno del color del cartel, sincronizada
left_edge = np.zeros((CH, CW))
for d, w_ in [(1, 1.0), (2, .8), (3, .45), (5, .2)]:
    sh = np.zeros_like(alpha); sh[:, d:] = alpha[:, :-d]
    left_edge = np.maximum(left_edge, (alpha - sh > 0) * 0 + ((alpha > 0) & (np.roll(alpha, d, 1) == 0) | (lx < d) & (alpha > 0)) * w_)
top_edge = ((alpha > 0) & (np.roll(alpha, 1, 0) == 0)).astype(float) * .7
rim = np.clip(np.maximum(left_edge, top_edge) * (lum > .08) * (ly < 100) * np.clip(1.2 - lx / CW, 0, 1), 0, 1)
rim = np.round(rim * 3) / 3
for i, (hi, _) in enumerate(SCREENS):
    k = f"scr{[n for n in range(1, 100)][0]}"
    lay = rgba(np.broadcast_to(np.array(hi, float), (CH, CW, 3)), rim * 230)
    lid = nid("rim")
    p0, p1 = i * SP, (i + 1) * SP
    if i == 0:
        css.append(f"@keyframes {lid}{{0%,{SP - .01:.2f}%{{opacity:1}}{SP:.2f}%,99.99%{{opacity:0}}100%{{opacity:1}}}}")
    else:
        css.append(f"@keyframes {lid}{{0%,{p0 - .01:.2f}%{{opacity:0}}{p0:.2f}%,{p1 - .01:.2f}%{{opacity:1}}{p1:.2f}%,100%{{opacity:0}}}}")
    body.append(f'<g style="animation:{lid} {CYC}s infinite steps(1)">{image(lay, CX, CY)}</g>')

# reflejo de la ciudad en los anteojos
s = CH / cut.height
ox = (cut.getchannel("A").getbbox() if False else (173, 0))[0]
LX0, LY0, LX1, LY1 = [round(v) for v in ((214 - 173) * s, 176 * s, (318 - 173) * s, 226 * s)]
lens = np.zeros((CH, CW))
lens[LY0:LY1, LX0:LX1] = 1
lens *= (alpha > 0) * (rgb[..., 2] > rgb[..., 0] * .95) * (lum > .25)
css.append("@keyframes glint{0%,100%{transform:translateX(-14px)}50%{transform:translateX(14px)}}")
gl = np.zeros((CH, CW, 4))
for d in range(-30, 30):
    m = (lx - ly == d) | (lx - ly == d + 1)
    if d in (0, 1, 2):
        gl[m & (lens > 0)] = (255, 236, 220, 120)
lens_tint = rgba(np.broadcast_to(np.array([255, 150, 110], float), (CH, CW, 3)), lens * 40)
body.append(image(lens_tint, CX, CY))
clip = nid("lc")
defs.append(f'<clipPath id="{clip}"><rect x="{CX + LX0}" y="{CY + LY0}" width="{LX1 - LX0}" height="{LY1 - LY0}"/></clipPath>')
body.append(f'<g clip-path="url(#{clip})"><g style="animation:glint 6s infinite ease-in-out">{image(gl.astype(np.uint8), CX, CY)}</g></g>')


# ============================================================ 10. lluvia
css.append(f"@keyframes rain{{from{{transform:translate(0,0)}}to{{transform:translate(-8px,{H}px)}}}}")
for n, op, dur, ln in [(160, .16, 1.1, 3), (90, .26, .7, 5)]:
    drops = []
    for _ in range(n):
        x, y = random.randrange(0, W + 8), random.randrange(0, H)
        for dx, dy in [(0, 0), (8, -H)]:
            drops.append(f'<rect x="{x + dx}" y="{y + dy}" width="1" height="{ln}"/>')
    body.append(f'<g fill="#c8d4e8" opacity="{op}" style="animation:rain {dur}s infinite linear">{"".join(drops)}</g>')

# ============================================================ 11. viñeta
defs.append('<radialGradient id="vig" cx=".5" cy=".5" r=".75"><stop offset=".55" stop-color="#000" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#000" stop-opacity=".65"/></radialGradient>')
body.append(rect(0, 0, W, H, "url(#vig)"))
body.append(rect(0, 0, W, 6, "#000") + rect(0, H - 6, W, 6, "#000"))  # franjas cinemascope

# ============================================================ 12. texto: nombre
FONT = {
 "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
 "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
 "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
 "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
 "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
 "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
 "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
 "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
 "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
 "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
 "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
 "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
 "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
 "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
 "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
 "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
 " ": ["00000"] * 7,
}
TEXT, TX, TY, TC = "LUCAS HIMELFARB", 12, 14, "#d8f6f0"
TCYC = 30  # ciclo maestro: nombre + dossier HUD
css.append(f"@keyframes bl{{0%,49%{{opacity:1}}50%,100%{{opacity:0}}}}"
           f"@keyframes tglitch{{0%,62%,64%,100%{{transform:translateX(0)}}62.5%{{transform:translateX(2px)}}63.2%{{transform:translateX(-1px)}}}}")
chars = []
for i, ch in enumerate(TEXT):
    px = {(TX + i * 6 + x, TY + y) for y, r in enumerate(FONT[ch]) for x, b in enumerate(r) if b == "1"}
    if not px:
        continue
    on = (.4 + i * .17) / TCYC * 100
    k = f"t{i}"
    css.append(f"@keyframes {k}{{0%,{on:.2f}%{{opacity:0}}{on + .1:.2f}%,98.5%{{opacity:1}}98.6%,100%{{opacity:0}}}}")
    chars.append(f'<g style="animation:{k} {TCYC}s infinite">'
                 f'<g fill="#000" opacity=".6">{runs({(x + 1, y + 1) for x, y in px})}</g>'
                 f'<g fill="{TC}">{runs(px)}</g></g>')
body.append(f'<g style="animation:tglitch {TCYC}s infinite steps(1)">{"".join(chars)}</g>')
# indicador + subrayado fino
body.append(rect(TX - 6, TY + 2, 3, 3, "#ff3048", ' style="animation:bl 1.2s infinite steps(1)"'))
css.append(f"@keyframes uline{{0%,9%{{transform:scaleX(0)}}12%,98.5%{{transform:scaleX(1)}}98.6%,100%{{transform:scaleX(0)}}}}")
body.append(f'<rect x="{TX}" y="{TY + 10}" width="{len(TEXT) * 6 - 1}" height="1" fill="{TC}" opacity=".45" '
            f'style="transform-origin:{TX}px 0;animation:uline {TCYC}s infinite"/>')

# ============================================================ 13. dossier HUD: perfil como persona y profesional
# Cada tarjeta: (encabezado, [(etiqueta, valor), ...]). Editá acá para cambiar el contenido.
CARDS = [
    ("PERFIL 01/04", [("CLASE", "SOFTWARE ENGINEER"), ("GREMIO", "HUMAND"), ("BASE", "CORDOBA, ARGENTINA")]),
    ("SIDE QUEST 02/04", [("NOMBRE", "LIMAO"), ("MISION", "IA PARA PYMES"), ("ESTADO", "EN CONSTRUCCION")]),
    ("INVENTARIO 03/04", [("X1", "MATE CEBADO"), ("X1", "TECLADO MECANICO"), ("X99", "CURIOSIDAD")]),
    ("LOG 04/04", [("DESDE", "2019 EN GITHUB"), ("COPILOT", "CLAUDE"), ("STATUS", "ONLINE")]),
]
PX, PY, PW, PH = 8, 27, 116, 34
OPEN, CARD = 3.2, 6.6                     # s: apertura del panel y duración de cada tarjeta
CLOSE = OPEN + CARD * len(CARDS)          # 29.6 s
HC_HEAD, HC_LBL, HC_VAL, HC_ACC = "#ffb050", "#6f98a4", "#d8f6f0", "#ff3048"
pct = lambda t: t / TCYC * 100

css.append(f"@keyframes hudp{{0%,{pct(OPEN - .25):.2f}%{{transform:scaleY(0)}}{pct(OPEN):.2f}%,{pct(CLOSE - .25):.2f}%"
           f"{{transform:scaleY(1)}}{pct(CLOSE):.2f}%,100%{{transform:scaleY(0)}}}}"
           "@keyframes scan{from{transform:translateY(0)}to{transform:translateY(32px)}}")
panel = (rect(PX, PY, PW, PH, "#05060c", ' opacity=".62"')
         + rect(PX, PY, PW, 7, HC_HEAD, ' opacity=".14"')
         + f'<rect x="{PX + .5}" y="{PY + .5}" width="{PW - 1}" height="{PH - 1}" fill="none" stroke="{HC_VAL}" '
           f'stroke-width="1" opacity=".22" shape-rendering="crispEdges"/>')
for cx_, cy_, dx_, dy_ in [(PX, PY, 1, 1), (PX + PW - 1, PY, -1, 1), (PX, PY + PH - 1, 1, -1), (PX + PW - 1, PY + PH - 1, -1, -1)]:
    panel += f'<g fill="{HC_VAL}" opacity=".8">{runs({(cx_ + dx_ * i, cy_) for i in range(4)} | {(cx_, cy_ + dy_ * i) for i in range(4)})}</g>'
panel += (f'<g opacity=".07" style="animation:scan 2.4s infinite linear">{rect(PX + 1, PY + 1, PW - 2, 1, HC_VAL)}</g>')
cards_svg = []
for ci, (head, rows) in enumerate(CARDS):
    t0, t1 = OPEN + ci * CARD, OPEN + (ci + 1) * CARD
    kv = nid("card")
    css.append(f"@keyframes {kv}{{0%,{pct(t0):.2f}%{{opacity:0}}{pct(t0) + .01:.2f}%,{pct(t1) - .2:.2f}%{{opacity:1}}"
               f"{pct(t1) - .19:.2f}%,100%{{opacity:0}}}}")
    lines = [(f"> {head}", HC_HEAD, None)] + [(f"{lbl}", HC_LBL, val) for lbl, val in rows]
    out, t = [], t0 + .15
    for li, (txt, col, val) in enumerate(lines):
        ly_ = PY + 2 + li * 8 if li else PY + 1
        full = txt if val is None else f"{txt:<8}{val}"
        n = len(full)
        px_lbl = mini_px(txt, PX + 3, ly_)
        px_val = mini_px(val, PX + 3 + 8 * 4, ly_) if val else set()
        if ci == 3 and val == "ONLINE":
            out_dot = rect(PX + 3 + (8 + len(val)) * 4 + 1, ly_ + 1, 3, 3, "#58e07a",
                           ' style="animation:bl 1.2s infinite steps(1)"')
        else:
            out_dot = ""
        cid, ka = nid("tw"), nid("ty")
        dur = n * .035
        n += 2 if out_dot else 0
        css.append(f"@keyframes {ka}{{0%,{pct(t):.2f}%{{transform:scaleX(0)}}{pct(t + dur):.2f}%,100%{{transform:scaleX(1)}}}}")
        defs.append(f'<clipPath id="{cid}"><rect x="{PX + 3}" y="{ly_ - 1}" width="{n * 4 + 6}" height="7" '
                    f'style="transform-origin:{PX + 3}px 0;animation:{ka} {TCYC}s infinite steps({n})"/></clipPath>')
        g = f'<g fill="{col}">{runs(px_lbl)}</g>' + (f'<g fill="{HC_VAL}">{runs(px_val)}</g>' if val else "")
        out.append(f'<g clip-path="url(#{cid})"><g fill="#000" opacity=".5" transform="translate(1 1)">'
                   f'{runs(px_lbl | px_val)}</g>{g}{out_dot}</g>')
        t += dur + .12
    if ci == 1:   # barra de progreso del side quest
        kb = nid("bar")
        css.append(f"@keyframes {kb}{{0%,{pct(t):.2f}%{{transform:scaleX(0)}}{pct(t + 1.2):.2f}%,100%{{transform:scaleX(.42)}}}}")
        out.append(rect(PX + 3 + 8 * 4 + 64, PY + 2 + 3 * 8 + 1, 12, 3, HC_VAL, ' opacity=".2"')
                   + f'<rect x="{PX + 3 + 8 * 4 + 64}" y="{PY + 2 + 3 * 8 + 1}" width="12" height="3" fill="#aaf03c" '
                     f'style="transform-origin:{PX + 99}px 0;animation:{kb} {TCYC}s infinite"/>')
    cards_svg.append(f'<g opacity="0" style="animation:{kv} {TCYC}s infinite steps(1)">{"".join(out)}</g>')
# indicador de páginas
dots = "".join(rect(PX + PW - 4 - (len(CARDS) - 1 - i) * 3, PY + 2, 2, 2, HC_HEAD, ' opacity=".25"') for i in range(len(CARDS)))
for ci in range(len(CARDS)):
    t0, t1 = OPEN + ci * CARD, OPEN + (ci + 1) * CARD
    kd = nid("dot")
    css.append(f"@keyframes {kd}{{0%,{pct(t0):.2f}%{{opacity:0}}{pct(t0) + .01:.2f}%,{pct(t1) - .2:.2f}%{{opacity:1}}"
               f"{pct(t1) - .19:.2f}%,100%{{opacity:0}}}}")
    dots += rect(PX + PW - 4 - (len(CARDS) - 1 - ci) * 3, PY + 2, 2, 2, HC_HEAD, f' opacity="0" style="animation:{kd} {TCYC}s infinite steps(1)"')
body.append(f'<g style="transform-origin:0 {PY}px;transform:scaleY(0);animation:hudp {TCYC}s infinite">'
            f'{panel}{"".join(cards_svg)}{dots}</g>')

# ============================================================ salida
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{OUT_W}" height="{OUT_H}" '
       f'shape-rendering="crispEdges"><defs>{"".join(defs)}</defs><style>{"".join(css)}</style>'
       f'{rect(0, 0, W, H, "#05060c")}{"".join(body)}</svg>')
out = ROOT / "assets" / "lucas-himelfarb-banner-v3.svg"
out.write_text(svg)
print(f"{out} ({len(svg) // 1024} KB)")
