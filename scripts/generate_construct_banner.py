#!/usr/bin/env python3
"""Genera assets/lucas-himelfarb-construct.svg: segundo banner del perfil.

Guiño a la escena del programa de entrenamiento: Claude (Morpheus) camina adelante
entre una multitud y le explica a Lucas (Neo) qué es su perfil. Se le cruza un
policía, aparece de repente una mujer de vestido rojo, Lucas se da vuelta a mirarla,
Claude dice "mirá de nuevo" y es un agente; con "congelalo" la calle se frena en
código verde mientras Claude termina de contar.

Uso:
    pip install pillow numpy
    python3 scripts/generate_construct_banner.py

Editá DIALOG para cambiar lo que se dice. Mismo formato que el banner principal
(320x128 escalado a 800x320, franjas cinemascope, viñeta, fuente 3x5).
"""
import base64
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
W, H = 320, 128
OUT_W, OUT_H = 800, 320
SEED = 1999
random.seed(SEED)
rng = np.random.default_rng(SEED)

T = 32.0                 # ciclo maestro (s)
F0, F1 = 20.0, 28.0      # congelado
WALK = T - (F1 - F0)     # tiempo efectivo caminando
NEAR_SPEED = 320 / WALK  # px/s de la fachada cercana (cámara que sigue a los protagonistas)

css = ["image{image-rendering:pixelated}"]
defs, body = [], []
uid = [0]


def nid(p):
    uid[0] += 1
    return f"{p}{uid[0]}"


def pct(t):
    return t / T * 100


# ============================================================ utilidades
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16) - .5


def bayer(h, w):
    return np.tile(BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w]


def posterize(rgb, colors=40, strength=10):
    h, w, _ = rgb.shape
    x = np.clip(rgb + bayer(h, w)[..., None] * strength, 0, 255).astype(np.uint8)
    q = Image.fromarray(x, "RGB").quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return np.asarray(q.convert("RGB"), dtype=float)


def png_uri(im):
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image(arr_or_im, x=0, y=0, attrs="", up=2):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(np.asarray(arr_or_im, np.uint8), "RGBA")
    bb = im.getchannel("A").getbbox()
    if not bb:
        return ""
    im = im.crop(bb)
    w, h = im.size
    im = im.resize((w * up, h * up), Image.NEAREST)
    return (f'<image x="{x + bb[0]}" y="{y + bb[1]}" width="{w}" height="{h}" preserveAspectRatio="none" '
            f'href="{png_uri(im)}"{attrs}/>')


def rect(x, y, w, h, fill, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{extra}/>'


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


def ramp(t, stops):
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (t >= p0) & (t <= p1)
        k = ((t - p0) / (p1 - p0))[..., None]
        out[m] = (np.array(c0) + (np.array(c1) - np.array(c0)) * k)[m]
    return out


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
 "-": ["000", "000", "111", "000", "000"], "?": ["110", "001", "010", "000", "010"],
 "!": ["010", "010", "010", "000", "010"], "'": ["010", "010", "000", "000", "000"],
}


def mini_px(text, x0, y0):
    return {(x0 + i * 4 + x, y0 + y) for i, ch in enumerate(text)
            for y, r in enumerate(MINI[ch]) for x, b in enumerate(r) if b == "1"}


def window(t0, t1, name=None, fade=0.0):
    """Keyframes de opacidad: visible solo entre t0 y t1 del ciclo."""
    k = name or nid("w")
    a, b = pct(t0), pct(t1)
    if fade:
        f = pct(fade)
        css.append(f"@keyframes {k}{{0%,{max(a - f, 0):.3f}%{{opacity:0}}{a:.3f}%,{max(b - f, a):.3f}%{{opacity:1}}"
                   f"{b:.3f}%,100%{{opacity:0}}}}")
    else:
        css.append(f"@keyframes {k}{{0%,{max(a - .001, 0):.3f}%{{opacity:0}}{a:.3f}%,{b - .001:.3f}%{{opacity:1}}"
                   f"{b:.3f}%,100%{{opacity:0}}}}")
    return f"animation:{k} {T}s infinite linear"


def eff_to_real(e):
    return e if e < F0 else e + (F1 - F0)


def track(x0, v, span, margin, y):
    """Movimiento horizontal que se detiene durante el congelado y da la vuelta sin saltos."""
    x0 = x0 % span
    pts = [(0.0, x0)]
    pos = x0
    e = 0.0
    events = []
    # vueltas (wraps) en tiempo efectivo
    while True:
        if v > 0:
            dt = (span - pos) / v
        else:
            dt = pos / -v
        if e + dt >= WALK - 1e-9:
            break
        e += dt
        events.append(e)
        pos = 0.0 if v > 0 else span
    for e in sorted(events + [F0]):
        p = (x0 + v * e) % span
        if e in events:
            end, start = (span, 0.0) if v > 0 else (0.0, span)
            r = eff_to_real(e)
            pts += [(r, end), (r + .002, start)]
        else:  # inicio y fin del congelado
            pts += [(F0, p), (F1, p)]
    pts.append((T, pos + v * (WALK - e)))
    pts.sort(key=lambda q: q[0])
    k = nid("tk")
    frames = "".join(f"{pct(t):.3f}%{{transform:translate({x - margin:.2f}px,{y}px)}}" for t, x in pts)
    css.append(f"@keyframes {k}{{{frames}}}")
    return f"transform:translate({x0 - margin:.2f}px,{y}px);animation:{k} {T}s infinite linear"


# ============================================================ fondo: ciudad del constructo (de día, verdosa)
yy, xx = np.mgrid[0:H, 0:W]
GROUND = 92
sky = ramp(yy / H, [(0, (150, 172, 150)), (.5, (196, 210, 186)), (1, (210, 218, 196))])
far = np.zeros((H, W), bool); far_rgb = sky.copy()
x = 0
while x < W:
    bw, top = random.randint(14, 34), random.randint(6, 40)
    far[top:GROUND, x:x + bw] = True
    c = np.array(random.choice([(120, 138, 122), (132, 148, 128), (110, 126, 112)]), float)
    far_rgb[top:GROUND, x:x + bw] = c
    far_rgb[top:GROUND, x] = c + 18
    for wy in range(top + 3, GROUND, 4):
        for wx in range(x + 2, min(W, x + bw - 2), 3):
            far_rgb[wy:wy + 2, wx] = c - 22 if random.random() < .85 else c + 30
    x += bw + random.randint(0, 2)
haze = np.clip((yy - 20) / 70, 0, 1)[..., None]
far_rgb = far_rgb * (1 - haze * .35) + np.array([200, 212, 190]) * haze * .35
body.append(image(Image.fromarray(np.dstack([posterize(far_rgb, 24, 8), np.full((H, W), 255)]).astype(np.uint8), "RGBA"), up=1))

# fachada cercana + vereda: un tile de 320 px que se repite y se desplaza
near = np.zeros((H, W, 4))
x = 0
FACADE_TOP = 18
while x < W:
    bw = random.randint(46, 78)
    bw = min(bw, W - x) if W - x < 40 else bw
    c = np.array(random.choice([(92, 104, 88), (104, 112, 94), (80, 92, 82), (112, 120, 106), (70, 80, 74)]), float)
    top = FACADE_TOP + random.randint(0, 14)
    near[top:GROUND, x:x + bw, :3] = c
    near[top:GROUND, x:x + bw, 3] = 255
    near[top:top + 2, x:x + bw, :3] = c + 26                         # cornisa
    near[top:GROUND, x:x + 1, :3] = c + 14
    near[top:GROUND, x + bw - 1:x + bw, :3] = c - 20
    for wy in range(top + 6, GROUND - 26, 10):                          # ventanas
        for wx in range(x + 5, x + bw - 7, 9):
            near[wy:wy + 6, wx:wx + 5, :3] = (48, 62, 56)
            near[wy:wy + 1, wx:wx + 5, :3] = (150, 170, 150)
            near[wy + 6, wx - 1:wx + 6, :3] = c + 20
    # planta baja: vidriera + puerta + toldo
    sy = GROUND - 22
    near[sy:GROUND, x + 3:x + bw - 3, :3] = (34, 46, 42)
    near[sy + 2:GROUND - 2, x + 5:x + bw - 5, :3] = (64, 86, 76)
    near[sy + 2:sy + 3, x + 5:x + bw - 5, :3] = (170, 196, 170)
    dx = x + bw // 2 - 4
    near[sy + 4:GROUND, dx:dx + 8, :3] = (28, 36, 32)
    if random.random() < .6:
        aw = np.array(random.choice([(40, 96, 62), (120, 128, 96), (60, 70, 66)]), float)
        for i in range(4):
            near[sy - 4 + i, x + 2 - i // 2:x + bw - 2 + i // 2, :3] = aw + (i % 2) * 18
            near[sy - 4 + i, x + 2 - i // 2:x + bw - 2 + i // 2, 3] = 255
    x += bw
# vereda
near[GROUND:, :, 3] = 255
near[GROUND:, :, :3] = ramp((yy[GROUND:] - GROUND) / (H - GROUND), [(0, (150, 160, 140)), (1, (118, 128, 112))])
near[GROUND, :, :3] = (60, 70, 62)
near[GROUND + 1, :, :3] = (176, 186, 164)
for lx in range(0, W, 22):                                              # juntas de baldosas
    for yrow in range(GROUND + 2, H):
        off = int((yrow - GROUND) * (lx - W / 2) / 260)
        if 0 <= lx + off < W:
            near[yrow, lx + off, :3] -= 22
for yrow in (GROUND + 7, GROUND + 17, GROUND + 30):
    if yrow < H:
        near[yrow, :, :3] -= 16
for _ in range(3):                                                      # postes / farolas
    px = random.randrange(10, W - 10)
    near[30:GROUND, px:px + 2, :3] = (40, 48, 44); near[30:GROUND, px:px + 2, 3] = 255
    near[30:32, px - 4:px + 6, :3] = (40, 48, 44); near[30:32, px - 4:px + 6, 3] = 255
near_rgb = posterize(np.clip(near[..., :3], 0, 255), 40, 8)
near_im = Image.fromarray(np.dstack([near_rgb, near[..., 3]]).astype(np.uint8), "RGBA")
tile = image(near_im, up=1) + image(near_im, x=-W, up=1)
body.append(f'<g style="{track(0, NEAR_SPEED, W, 0, 0)}">{tile}</g>')

# ============================================================ figuras (dibujadas con PIL)
SKIN = [(232, 196, 166), (200, 150, 116), (160, 112, 82), (110, 76, 56), (240, 210, 186)]
HAIR = [(30, 24, 20), (70, 50, 32), (110, 84, 56), (150, 140, 130), (20, 18, 18)]
SUIT = [(58, 64, 62), (44, 50, 60), (84, 80, 72), (70, 76, 70), (36, 40, 40), (96, 98, 92)]


def figure(h, frame, look, pose="walk"):
    """Persona de perfil mirando a la izquierda; ancla abajo-centro en (w/2, h)."""
    u = h / 24
    w = int(12 * u) + 4
    im = Image.new("RGBA", (w, h + 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w / 2
    coat, shirt, skin, hair = look["coat"], look.get("shirt", (200, 206, 196)), look["skin"], look["hair"]
    shoe = look.get("shoe", (18, 18, 18))
    pants = look.get("pants", tuple(max(0, c - 10) for c in coat))
    hip, foot = 12.5 * u, h - 1
    stride = {"walk": [2.4, 0.4][frame], "idle": 0.6}[pose] * u
    # piernas
    for sgn, shade in ((1, .8), (-1, 1.0)):                             # pierna de atrás primero
        col = tuple(int(c * shade) for c in pants)
        fx = cx - sgn * stride
        d.polygon([(cx - 1.1 * u, hip), (cx + 1.1 * u, hip), (fx + 1 * u, foot - u), (fx - 1 * u, foot - u)], fill=col)
        d.rectangle([fx - 1.6 * u, foot - u, fx + 0.8 * u, foot], fill=shoe)
    # torso / abrigo
    coat_len = look.get("coat_len", 13.5) * u
    d.polygon([(cx - 1.9 * u, 4.2 * u), (cx + 1.6 * u, 4.2 * u), (cx + 2.2 * u + (stride * .3 if coat_len > 14 * u else 0), coat_len),
               (cx - 2.4 * u - (stride * .5 if coat_len > 14 * u else 0), coat_len)], fill=coat)
    d.polygon([(cx - 1.9 * u, 4.4 * u), (cx - 0.9 * u, 4.4 * u), (cx - 1.4 * u, 7.5 * u)], fill=shirt)   # cuello
    if look.get("tie"):
        d.line([(cx - 1.5 * u, 5 * u), (cx - 1.7 * u, 8 * u)], fill=look["tie"], width=max(1, int(u * .6)))
    # brazo (balanceo opuesto a la pierna delantera)
    swing = {"walk": [-1.6, -0.2][frame], "idle": -0.2}[pose] * u
    arm = tuple(int(c * .85) for c in coat)
    d.line([(cx, 5 * u), (cx + swing, 10.8 * u)], fill=arm, width=max(2, int(1.5 * u)))
    d.ellipse([cx + swing - .7 * u, 10.6 * u, cx + swing + .7 * u, 11.8 * u], fill=skin)
    # cabeza
    hx, hy = cx - .3 * u, 2.2 * u
    d.rectangle([cx - .7 * u, 3.4 * u, cx + .3 * u, 4.4 * u], fill=skin)                                  # cuello
    d.ellipse([hx - 1.5 * u, hy - 1.9 * u, hx + 1.5 * u, hy + 1.7 * u], fill=skin)
    d.chord([hx - 1.6 * u, hy - 2.1 * u, hx + 1.7 * u, hy + 1.2 * u], 160, 360, fill=hair)
    d.rectangle([hx + .2 * u, hy - 1.4 * u, hx + 1.7 * u, hy + .6 * u], fill=hair)
    if look.get("curly"):
        for ox, oy in ((-1.4, -1.9), (-.4, -2.4), (.6, -2.2), (1.5, -1.5), (1.8, -.4)):
            d.ellipse([hx + (ox - .6) * u, hy + (oy - .5) * u, hx + (ox + .6) * u, hy + (oy + .5) * u], fill=hair)
    if look.get("shades"):
        d.rectangle([hx - 1.6 * u, hy - .5 * u, hx - .2 * u, hy + .2 * u], fill=look["shades"])
        d.line([(hx - .2 * u, hy - .4 * u), (hx + 1.2 * u, hy - .5 * u)], fill=look.get("frame", (20, 20, 20)), width=1)
    if look.get("beard"):
        d.rectangle([hx - 1.3 * u, hy + .6 * u, hx + .4 * u, hy + 1.5 * u], fill=look["beard"])
    if look.get("skirt"):
        sl = look.get("skirt_len", 16) * u
        d.polygon([(cx - 2 * u, 9 * u), (cx + 1.8 * u, 9 * u), (cx + 2.8 * u + stride * .2, sl), (cx - 3 * u - stride * .3, sl)],
                  fill=look["skirt"])
    if look.get("long_hair"):                                          # pelo largo cayendo por la espalda
        d.polygon([(hx - .2 * u, hy - 1.6 * u), (hx + 1.8 * u, hy - 1.2 * u), (hx + 2.2 * u, hy + 3.6 * u),
                   (hx + .4 * u, hy + 3.2 * u)], fill=hair)
        d.chord([hx - 1.7 * u, hy - 2.2 * u, hx + 1.8 * u, hy + 1.0 * u], 150, 360, fill=hair)
    if look.get("cap"):                                                # gorra de policía con visera
        d.rectangle([hx - 1.6 * u, hy - 2.4 * u, hx + 1.8 * u, hy - 1.0 * u], fill=look["cap"])
        d.rectangle([hx - 1.6 * u, hy - 1.3 * u, hx + 1.8 * u, hy - 1.0 * u], fill=(10, 10, 12))
        d.rectangle([hx - 2.6 * u, hy - 1.2 * u, hx - 1.2 * u, hy - .8 * u], fill=(10, 10, 12))
        d.rectangle([hx - 1.0 * u, hy - 2.1 * u, hx - .4 * u, hy - 1.5 * u], fill=(220, 180, 60))
    if look.get("badge"):
        d.rectangle([cx - 1.3 * u, 6 * u, cx - .6 * u, 6.8 * u], fill=look["badge"])
    if look.get("belt"):
        d.rectangle([cx - 2.1 * u, 11.6 * u, cx + 1.9 * u, 12.3 * u], fill=look["belt"])
    if look.get("earpiece"):
        d.line([(hx + .6 * u, hy), (hx + 1.0 * u, hy + 2.4 * u)], fill=look["earpiece"], width=1)
    return im


def claude_figure(h, frame, pose="walk", facing_right=False):
    """Claude como Morpheus: abrigo largo, anteojos negros y cabeza de chispa naranja."""
    look = dict(coat=(30, 40, 34), shirt=(60, 74, 66), skin=(217, 119, 87), hair=(0, 0, 0, 0),
                pants=(26, 30, 28), coat_len=21, tie=None)
    base = figure(h, frame, look, pose)
    u = h / 24
    pad = int(1.4 * u)                                   # espacio arriba para las puntas de la chispa
    im = Image.new("RGBA", (base.width, base.height + pad), (0, 0, 0, 0))
    im.paste(base, (0, pad))
    d = ImageDraw.Draw(im)
    w = im.width
    hx, hy = w / 2 - .3 * u, 2.2 * u + pad - .3 * u
    # borrar cabeza y dibujar chispa
    d.rectangle([hx - 3.4 * u, 0, hx + 3.4 * u, pad + 3.4 * u], fill=(0, 0, 0, 0))
    orange = (217, 119, 87, 255)
    for a in range(8):
        ang = a * np.pi / 4 + np.pi / 8
        r = 3.1 * u if a % 2 == 0 else 2.3 * u
        d.line([(hx, hy), (hx + np.cos(ang) * r, hy + np.sin(ang) * r)], fill=orange, width=max(1, int(.9 * u)))
    d.ellipse([hx - 1.2 * u, hy - 1.2 * u, hx + 1.2 * u, hy + 1.2 * u], fill=orange)
    d.rectangle([hx - 1.8 * u, hy - .55 * u, hx - .1 * u, hy + .15 * u], fill=(8, 10, 10, 255))   # anteojos
    d.line([(hx - .1 * u, hy - .45 * u), (hx + 1.4 * u, hy - .45 * u)], fill=(8, 10, 10, 255), width=1)
    if facing_right:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


LUCAS = dict(coat=(18, 20, 22), shirt=(30, 34, 40), skin=(214, 168, 136), hair=(58, 40, 28), curly=True,
             shades=(70, 96, 104), frame=(170, 176, 180), beard=(150, 112, 92), pants=(16, 16, 18), coat_len=21)


def frozen(im):
    """Versión congelada: silueta de código verde."""
    a = np.asarray(im, dtype=float)
    lum = (a[..., :3] @ np.array([.3, .59, .11])) / 255
    out = np.zeros_like(a)
    out[..., 0] = 10 + lum * 40
    out[..., 1] = 70 + lum * 170
    out[..., 2] = 30 + lum * 60
    alpha = a[..., 3] > 0
    edge = alpha & ~(np.roll(alpha, 1, 1) & np.roll(alpha, -1, 1) & np.roll(alpha, 1, 0) & np.roll(alpha, -1, 0))
    out[edge, :3] = (150, 255, 170)
    scan = (np.arange(a.shape[0]) % 2 == 0)[:, None]
    out[..., :3] *= np.where(scan, 1, .7)[..., None]
    out[..., 3] = alpha * 235
    return Image.fromarray(out.astype(np.uint8), "RGBA")


css.append("@keyframes fa{0%,49.9%{opacity:1}50%,100%{opacity:0}}@keyframes fb{0%,49.9%{opacity:0}50%,100%{opacity:1}}")
WALK_VIS = window(0, T, "walkvis")  # se reemplaza abajo
css.pop()
css.append(f"@keyframes walkvis{{0%,{pct(F0) - .001:.3f}%{{opacity:1}}{pct(F0):.3f}%,{pct(F1) - .001:.3f}%{{opacity:0}}"
           f"{pct(F1):.3f}%,100%{{opacity:1}}}}")
WALK_VIS = f"animation:walkvis {T}s infinite linear"
FRZ_VIS = window(F0, F1, "frzvis")


def sprite_group(frames, ox, oy, step=.5, freeze_img=None, flip=False):
    """frames: [A, B] caminando; freeze_img: cuadro durante el congelado."""
    fx = f' transform="translate({2 * ox + frames[0].width} 0) scale(-1 1)"' if flip else ""
    a, b = frames
    walk = (f'<g style="animation:fa {step}s infinite steps(1)">{image(a, ox, oy, up=1)}</g>'
            f'<g style="animation:fb {step}s infinite steps(1)">{image(b, ox, oy, up=1)}</g>')
    s = f'<g style="{WALK_VIS}">{walk}</g>'
    if freeze_img is not None:
        s += f'<g opacity="0" style="{FRZ_VIS}">{image(freeze_img, ox, oy, up=1)}</g>'
    return f'<g{fx}>{s}</g>'


# ============================================================ capa de congelado (debajo de la multitud)
css.append(f"@keyframes frzbg{{0%,{pct(F0) - .01:.3f}%{{opacity:0}}{pct(F0):.3f}%{{opacity:.95}}{pct(F0 + .5):.3f}%,"
           f"{pct(F1 - .4):.3f}%{{opacity:.82}}{pct(F1):.3f}%,100%{{opacity:0}}}}")
body.append(rect(0, 0, W, H, "#021a0c", f' opacity="0" style="animation:frzbg {T}s infinite linear"'))

# lluvia de código (solo durante el congelado)
GLYPHS = [c for c in MINI if c not in " .,'"]
rain = []
for cx_ in range(2, W, 7):
    if random.random() < .35:
        continue
    n = random.randint(6, 14)
    px_head, px_body = set(), set()
    for j in range(n):
        g = mini_px(random.choice(GLYPHS), cx_, j * 6)
        (px_head if j == n - 1 else px_body).update(g)
    dur, dl = random.uniform(2.2, 4.5), -random.uniform(0, 4)
    k = nid("cr")
    css.append(f"@keyframes {k}{{from{{transform:translateY({-n * 6}px)}}to{{transform:translateY({H}px)}}}}")
    rain.append(f'<g style="animation:{k} {dur:.2f}s {dl:.2f}s infinite linear">'
                f'<g fill="#1f8a44" opacity="{random.uniform(.35, .7):.2f}">{runs(px_body)}</g>'
                f'<g fill="#b8ffc8">{runs(px_head)}</g></g>')
body.append(f'<g opacity="0" style="{FRZ_VIS}">{"".join(rain)}</g>')

# ============================================================ multitud (fila de atrás)
SPAN, MARGIN = W + 60, 30
crowd_back = []
for i in range(11):
    h = random.randint(34, 40)
    look = dict(coat=random.choice(SUIT), skin=random.choice(SKIN), hair=random.choice(HAIR),
                shirt=random.choice([(200, 206, 196), (170, 190, 176), (220, 220, 210)]),
                tie=random.choice([None, (40, 60, 50), (90, 40, 40), (30, 30, 60)]))
    if random.random() < .3:
        look["skirt"] = tuple(int(c * 1.05) for c in look["coat"])
    if random.random() < .25:
        look["coat_len"] = 16
    k = random.choice([-1, 1, 1, 2, 2, 3])
    v = k * SPAN / WALK
    flip = k > 0                                   # los que van a la derecha miran a la derecha
    fr = [figure(h, 0, look), figure(h, 1, look)]
    s = sprite_group(fr, 0, -h, step=random.choice([.45, .5, .55]), freeze_img=frozen(fr[0]), flip=flip)
    crowd_back.append(f'<g style="{track(i * SPAN / 11 + random.uniform(-8, 8), v, SPAN, MARGIN, 102)}">{s}</g>')
body.extend(crowd_back)

# ============================================================ la mujer del vestido rojo (y el agente)
RED = dict(coat=(200, 28, 44), shirt=(200, 28, 44), skin=(236, 200, 170), hair=(226, 186, 96), long_hair=True,
           skirt=(214, 32, 50), skirt_len=19.5, pants=(236, 200, 170), shoe=(160, 16, 30), coat_len=9.5)
AGENT = dict(coat=(36, 40, 40), shirt=(226, 230, 222), skin=(214, 172, 140), hair=(28, 24, 20), tie=(8, 8, 8),
             shades=(6, 6, 8), earpiece=(200, 200, 190), pants=(30, 32, 34), coat_len=13.5)
WH, APPEAR, TURN = 50, 13.0, 18.6
WY = 110                                            # carril medio: entre la multitud y los protagonistas
wred = [figure(WH, 0, RED), figure(WH, 1, RED)]
wagt = [figure(WH, 0, AGENT), figure(WH, 1, AGENT)]
# aparece de golpe (con un parpadeo) y cruza; al "mirá de nuevo" se vuelve agente
css.append(f"@keyframes wred{{0%,{pct(APPEAR) - .001:.3f}%{{opacity:0}}{pct(APPEAR):.3f}%{{opacity:1}}"
           f"{pct(APPEAR + .08):.3f}%{{opacity:.15}}{pct(APPEAR + .16):.3f}%,{pct(TURN) - .001:.3f}%{{opacity:1}}"
           f"{pct(TURN):.3f}%,100%{{opacity:0}}}}")
css.append(f"@keyframes wagt{{0%,{pct(TURN) - .001:.3f}%{{opacity:0}}{pct(TURN):.3f}%{{opacity:1}}{pct(TURN + .1):.3f}%{{opacity:.2}}"
           f"{pct(TURN + .2):.3f}%{{opacity:1}}{pct(TURN + .3):.3f}%{{opacity:.4}}{pct(TURN + .4):.3f}%,99.9%{{opacity:1}}100%{{opacity:0}}}}")
w_red = f'<g opacity="0" style="animation:wred {T}s infinite linear">{sprite_group(wred, 0, -WH, step=.42, flip=True)}</g>'
w_agt = (f'<g opacity="0" style="animation:wagt {T}s infinite linear">'
         f'{sprite_group(wagt, 0, -WH, step=.42, freeze_img=frozen(wagt[0]), flip=True)}</g>')
WV, WX0 = 25, 50
x_at = lambda t: WX0 + WV * (t - APPEAR)
kw = nid("wk")
css.append(f"@keyframes {kw}{{0%,{pct(APPEAR):.3f}%{{transform:translate({WX0}px,{WY}px)}}"
           f"{pct(F0):.3f}%,{pct(F1):.3f}%{{transform:translate({x_at(F0):.1f}px,{WY}px)}}"
           f"100%{{transform:translate({x_at(F0) + WV * (T - F1):.1f}px,{WY}px)}}}}")
body.append(f'<g style="transform:translate({WX0}px,{WY}px);animation:{kw} {T}s infinite linear">{w_red}{w_agt}</g>')

# ============================================================ protagonistas
PH = 64
FEET = 118


def protagonist(frames, idle, ox, bob_delay=0):
    a, b = frames
    walk = (f'<g style="animation:fa .6s infinite steps(1)">{image(a, ox, FEET - a.height + 1, up=1)}</g>'
            f'<g style="animation:fb .6s infinite steps(1)">{image(b, ox, FEET - b.height + 1, up=1)}</g>')
    return (f'<g style="{WALK_VIS}"><g style="animation:pbob .6s {bob_delay}s infinite steps(1)">{walk}</g></g>'
            f'<g opacity="0" style="{FRZ_VIS}">{image(idle, ox, FEET - idle.height + 1, up=1)}</g>')


css.append("@keyframes pbob{0%,49.9%{transform:translateY(0)}50%,100%{transform:translateY(-1px)}}")
cl = [claude_figure(PH, 0), claude_figure(PH, 1)]
cl_idle = claude_figure(PH, 0, pose="idle", facing_right=True)
lu = [figure(PH, 0, LUCAS), figure(PH, 1, LUCAS)]
lu_idle = figure(PH, 0, LUCAS, pose="idle")
CLX, LUX = 92, 150
# sombras
for x_ in (CLX, LUX):
    body.append(f'<ellipse cx="{x_ + cl[0].width / 2}" cy="{FEET}" rx="10" ry="2" fill="#000" opacity=".22"/>')
body.append(protagonist(cl, cl_idle, CLX))
LOOK0, LOOK1 = 16.6, 19.0                          # Lucas se da vuelta para mirarla
css.append(f"@keyframes lturn{{0%,{pct(LOOK0):.3f}%{{transform:scaleX(1)}}{pct(LOOK0) + .001:.3f}%,{pct(LOOK1):.3f}%"
           f"{{transform:scaleX(-1)}}{pct(LOOK1) + .001:.3f}%,100%{{transform:scaleX(1)}}}}")
body.append(f'<g style="transform-origin:{LUX + lu[0].width / 2}px 0;animation:lturn {T}s infinite linear">'
            f'{protagonist(lu, lu_idle, LUX, .3)}</g>')

# ============================================================ el policía que se le cruza a Lucas
COP = dict(coat=(30, 40, 74), shirt=(150, 170, 200), skin=(196, 150, 118), hair=(30, 24, 20), cap=(26, 34, 64),
           badge=(230, 190, 70), belt=(14, 14, 16), pants=(24, 32, 60), coat_len=13)
CPH, CPY, C0, C1 = 66, 123, 3.0, 12.0
cop = [figure(CPH, 0, COP), figure(CPH, 1, COP)]
kc = nid("cop")
css.append(f"@keyframes {kc}{{0%,{pct(C0):.3f}%{{transform:translate(-50px,{CPY}px)}}{pct(C1):.3f}%,100%"
           f"{{transform:translate(330px,{CPY}px)}}}}")
body.append(f'<g style="transform:translate(-50px,{CPY}px);animation:{kc} {T}s infinite linear">'
            f'{sprite_group(cop, 0, -CPH, step=.5, flip=True)}</g>')

# ============================================================ multitud (fila de adelante, fuera de foco)
for i in range(2):
    h = random.randint(86, 96)
    look = dict(coat=(24, 30, 28), skin=(40, 46, 42), hair=(16, 20, 18), shirt=(34, 40, 36), shoe=(14, 16, 14))
    fr = [figure(h, 0, look), figure(h, 1, look)]
    s = sprite_group(fr, 0, -h, step=.4, freeze_img=frozen(fr[0]), flip=True)
    FS = W + 420                                   # tramo largo: pasan de vez en cuando
    v = 2 * FS / WALK
    at_freeze = [10, 560][i]                       # fuera de cuadro al congelar: no tapan a los protagonistas
    body.append(f'<g opacity=".93" style="{track(at_freeze - v * F0, v, FS, 100, 140)}">{s}</g>')

# ============================================================ diálogo
# (inicio, fin, hablante, [líneas]) en segundos del ciclo; 3x5 => máximo ~45 caracteres por línea
DIALOG = [
    (0.3, 4.4, "LUCAS", ["DONDE ESTAMOS?"]),
    (4.4, 9.2, "CLAUDE", ["EN TU PERFIL, LUCAS. ES UN SISTEMA.", "DE DIA: SOFTWARE ENGINEER EN HUMAND."]),
    (9.2, 14.0, "CLAUDE", ["BASE: CORDOBA, ARGENTINA.", "DE NOCHE: LIMAO, IA PARA PYMES."]),
    (14.0, 17.6, "CLAUDE", ["ESTABAS MIRANDO A LA MUJER DE ROJO?"]),
    (17.6, 20.0, "CLAUDE", ["MIRA DE NUEVO."]),
    (20.0, 24.0, "CLAUDE", ["CONGELALO.", "ESTO ES UN PROGRAMA DE ENTRENAMIENTO."]),
    (24.0, 28.0, "CLAUDE", ["INVENTARIO: MATE CEBADO, CURIOSIDAD X99.", "EN GITHUB DESDE 2019. TU COPILOTO: YO."]),
    (28.0, 31.8, "CLAUDE", ["SEGUIME, LUCAS.", "HAY CODIGO POR ESCRIBIR."]),
]
BX, BY, BW, BH = 8, 10, 172, 27
HEAD_COL = {"CLAUDE": "#ff9a70", "LUCAS": "#ffb050"}
TXT = "#d8f6f0"
panel = (rect(BX, BY, BW, BH, "#05060c", ' opacity=".72"')
         + f'<rect x="{BX + .5}" y="{BY + .5}" width="{BW - 1}" height="{BH - 1}" fill="none" stroke="{TXT}" opacity=".22"/>')
for cx_, cy_, dx_, dy_ in [(BX, BY, 1, 1), (BX + BW - 1, BY, -1, 1), (BX, BY + BH - 1, 1, -1), (BX + BW - 1, BY + BH - 1, -1, -1)]:
    panel += f'<g fill="{TXT}" opacity=".8">{runs({(cx_ + dx_ * i, cy_) for i in range(4)} | {(cx_, cy_ + dy_ * i) for i in range(4)})}</g>'
# el panel se pone verde durante el congelado
panel += rect(BX, BY, BW, BH, "#0a3a1a", f' opacity="0" style="animation:frzp {T}s infinite linear"')
css.append(f"@keyframes frzp{{0%,{pct(F0) - .01:.3f}%{{opacity:0}}{pct(F0):.3f}%,{pct(F1) - .01:.3f}%{{opacity:.5}}{pct(F1):.3f}%,100%{{opacity:0}}}}")
lines_svg = []
for t0, t1, who, lines in DIALOG:
    seg = []
    t = t0 + .15
    rows = [(f"> {who}", HEAD_COL[who])] + [(ln, TXT) for ln in lines]
    for li, (txt, col) in enumerate(rows):
        ly = BY + 3 + li * 8
        n = len(txt)
        px = mini_px(txt, BX + 4, ly)
        cid, ka = nid("cl"), nid("ty")
        dur = 0.01 if li == 0 else n * .045
        css.append(f"@keyframes {ka}{{0%,{pct(t):.3f}%{{transform:scaleX(0)}}{pct(t + dur):.3f}%,100%{{transform:scaleX(1)}}}}")
        defs.append(f'<clipPath id="{cid}"><rect x="{BX + 4}" y="{ly - 1}" width="{n * 4 + 1}" height="7" '
                    f'style="transform-origin:{BX + 4}px 0;animation:{ka} {T}s infinite steps({max(n, 1)})"/></clipPath>')
        seg.append(f'<g clip-path="url(#{cid})"><g fill="#000" opacity=".5" transform="translate(1 1)">{runs(px)}</g>'
                   f'<g fill="{col}">{runs(px)}</g></g>')
        t += dur + (.25 if li else .1)
    lines_svg.append(f'<g opacity="0" style="{window(t0, t1)}">{"".join(seg)}</g>')
css.append(f"@keyframes box{{0%{{transform:scaleY(0)}}{pct(.25):.3f}%,{pct(T - .25):.3f}%{{transform:scaleY(1)}}100%{{transform:scaleY(0)}}}}")
css.append("@keyframes cur{0%,49.9%{opacity:1}50%,100%{opacity:0}}")
cursor = rect(BX + BW - 7, BY + BH - 6, 3, 3, TXT, ' style="animation:cur .8s infinite steps(1)"')
body.append(f'<g style="transform-origin:0 {BY}px;animation:box {T}s infinite linear">{panel}{"".join(lines_svg)}{cursor}</g>')

# ============================================================ viñeta, franjas y tinte verde
defs.append('<radialGradient id="vig" cx=".5" cy=".5" r=".75"><stop offset=".55" stop-color="#000" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#000" stop-opacity=".55"/></radialGradient>')
body.insert(len(body) - 1, rect(0, 0, W, H, "#2a6a3a", ' opacity=".10"'))
body.insert(len(body) - 1, rect(0, 0, W, H, "url(#vig)"))
body.append(rect(0, 0, W, 6, "#000") + rect(0, H - 6, W, 6, "#000"))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{OUT_W}" height="{OUT_H}" '
       f'shape-rendering="crispEdges"><defs>{"".join(defs)}</defs><style>{"".join(css)}</style>'
       f'{rect(0, 0, W, H, "#05060c")}{"".join(body)}</svg>')
out = ROOT / "assets" / "lucas-himelfarb-construct.svg"
out.write_text(svg)
print(f"{out} ({len(svg) // 1024} KB)")
