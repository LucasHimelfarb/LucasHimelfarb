#!/usr/bin/env python3
"""Genera assets/lucas-himelfarb-construct.svg: segundo banner del perfil.

Guiño a la escena del programa de entrenamiento: Claude (Morpheus) camina adelante
y Lucas (Neo) lo sigue entre la multitud. Aparece de repente
una mujer de vestido rojo, Lucas se da vuelta a mirarla y ella se convierte en un
agente; después la calle se congela en código verde.

Uso:
    pip install pillow numpy
    python3 scripts/generate_construct_banner.py

Mismo formato que el banner principal (320x128 escalado a 800x320, franjas
cinemascope, viñeta).
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
GROUND = 96                                              # nuestra vereda (la avenida suma un carril de estacionamiento)
NEAR_LANE = 91                                           # base del carril de circulación cercano
sky = ramp(yy / H, [(0, (150, 172, 150)), (.5, (196, 210, 186)), (1, (210, 218, 196))])
# (el skyline plano del fondo se retiró: solo queda el cielo con bruma. El bucle se conserva
#  sin dibujar para no alterar la secuencia aleatoria del resto de la escena)
far = np.zeros((H, W), bool); far_rgb = sky.copy(); _discard = sky.copy()
x = 0
while x < W:
    bw, top = random.randint(14, 34), random.randint(6, 40)
    c = np.array(random.choice([(120, 138, 122), (132, 148, 128), (110, 126, 112)]), float)
    _discard[top:92, x:x + bw] = c
    _discard[top:92, x] = c + 18
    for wy in range(top + 3, 92, 4):
        for wx in range(x + 2, min(W, x + bw - 2), 3):
            _discard[wy:wy + 2, wx] = c - 22 if random.random() < .85 else c + 30
    x += bw + random.randint(0, 2)
haze = np.clip((yy - 20) / 70, 0, 1)[..., None]
far_rgb = far_rgb * (1 - haze * .35) + np.array([200, 212, 190]) * haze * .35
body.append(image(Image.fromarray(np.dstack([posterize(far_rgb, 24, 8), np.full((H, W), 255)]).astype(np.uint8), "RGBA"), up=1))

def facade(u, v, hb, base, aw, det, bid, kind="shop", corner_edge=False):
    """Fachada de piedra clara (tipo monumentos de Washington DC), común a toda la ciudad.

    u: posición horizontal sobre la cara, v: altura sobre el piso, hb: altura del edificio.
    Planta baja: comercio (vidriera, puerta, toldo opcional) u hall de entrada (columnas,
    cornisa, puerta iluminada) sobre un basamento de bloques. Arriba: pisos cada 10,
    columnas de ventanas cada 9, una moldura cada 4 pisos y cornisa en el techo.
    """
    det = np.asarray(det, float)
    det = det if det.ndim == 3 else np.broadcast_to(det, u.shape)[..., None]
    c = np.broadcast_to(base, u.shape + (3,)).copy()
    d = hb - v                                          # distancia al techo
    fl = np.floor((d - 6) / 10)
    r, col = (d - 6) % 10, u % 9
    upper = v > 31
    win = (d >= 6) & (r < 6) & upper & (col >= 2) & (col < 7)
    lit = ((fl * 7 + np.floor(u / 9) * 13 + hash(bid) % 11) % 19) == 0
    glass = np.where(lit[..., None], [196, 206, 170], [52, 62, 60])
    c = np.where(win[..., None], c * (1 - det) + glass * det, c)
    c = np.where((win & (r < 1))[..., None], c * (1 - det) + (base * .62) * det, c)                  # dintel en sombra
    sill = (d >= 6) & (r >= 6) & (r < 7) & upper & (col >= 1) & (col < 8)
    c = np.where(sill[..., None], c + 16 * det, c)
    course = (d >= 6) & ((fl % 4) == 3) & (r >= 7.5) & upper
    c = np.where(course[..., None], c - 18 * det, c)                                                # moldura
    pil = upper & (col < .9)
    c = np.where(pil[..., None], c + 8 * det, c)
    base_z = v < 31                                     # basamento de bloques
    blocks = base_z & (((v % 4) < .8) | ((((u + (np.floor(v / 4) % 2) * 6) % 12) < .7) & (v > 26)))
    c = np.where(blocks[..., None], c - 16 * det, c)
    e = u % 40                                          # una entrada cada 40
    if kind == "lobby":                                 # hall de entrada clásico
        door = (e > 14) & (e < 26) & (v < 20)
        lc = np.where(((e > 15.5) & (e < 24.5) & (v < 18.5))[..., None], [214, 196, 140], [40, 44, 42])
        lc = np.where((((e > 19.6) & (e < 20.4)) & (v < 18.5))[..., None], [90, 80, 60], lc)
        c = np.where(door[..., None], c * (1 - det) + lc * det, c)
        colm = (((e > 10) & (e < 12.6)) | ((e > 27.4) & (e < 30))) & (v < 24)
        c = np.where(colm[..., None], c + 22 * det, c)
        c = np.where((colm & (((e > 11.8) & (e < 12.6)) | ((e > 29.2) & (e < 30))))[..., None], c - 30 * det, c)
        ent = (e > 9) & (e < 31) & (v >= 24) & (v < 28)
        c = np.where(ent[..., None], c + 26 * det, c)
        c = np.where((ent & (v < 25))[..., None], c - 40 * det, c)
        step = (e > 9) & (e < 31) & (v < 1.6)
        c = np.where(step[..., None], c + 20 * det, c)
    else:                                               # comercio
        shop = (v > 1.5) & (v < 21) & (e > 3) & (e < 37)
        sc = np.where(((v > 3) & (v < 19.5) & (e > 5) & (e < 35))[..., None], [70, 92, 84], [44, 52, 50])
        sc = np.where(((v > 18.5) & (v < 19.5) & (e > 5) & (e < 35))[..., None], [176, 198, 180], sc)
        sc = np.where(((e > 16) & (e < 24) & (v < 17))[..., None], [34, 42, 38], sc)
        c = np.where(shop[..., None], c * (1 - det) + sc * det, c)
        if aw is not None:
            band = (v >= 21) & (v < 25) & (e > 2) & (e < 38)
            c = np.where(band[..., None], c * (1 - det) + (aw + ((np.floor(v) % 2) * 18)[..., None]) * det, c)
    c = np.where((d < 2)[..., None], c + 20, c)         # cornisa
    if corner_edge:
        c = np.where((u < 1.2)[..., None], c + 14, c)   # arista de la esquina
    return np.clip(c, 0, 255)


# ============================================================ vehículos (dibujados a 4x y reducidos)
VS = 4                                                   # supermuestreo


def _shade(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def _finish(big, w, h):
    """Reduce el dibujo supermuestreado a pixel art: color promediado, alfa binario."""
    sm = big.resize((w, h), Image.BOX)
    a = np.asarray(sm, dtype=float)
    out = a.copy()
    alpha = a[..., 3] > 110
    out[..., :3] = np.where(a[..., 3:4] > 0, a[..., :3] / np.maximum(a[..., 3:4], 1) * 255, 0)
    out[..., 3] = alpha * 255
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def vehicle_side(length, body, kind="sedan"):
    """Vehículo de perfil mirando a la izquierda. Altura con proporción real (auto ~1.45 m x 4.6 m)."""
    ratio = {"sedan": .31, "taxi": .31, "police": .31, "van": .42, "bus": .30}[kind]
    L, Hh = length * VS, length * ratio * VS
    top_pad = int(Hh * (.14 if kind in ("taxi", "police") else .02))
    im = Image.new("RGBA", (int(L), int(Hh) + top_pad + VS * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda pts: [(x * L, y * Hh + top_pad) for x, y in pts]
    hi, mid, lo, dark = _shade(body, 1.22), body, _shade(body, .78), _shade(body, .5)
    glass, glass_hi = (46, 58, 64), (120, 140, 140)
    # sombra en el piso
    d.ellipse([L * .02, Hh + top_pad - Hh * .06, L * .98, Hh + top_pad + Hh * .07], fill=(0, 0, 0, 120))
    if kind == "bus":
        d.rounded_rectangle([0, top_pad, L - 1, Hh * .86 + top_pad], radius=Hh * .06, fill=mid)
        d.rectangle([0, Hh * .62 + top_pad, L - 1, Hh * .86 + top_pad], fill=lo)
        d.rectangle([0, Hh * .58 + top_pad, L - 1, Hh * .62 + top_pad], fill=hi)                       # franja
        d.polygon(P([(.0, .08), (.07, .08), (.07, .52), (.0, .52)]), fill=glass)                       # parabrisas
        x = .1
        while x < .95:
            d.rectangle(P([(x, .1), (x + .075, .48)])[0] + P([(x + .075, .48)])[0], fill=glass)
            d.line(P([(x + .005, .12), (x + .03, .12)]), fill=glass_hi, width=VS)
            x += .088
        d.rectangle(P([(.14, .12), (.2, .84)])[0] + P([(.2, .84)])[0], fill=_shade(glass, 1.2))         # puerta
        d.line(P([(.17, .12), (.17, .84)]), fill=dark, width=VS)
        wheels = [(.16, .86), (.78, .86)]
        r = Hh * .12
    else:
        if kind == "van":
            shell = [(0, .56), (.02, .44), (.16, .34), (.26, .06), (.3, .02), (.97, .02), (1, .07), (1, .76), (.97, .84),
                     (.03, .84), (0, .76)]
            win = [(.27, .34), (.32, .08), (.5, .08), (.5, .34)]
            win2 = [(.53, .08), (.95, .08), (.95, .34), (.53, .34)]
            wheels = [(.17, .82), (.82, .82)]
        else:
            shell = [(0, .58), (.01, .48), (.06, .44), (.25, .39), (.31, .37), (.4, .07), (.45, .03), (.66, .03), (.72, .07),
                     (.82, .34), (.96, .37), (1, .44), (1, .74), (.97, .81), (.03, .81), (0, .74)]
            win = [(.34, .36), (.42, .09), (.555, .085), (.555, .355)]
            win2 = [(.575, .085), (.67, .085), (.71, .11), (.79, .345), (.575, .35)]
            wheels = [(.19, .78), (.8, .78)]
        r = Hh * (.2 if kind != "van" else .18)
        d.polygon(P(shell), fill=mid)
        d.polygon(P([(x, max(y, .6)) for x, y in shell]), fill=lo)                                      # mitad baja en sombra
        d.line(P([(.06, .45), (.97, .4)]), fill=hi, width=VS * 2)                                       # línea de hombros
        d.polygon(P(win), fill=glass); d.polygon(P(win2), fill=glass)
        d.line(P([(win[0][0] + .03, win[0][1] - .05), (win[1][0] + .05, win[1][1] + .06)]), fill=glass_hi, width=VS)  # reflejo
        d.line(P([(.03, .63), (.98, .63)]), fill=_shade(body, 1.05), width=VS)                         # moldura
        for dx in ((.36, .56, .76) if kind != "van" else (.27, .52)):
            d.line(P([(dx, .38), (dx, .78)]), fill=dark, width=max(1, VS // 2))                         # puertas
        for hx in ((.45, .65) if kind != "van" else (.36,)):
            d.rectangle(P([(hx, .47), (hx + .04, .5)])[0] + P([(hx + .04, .5)])[0], fill=hi)           # manijas
        d.rectangle(P([(0, .62), (.06, .76)])[0] + P([(.06, .76)])[0], fill=(40, 42, 44))               # paragolpes
        d.rectangle(P([(.94, .62), (1, .76)])[0] + P([(1, .76)])[0], fill=(40, 42, 44))
        d.rectangle(P([(0, .47), (.035, .55)])[0] + P([(.035, .55)])[0], fill=(250, 244, 200))          # faro
        d.rectangle(P([(.965, .44), (1, .54)])[0] + P([(1, .54)])[0], fill=(220, 30, 36))                # luz trasera
        if kind == "police":                                                    # puertas blancas, escudo, barral
            d.polygon(P([(.33, .4), (.78, .37), (.79, .62), (.33, .62)]), fill=(232, 232, 226))
            d.line(P([(.36, .38), (.36, .62)]), fill=dark, width=max(1, VS // 2))
            d.line(P([(.56, .38), (.56, .62)]), fill=(150, 150, 150), width=max(1, VS // 2))
            d.ellipse(P([(.43, .45), (.49, .56)])[0] + P([(.49, .56)])[0], fill=(206, 170, 60))
            d.rectangle([L * .46, top_pad - Hh * .1, L * .64, top_pad + Hh * .03], fill=(30, 30, 34))
            d.rectangle([L * .47, top_pad - Hh * .09, L * .55, top_pad + Hh * .0], fill=(220, 40, 50))
            d.rectangle([L * .55, top_pad - Hh * .09, L * .63, top_pad + Hh * .0], fill=(50, 90, 230))
        if kind == "taxi":
            d.rectangle([L * .5, top_pad - Hh * .12, L * .6, top_pad + Hh * .04], fill=(240, 236, 200))
            d.line(P([(.08, .55), (.95, .52)]), fill=(30, 30, 30), width=VS * 2)                        # franja
    for wx, wy in wheels:                                                                                 # ruedas
        cx, cy = wx * L, wy * Hh + top_pad
        d.ellipse([cx - r * 1.18, cy - r * 1.18, cx + r * 1.18, cy + r * 1.18], fill=(22, 22, 24))      # pasarrueda
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(14, 14, 16))
        d.ellipse([cx - r * .58, cy - r * .58, cx + r * .58, cy + r * .58], fill=(150, 154, 156))         # llanta
        d.ellipse([cx - r * .4, cy - r * .4, cx + r * .4, cy + r * .4], fill=(104, 108, 110))
        d.ellipse([cx - r * .14, cy - r * .14, cx + r * .14, cy + r * .14], fill=(40, 40, 42))
    return _finish(im, int(length), int(im.height / VS))


def vehicle_end(width, body, rear=False):
    """Auto visto de frente (faros) o de atrás (luces rojas). Alto/ancho reales (~1.45 m / 1.8 m)."""
    Wp, Hp = width * VS, width * .82 * VS
    im = Image.new("RGBA", (int(Wp), int(Hp)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda pts: [(x * Wp, y * Hp) for x, y in pts]
    hi, mid, lo = _shade(body, 1.2), body, _shade(body, .72)
    d.ellipse(P([(.0, .9), (1, 1.0)])[0] + P([(1, 1.0)])[0], fill=(0, 0, 0, 120))                        # sombra
    for tx in (.06, .76):                                                                                  # cubiertas
        d.rectangle(P([(tx, .78), (tx + .18, .99)])[0] + P([(tx + .18, .99)])[0], fill=(14, 14, 16))
    d.polygon(P([(.16, .02), (.84, .02), (.95, .44), (.05, .44)]), fill=lo)                               # cabina
    d.polygon(P([(.2, .07), (.8, .07), (.89, .41), (.11, .41)]), fill=(52, 66, 72) if not rear else (40, 50, 56))
    d.line(P([(.3, .1), (.42, .38)]), fill=(140, 160, 160), width=VS)                                      # reflejo
    d.rounded_rectangle(P([(.01, .42), (.99, .86)])[0] + P([(.99, .86)])[0], radius=VS * 2, fill=mid)    # carrocería
    d.line(P([(.03, .46), (.97, .46)]), fill=hi, width=VS)
    d.rectangle(P([(-.03, .36), (.04, .44)])[0] + P([(.04, .44)])[0], fill=lo)                            # espejos
    d.rectangle(P([(.96, .36), (1.03, .44)])[0] + P([(1.03, .44)])[0], fill=lo)
    lamp = (220, 30, 40) if rear else (252, 246, 210)
    d.rectangle(P([(.05, .52), (.24, .62)])[0] + P([(.24, .62)])[0], fill=lamp)                            # faros / luces
    d.rectangle(P([(.76, .52), (.95, .62)])[0] + P([(.95, .62)])[0], fill=lamp)
    if not rear:
        d.rectangle(P([(.3, .52), (.7, .66)])[0] + P([(.7, .66)])[0], fill=(30, 34, 34))                  # parrilla
        for gx in (.36, .44, .52, .6):
            d.line(P([(gx, .54), (gx, .64)]), fill=(70, 74, 74), width=max(1, VS // 2))
    else:
        d.rectangle(P([(.28, .5), (.72, .6)])[0] + P([(.72, .6)])[0], fill=lo)                            # baúl
    d.rectangle(P([(.38, .7), (.62, .78)])[0] + P([(.62, .78)])[0], fill=(226, 226, 210))                  # patente
    d.rectangle(P([(.0, .78), (1, .86)])[0] + P([(1, .86)])[0], fill=(40, 42, 44))                         # paragolpes
    return _finish(im, int(width), int(Hp / VS))


VEH_COLS = [(150, 36, 40), (52, 72, 110), (196, 194, 184), (36, 40, 44), (80, 98, 72), (120, 120, 124), (96, 60, 44)]


# fachada cercana + vereda: un tile de 320 px que se repite y se desplaza
near = np.zeros((H, W, 4))
AVE_TOP = 74                                                          # base de los edificios (vereda de enfrente)
x = 0
# piedra caliza clara, como los monumentos de Washington DC
FRONT_PAL = [(206, 204, 196), (196, 196, 190), (216, 213, 204), (188, 190, 184), (224, 221, 212), (200, 198, 188)]
AWN = [(52, 104, 74), (128, 120, 92), (70, 80, 76), (120, 60, 56)]
blds = []                                                             # (x, ancho, techo, color, toldo, tipo)
fi = 0
while x < W:
    bw = random.randint(46, 78)
    bw = min(bw, W - x) if W - x < 40 else bw
    c = np.array(random.choice(FRONT_PAL), float)
    top = -random.randint(30, 110)                                    # muy altos: el techo queda fuera de cuadro
    kind = "lobby" if random.random() < .45 else "shop"
    aw = np.array(random.choice(AWN), float) if kind == "shop" and random.random() < .7 else None
    yy_, xx_ = np.mgrid[0:AVE_TOP, x:min(W, x + bw)].astype(float)
    face = facade(xx_ - x + .5 - (bw % 40) / 2, AVE_TOP - yy_ - .5, AVE_TOP - top, c, aw, 1.0, ("f", fi), kind)
    near[0:AVE_TOP, x:x + bw, :3] = face
    near[0:AVE_TOP, x:x + bw, 3] = 255
    near[0:AVE_TOP, x:x + 1, :3] = np.clip(c + 14, 0, 255)            # juntas entre edificios
    near[0:AVE_TOP, min(W, x + bw) - 1, :3] = c * .78
    blds.append((x, bw, top, c.copy(), aw, kind))
    fi += 1
    x += bw
# avenida entre la vereda de enfrente y la nuestra, con bocacalle en la esquina
GX, GW, VY = 168, 92, 60                                              # hueco de la bocacalle
GC = GX + GW / 2
near[:AVE_TOP, GX:GX + GW, 3] = 0
cv = Image.new("RGBA", (W, H), (0, 0, 0, 0)); cd = ImageDraw.Draw(cv)
cd.rectangle([0, AVE_TOP, W, AVE_TOP + 3], fill=(150, 158, 140, 255))      # vereda de enfrente
cd.line([(0, AVE_TOP + 3), (W, AVE_TOP + 3)], fill=(190, 196, 180, 255))  # cordón
cd.rectangle([0, AVE_TOP + 4, W, GROUND - 1], fill=(64, 70, 68, 255))     # calzada
for lx in range(0, W, 12):                                                # línea divisoria
    cd.line([(lx, AVE_TOP + 11), (lx + 6, AVE_TOP + 11)], fill=(210, 204, 150, 255))
cd.line([(0, GROUND - 1), (W, GROUND - 1)], fill=(44, 50, 48, 255))
# senda peatonal también sobre la avenida, frente a la bocacalle
for lx in range(GX + 6, GX + GW - 6, 5):
    cd.rectangle([lx, AVE_TOP + 5, lx + 2, GROUND - 3], fill=(206, 208, 196, 255))
# semáforo en la esquina
sx_ = GX + GW + 2
cd.rectangle([sx_, 40, sx_ + 1, GROUND], fill=(30, 36, 32, 255))
cd.rectangle([sx_ - 2, 32, sx_ + 3, 42], fill=(24, 28, 26, 255))
cd.rectangle([sx_ - 1, 33, sx_ + 2, 35], fill=(255, 60, 60, 255))
cd.rectangle([sx_ - 1, 36, sx_ + 2, 38], fill=(60, 50, 30, 255))
cd.rectangle([sx_ - 1, 39, sx_ + 2, 41], fill=(30, 60, 40, 255))
cva = np.asarray(cv, dtype=float)
m = cva[..., 3] > 0
near[m, :3] = cva[m, :3]; near[m, 3] = 255


F, Z0 = 100.0, 100.0                                  # perspectiva de la calle transversal
VPX, VPY = GC, 46.0                                   # punto de fuga
WC = (GW / 2) * Z0 / F                                # semiancho hasta las fachadas
HC = (AVE_TOP - VPY) * Z0 / F                         # altura de la cámara: el piso en z0 cae en la base de las fachadas
WR = WC * .72                                         # semiancho de la calzada


def street_corridor():
    """Calle transversal en perspectiva de un punto, con edificios como volúmenes.

    Raycast por píxel contra cajas: cada edificio tiene fachada a la calle, cara de frente
    (hacia la cámara) y techo. Retiros y alturas distintos dan caras escalonadas; algunos
    tienen un piso superior retirado. Lo que no choca con nada deja ver el skyline del fondo.
    """
    rnd = random.Random(42)
    corner = {-1: next(b for b in blds if b[0] <= GX - 1 < b[0] + b[1]),
              1: next(b for b in blds if b[0] <= GX + GW < b[0] + b[1])}
    boxes = []                                            # (lado, x0, z0, z1, alto, color, id, toldo)
    for side in (-1, 1):
        z, bid = Z0, 0
        while z < 3600:
            ln = rnd.uniform(70, 200)
            if z == Z0:                                   # el mismo edificio de la avenida dobla la esquina
                _, _, top, col, aw, kind = corner[side]
                setback, hb, ln = 0, AVE_TOP - top, rnd.uniform(110, 170)
            else:
                setback = rnd.choice([0, 0, 2, 5, 9])
                hb = rnd.choice([rnd.uniform(70, 110), rnd.uniform(90, 150), rnd.uniform(120, 200)])
                col = np.array(rnd.choice(FRONT_PAL), float) * rnd.uniform(.94, 1.03)
                kind = "lobby" if rnd.random() < .45 else "shop"
                aw = np.array(rnd.choice(AWN), float) if kind == "shop" and rnd.random() < .6 else None
            boxes.append((side, WC + setback, z, z + ln, hb, col, (side, bid), aw, kind))
            if hb > HC + 20 and rnd.random() < .45:     # piso superior retirado
                boxes.append((side, WC + setback + 7, z + 10, z + ln - 10, hb + rnd.uniform(10, 26), col * .94, (side, bid, 1), None, kind))
            bid += 1
            z += ln + (rnd.uniform(30, 70) if rnd.random() < .12 else 0)   # a veces, otra bocacalle

    SS = 3                                                # supermuestreo 3x3: bordes en diagonal sin escalones
    rows, cols = AVE_TOP + 4, GW
    py, px = np.meshgrid((np.arange(rows * SS) + .5) / SS, GX + (np.arange(cols * SS) + .5) / SS, indexing="ij")
    dx, dy = px - VPX, py - VPY
    adx = np.abs(dx)
    sgn = np.where(dx < 0, -1, 1)
    best = np.full(px.shape, np.inf)
    rgb = np.zeros(px.shape + (3,))
    hit = np.zeros(px.shape, bool)

    def put(mask, z, color):
        m = mask & (z < best)
        best[m] = z[m]
        rgb[m] = color[m] if color.ndim == 3 else color
        hit[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        # piso: calzada, cordón, vereda
        zg = np.where(dy > 1e-6, F * HC / dy, np.inf)
        X = adx * zg / F
        g = np.zeros(px.shape + (3,)) + (64, 70, 68)
        g[X > WR] = (190, 196, 180)
        g[X > WR + 1.2] = (150, 158, 140)
        dash = (X < 1.4) & ((zg % 50) < 24)
        g[dash] = (210, 204, 150)
        g[(X <= WR) & (X > WR - 1.5)] -= 10                   # sombra del cordón
        put(np.isfinite(zg), zg, g)

        for side, x0, z0, z1, hb, col, bid, aw, kind in boxes:
            on = sgn == side
            light = 1.08 if side < 0 else .74                 # una vereda al sol, la otra a la sombra
            # fachada a la calle (plano X = x0)
            zf = F * x0 / adx
            hf = HC - dy * zf / F                             # altura sobre el piso
            mf = on & (zf >= z0) & (zf < z1) & (hf >= 0) & (hf <= hb)
            if mf.any():
                u, v = zf - z0, hf
                det = np.exp(-(zf - Z0) / 900)[..., None]     # el detalle se esfuma con la distancia
                c = facade(u, v, hb, col * light, aw, det, bid, kind, corner_edge=(z0 == Z0))
                put(mf, zf, c)
            # cara de frente (plano z = z0, mira a la cámara)
            xf = adx * z0 / F
            hz = HC - dy * z0 / F
            mz = on & (xf >= x0) & (xf <= x0 + 80) & (hz >= 0) & (hz <= hb)
            if mz.any():
                u, v = xf - x0, hz
                det = np.exp(-(z0 - Z0) / 900)
                c = facade(u, v, hb, col * (.9 if side < 0 else .84), aw, det, bid, kind, corner_edge=True)
                put(mz, np.full(px.shape, z0, float), c)
            # techo (visible solo en edificios más bajos que la cámara)
            if hb < HC:
                zr = np.where(dy > 1e-6, F * (HC - hb) / dy, np.inf)
                xr = adx * zr / F
                mr = on & (zr >= z0) & (zr < z1) & (xr >= x0) & (xr <= x0 + 80)
                if mr.any():
                    c = np.broadcast_to(col * 1.14, px.shape + (3,)).copy()
                    c = np.where(((xr - x0) < 1.6)[..., None], c + 22, c)                  # parapeto
                    c = np.where(((((zr - z0) % 40) < 6) & ((xr - x0) > 18) & ((xr - x0) < 30))[..., None], c * .7, c)  # tanques
                    put(mr, zr, c)

    t = 1 - np.exp(-np.clip(best - Z0, 0, None) / 1100)
    haze = np.array([196, 210, 186], float)
    fine = np.clip(rgb * (1 - t[..., None]) + haze * t[..., None], 0, 255)
    sky_bg = np.repeat(np.repeat(far_rgb[:rows, GX:GX + GW], SS, 0), SS, 1)   # lo que se ve detrás: el cielo
    fine = np.where(hit[..., None], fine, sky_bg)
    out = np.zeros((rows, cols, 4))
    out[..., :3] = fine.reshape(rows, SS, cols, SS, 3).mean((1, 3))
    out[..., 3] = (hit.reshape(rows, SS, cols, SS).mean((1, 3)) > 0) * 255
    return out, 0


corr, cy0 = street_corridor()
# fila de autos esperando el semáforo (carril de mano contraria, vistos de frente con los faros prendidos)
QW = 15                                                  # ancho de un auto a la distancia z0
for zq, colq in sorted([(112, VEH_COLS[2]), (150, VEH_COLS[1]), (198, VEH_COLS[0]), (262, VEH_COLS[5])], reverse=True):
    car = vehicle_end(max(4, round(QW * Z0 / zq)), colq)
    ca = np.asarray(car, dtype=float)
    qx = VPX + F * (-WR * .5) / zq - car.width / 2 - GX
    qy = VPY + F * HC / zq - car.height - cy0
    t_h = 1 - np.exp(-(zq - Z0) / 1100)
    ca[..., :3] = ca[..., :3] * (1 - t_h) + np.array([196, 210, 186]) * t_h
    for yy_q in range(car.height):
        for xx_q in range(car.width):
            ty_, tx_ = int(qy) + yy_q, int(qx) + xx_q
            if ca[yy_q, xx_q, 3] > 0 and 0 <= ty_ < corr.shape[0] and 0 <= tx_ < corr.shape[1]:
                corr[ty_, tx_, :3] = ca[yy_q, xx_q, :3]; corr[ty_, tx_, 3] = 255
# alumbrado público en las veredas de la calle transversal, en perspectiva
LS = 3                                                   # postes dibujados a 3x y reducidos (sin escalones)
lamp_cv = Image.new("RGBA", (corr.shape[1] * LS, corr.shape[0] * LS), (0, 0, 0, 0)); ld = ImageDraw.Draw(lamp_cv)
for zl in sorted(np.arange(118, 2600, 95), reverse=True):
    t_h = 1 - np.exp(-(zl - Z0) / 1100)
    pole = tuple(int(v) for v in np.array([44, 50, 48]) * (1 - t_h) + np.array([196, 210, 186]) * t_h) + (255,)
    head = tuple(int(v) for v in np.array([226, 230, 210]) * (1 - t_h) + np.array([196, 210, 186]) * t_h) + (255,)
    wdt = max(1, round(1.4 * Z0 / zl * LS))
    for sd in (-1, 1):
        Xl = sd * (WR + 4)
        xb = (VPX + F * Xl / zl - GX) * LS
        yb, yt = (VPY + F * HC / zl - cy0) * LS, (VPY + F * (HC - 34) / zl - cy0) * LS
        xa = (VPX + F * (Xl - sd * 7) / zl - GX) * LS
        hw, hh = 2.2 * Z0 / zl * LS, max(LS, 1.2 * Z0 / zl * LS)
        ld.line([(xb, yb), (xb, yt)], fill=pole, width=wdt)
        ld.line([(xb, yt), (xa, yt)], fill=pole, width=wdt)
        ld.rectangle([min(xa, xa + sd * hw), yt, max(xa, xa + sd * hw), yt + hh], fill=head)
la = np.asarray(lamp_cv.resize((corr.shape[1], corr.shape[0]), Image.BOX), dtype=float)
lam = la[..., 3:4] / 255
lrgb = np.where(lam > 0, la[..., :3] / np.maximum(lam, 1e-6), 0)
corr[..., :3] = corr[..., :3] * (1 - lam) + lrgb * lam
corr[..., 3] = np.maximum(corr[..., 3], (lam[..., 0] > .35) * 255)
cm = corr[..., 3] > 0
reg = near[cy0:cy0 + corr.shape[0], GX:GX + GW]
reg[cm, :3] = corr[cm, :3]; reg[cm, 3] = 255

# vereda neoyorquina: losas grandes de hormigón, cordón de granito, puertas de sótano de chapa,
# tapas de servicio, chicles, grietas y un hidrante. Generador aparte para no alterar el resto.
srng = np.random.default_rng(1212)
sr = random.Random(1212)
near[GROUND:, :, 3] = 255
SW_ROWS = [GROUND + 3, GROUND + 9, GROUND + 18, H]                     # filas de losas (más altas cerca)
SLAB = 30                                                               # ancho de losa (~1.5 m)
grain = srng.normal(0, 1, (H, W))
SW_OFF = 7                                                              # juntas alineadas: grilla, no ladrillo
for r0, r1 in zip(SW_ROWS, SW_ROWS[1:]):
    off_ = SW_OFF
    for sx0 in range(-off_, W, SLAB):
        base_c = np.array([186, 186, 178]) + sr.uniform(-12, 10) + np.array([sr.uniform(-3, 3), 0, sr.uniform(-4, 2)])
        x0_, x1_ = max(0, sx0), min(W, sx0 + SLAB)
        if x1_ <= x0_:
            continue
        blk = np.broadcast_to(base_c, (r1 - r0, x1_ - x0_, 3)).copy()
        blk += grain[r0:r1, x0_:x1_, None] * 3.2                        # grano del hormigón
        vy = np.linspace(0, 1, r1 - r0)[:, None, None]
        blk *= (1.02 - .05 * vy)                                         # leve desgaste hacia el borde
        near[r0:r1, x0_:x1_, :3] = blk
        if sx0 >= 0:                                                    # junta vertical: sombra + borde claro
            near[r0:r1, sx0, :3] = (118, 122, 116)
            if sx0 + 1 < W:
                near[r0:r1, sx0 + 1, :3] = np.minimum(near[r0:r1, sx0 + 1, :3] + 16, 255)
        if sr.random() < .22:                                           # grieta
            cxk, cyk = sr.uniform(x0_ + 3, max(x0_ + 4, x1_ - 3)), r0 + 1
            for _ in range(int((r1 - r0) * 1.4)):
                cyk += sr.uniform(.4, 1); cxk += sr.uniform(-1.1, 1.1)
                if r0 <= int(cyk) < r1 and x0_ <= int(cxk) < x1_:
                    near[int(cyk), int(cxk), :3] = (124, 126, 120)
    near[r0, :, :3] = (122, 126, 120)                                    # junta horizontal
    if r0 + 1 < H:
        near[r0 + 1, :, :3] = np.minimum(near[r0 + 1, :, :3] + 12, 255)
# cordón de granito con el borde iluminado y la cuneta en sombra
near[GROUND, :, :3] = (46, 50, 48)
near[GROUND + 1, :, :3] = (214, 214, 206)
near[GROUND + 2, :, :3] = (168, 170, 164) + grain[GROUND + 2, :, None] * 6
for gx_ in range(0, W, 47):
    near[GROUND + 1:GROUND + 3, gx_, :3] = (120, 122, 118)
# chicles y manchas
for _ in range(90):
    gx_, gy_ = sr.randrange(0, W), sr.randrange(GROUND + 4, H)
    near[gy_, gx_, :3] = np.array(near[gy_, gx_, :3]) * .7
for _ in range(5):
    ox_, oy_, rw = sr.randrange(0, W), sr.randrange(GROUND + 4, GROUND + 12), sr.randrange(4, 9)
    yy_o, xx_o = np.ogrid[GROUND + 3:H, 0:W]
    m_o = ((xx_o - ox_) / rw) ** 2 + ((yy_o - oy_) / (rw * .35)) ** 2 < 1
    near[GROUND + 3:H, :, :3][m_o] *= .86


def diamond_plate(x0, y0, w, h):
    """Puerta de sótano de chapa diamantada (dos hojas)."""
    yy_d, xx_d = np.mgrid[0:h, 0:w]
    plate = np.zeros((h, w, 3)) + (120, 124, 126)
    patt = ((xx_d + (yy_d // 2) * 2) % 4 == 0) & (yy_d % 2 == 0)
    plate[patt] = (176, 180, 182)
    plate[:, w // 2] = (70, 74, 76)                                      # junta entre hojas
    plate[0, :] = (200, 204, 204); plate[-1, :] = (60, 62, 64)
    plate[:, 0] = (190, 194, 194); plate[:, -1] = (66, 68, 70)
    for hx_ in (2, w - 3):
        plate[h // 3, hx_] = plate[2 * h // 3, hx_] = (50, 52, 54)      # bisagras
    near[y0:y0 + h, x0:x0 + w, :3] = plate


def manhole(cx_, cy_, rx, ry):
    yy_m, xx_m = np.ogrid[0:H, 0:W]
    m_ = ((xx_m - cx_) / rx) ** 2 + ((yy_m - cy_) / ry) ** 2
    near[(m_ < 1) & (yy_m >= GROUND + 3), :3] = (88, 90, 88)
    ring = (m_ >= .7) & (m_ < 1) & (yy_m >= GROUND + 3)
    near[ring, :3] = (60, 62, 60)
    grid_ = (m_ < .7) & ((xx_m + yy_m) % 3 == 0) & (yy_m >= GROUND + 3)
    near[grid_, :3] = (110, 112, 108)


diamond_plate(22, H - 12, 22, 9)
diamond_plate(GX + GW + 22, H - 13, 24, 10)
manhole(120, GROUND + 7, 6, 2.4)
# hidrante junto al cordón
HX, HB = GX - 30, GROUND + 6
hyd = Image.new("RGBA", (9, 14), (0, 0, 0, 0)); hd = ImageDraw.Draw(hyd)
hd.rectangle([2, 4, 6, 12], fill=(176, 40, 40)); hd.rectangle([1, 12, 7, 13], fill=(140, 30, 30))
hd.ellipse([2, 1, 6, 5], fill=(190, 48, 48)); hd.rectangle([3, 0, 5, 1], fill=(150, 34, 34))
hd.rectangle([0, 6, 8, 7], fill=(160, 36, 36)); hd.point((4, 3), fill=(230, 120, 110))
hd.rectangle([2, 5, 2, 11], fill=(214, 90, 84))
ha = np.asarray(hyd, dtype=float); hm = ha[..., 3] > 0
ys_h, xs_h = np.nonzero(hm)
near[HB - 14 + ys_h, HX + xs_h, :3] = ha[ys_h, xs_h, :3]

# bocacalle de nuestro lado: calzada que se abre hacia la cámara, con senda de rayas verticales
# (en el sentido de la calle que cruzamos), en perspectiva hacia el punto de fuga de la calle del fondo
VPY_S = 46.0
for y_ in range(GROUND, H):
    f_ = (y_ + .5 - VPY_S) / (GROUND - VPY_S)
    half = (GW / 2 + 2) * f_
    xa_, xb_ = GC - half, GC + half
    xs_all = np.arange(W) + .5
    road = (xs_all >= xa_) & (xs_all <= xb_)
    near[y_, road, :3] = (70, 76, 74)
    if y_ >= GROUND + 2:
        Xn = (xs_all - GC) / f_                                          # posición "real" en la calle
        stripe = road & (np.abs(Xn) < GW / 2 - 3) & (((Xn + GW / 2) % 7) < 3.6)
        near[y_, stripe, :3] = (226, 228, 214)
    for xe in (xa_ - 1, xb_ + 1):                                        # cordones en perspectiva
        xi = int(round(xe))
        if 0 <= xi < W:
            near[y_, xi, :3] = (200, 202, 196)
for _ in range(3):                                                      # postes / farolas
    px = random.choice([random.randrange(10, GX - 12), random.randrange(GX + GW + 8, W - 10)])
    near[30:GROUND + 2, px:px + 2, :3] = (40, 48, 44); near[30:GROUND + 2, px:px + 2, 3] = 255
    near[30:32, px - 4:px + 6, :3] = (40, 48, 44); near[30:32, px - 4:px + 6, 3] = 255


near_rgb = posterize(np.clip(near[..., :3], 0, 255), 64, 6)
near_im = Image.fromarray(np.dstack([near_rgb, near[..., 3]]).astype(np.uint8), "RGBA")
tile = image(near_im, up=1) + image(near_im, x=-W, up=1)
# fase del tile: al congelar, la senda queda justo debajo de Claude y Lucas (x ≈ 93..185)
TILE0 = 93 - GX - NEAR_SPEED * F0
body.append(f'<g style="{track(TILE0, NEAR_SPEED, W, 0, 0)}">{tile}</g>')

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
    fem = look.get("female")
    lw = .75 if fem else 1.0                                            # ancho de pierna
    for sgn, shade in ((1, .8), (-1, 1.0)):                             # pierna de atrás primero
        col = tuple(int(c * shade) for c in pants)
        fx = cx - sgn * stride
        d.polygon([(cx - 1.1 * u, hip), (cx + 1.1 * u, hip), (fx + lw * u, foot - u), (fx - lw * u, foot - u)], fill=col)
        if look.get("heels"):
            d.polygon([(fx - 1.3 * u, foot - .4 * u), (fx + .3 * u, foot - 1.1 * u), (fx + .5 * u, foot - .3 * u),
                       (fx - 1.3 * u, foot)], fill=shoe)
        else:
            d.rectangle([fx - 1.6 * u, foot - u, fx + 0.8 * u, foot], fill=shoe)
            if look.get("sole"):
                d.rectangle([fx - 1.6 * u, foot - .35 * u, fx + 0.8 * u, foot], fill=look["sole"])
    # torso / abrigo
    coat_len = look.get("coat_len", 13.5) * u
    if look.get("backpack"):
        d.rounded_rectangle([cx + 1.1 * u, 4.8 * u, cx + 2.9 * u, 10.4 * u], radius=max(1, int(.6 * u)), fill=look["backpack"])
    long_coat = coat_len > 14 * u
    if fem:
        d.polygon([(cx - 1.5 * u, 4.3 * u), (cx + 1.3 * u, 4.3 * u), (cx + 1.0 * u, 8.4 * u),
                   (cx + 1.8 * u + (stride * .3 if long_coat else 0), max(11 * u, coat_len)),
                   (cx - 1.7 * u - (stride * .4 if long_coat else 0), max(11 * u, coat_len)),
                   (cx - 1.1 * u, 8.4 * u), (cx - 1.8 * u, 6.2 * u)], fill=coat)
        if coat_len < 11 * u:                                              # top corto: cadera del pantalón
            d.polygon([(cx - 1.1 * u, coat_len), (cx + 1.0 * u, coat_len), (cx + 1.6 * u, hip), (cx - 1.6 * u, hip)], fill=pants)
    else:
        d.polygon([(cx - 1.9 * u, 4.2 * u), (cx + 1.6 * u, 4.2 * u), (cx + 2.2 * u + (stride * .3 if long_coat else 0), coat_len),
                   (cx - 2.4 * u - (stride * .5 if long_coat else 0), coat_len)], fill=coat)
    if look.get("backpack"):
        d.line([(cx - .9 * u, 4.6 * u), (cx - .3 * u, 8.5 * u)], fill=tuple(int(c * .7) for c in look["backpack"]), width=1)
    d.polygon([(cx - 1.9 * u, 4.4 * u), (cx - 0.9 * u, 4.4 * u), (cx - 1.4 * u, 7.5 * u)], fill=shirt)   # cuello
    if look.get("tie"):
        d.line([(cx - 1.5 * u, 5 * u), (cx - 1.7 * u, 8 * u)], fill=look["tie"], width=max(1, int(u * .6)))
    # brazo (balanceo opuesto a la pierna delantera)
    swing = {"walk": [-1.6, -0.2][frame], "idle": -0.2}[pose] * u
    arm = tuple(int(c * .85) for c in coat)
    aw_ = max(2, int((1.15 if fem else 1.5) * u))
    if look.get("short_sleeves"):
        d.line([(cx, 5 * u), (cx + swing, 10.8 * u)], fill=skin, width=aw_)
        d.line([(cx, 5 * u), (cx + swing * .35, 7.2 * u)], fill=arm, width=aw_ + 1)
    else:
        d.line([(cx, 5 * u), (cx + swing, 10.8 * u)], fill=arm, width=aw_)
    d.ellipse([cx + swing - .7 * u, 10.6 * u, cx + swing + .7 * u, 11.8 * u], fill=skin)
    if look.get("bag"):
        d.line([(cx + swing, 10.9 * u), (cx + swing, 12 * u)], fill=tuple(int(c * .6) for c in look["bag"]), width=1)
        d.rectangle([cx + swing - 1.0 * u, 12 * u, cx + swing + 1.0 * u, 14.2 * u], fill=look["bag"])
    # cabeza
    hx, hy = cx - .3 * u, 2.2 * u
    d.rectangle([cx - .7 * u, 3.4 * u, cx + .3 * u, 4.4 * u], fill=skin)                                  # cuello
    d.ellipse([hx - 1.5 * u, hy - 1.9 * u, hx + 1.5 * u, hy + 1.7 * u], fill=skin)
    d.chord([hx - 1.6 * u, hy - 2.1 * u, hx + 1.7 * u, hy + 1.2 * u], 160, 360, fill=hair)
    d.rectangle([hx + .2 * u, hy - 1.4 * u, hx + 1.7 * u, hy + .6 * u], fill=hair)
    if look.get("curly"):
        for ox, oy in ((-1.4, -1.9), (-.4, -2.4), (.6, -2.2), (1.5, -1.5), (1.8, -.4)):
            d.ellipse([hx + (ox - .6) * u, hy + (oy - .5) * u, hx + (ox + .6) * u, hy + (oy + .5) * u], fill=hair)
    if look.get("ponytail"):
        sw_ = (.3 if frame == 0 else -.2) * u
        d.polygon([(hx + 1.0 * u, hy - 1.4 * u), (hx + 1.8 * u, hy - 1.1 * u), (hx + 2.4 * u + sw_, hy + 2.6 * u),
                   (hx + 1.7 * u + sw_, hy + 2.8 * u)], fill=hair)
    if look.get("bun"):
        d.ellipse([hx + .9 * u, hy - 2.1 * u, hx + 2.3 * u, hy - .7 * u], fill=hair)
    if look.get("ballcap"):
        d.chord([hx - 1.6 * u, hy - 2.3 * u, hx + 1.7 * u, hy + .4 * u], 180, 360, fill=look["ballcap"])
        d.rectangle([hx - 2.7 * u, hy - 1.05 * u, hx - .6 * u, hy - .6 * u], fill=tuple(int(c * .7) for c in look["ballcap"]))
    if look.get("shades"):
        d.rectangle([hx - 1.6 * u, hy - .5 * u, hx - .2 * u, hy + .2 * u], fill=look["shades"])
        d.line([(hx - .2 * u, hy - .4 * u), (hx + 1.2 * u, hy - .5 * u)], fill=look.get("frame", (20, 20, 20)), width=1)
    if look.get("beard"):
        d.rectangle([hx - 1.3 * u, hy + .6 * u, hx + .4 * u, hy + 1.5 * u], fill=look["beard"])
    if look.get("skirt"):
        sl = look.get("skirt_len", 16) * u
        if look.get("pencil"):
            d.polygon([(cx - 1.6 * u, 9.6 * u), (cx + 1.7 * u, 9.6 * u), (cx + 1.5 * u, sl), (cx - 1.4 * u, sl)], fill=look["skirt"])
        else:
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
        if look.get("badge"):
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

# ============================================================ tránsito sobre la avenida
# ============================================================ arquetipos de la multitud
DENIM = [(54, 70, 104), (70, 86, 118), (40, 48, 70)]
TEE = [(196, 60, 52), (214, 170, 70), (60, 120, 118), (226, 226, 218), (90, 110, 150), (130, 70, 110)]
SNEAK = [(226, 226, 220), (40, 40, 44), (190, 60, 50)]


def crowd_look(kind, rng):
    base = dict(skin=rng.choice(SKIN), hair=rng.choice(HAIR))
    if kind == "m_suit":        # hombre de traje
        return dict(base, coat=rng.choice(SUIT), shirt=rng.choice([(200, 206, 196), (220, 220, 210)]),
                    tie=rng.choice([(40, 60, 50), (90, 40, 40), (30, 30, 60)]), coat_len=13.5)
    if kind == "m_casual":      # remera, jean, zapatillas; a veces gorra y mochila
        tee = rng.choice(TEE)
        return dict(base, coat=tee, shirt=tee, pants=rng.choice(DENIM), shoe=rng.choice(SNEAK), sole=(236, 236, 230),
                    short_sleeves=True, coat_len=12.2, ballcap=rng.choice([None, (40, 60, 100), (150, 40, 40)]),
                    backpack=rng.choice([None, (60, 66, 70), (120, 90, 50)]))
    if kind == "m_jacket":      # campera, chino, barba
        return dict(base, coat=rng.choice([(90, 96, 60), (70, 48, 34), (40, 40, 44), (120, 96, 70)]),
                    shirt=rng.choice(TEE), pants=rng.choice([(150, 132, 96), (60, 62, 58), (48, 56, 76)]),
                    shoe=(70, 50, 36), coat_len=12.0, beard=rng.choice([None, base["hair"]]))
    if kind == "f_suit":        # mujer de saco y pollera tubo, rodete, tacos
        c = rng.choice(SUIT + [(110, 40, 50), (40, 50, 80)])
        return dict(base, female=True, coat=c, skirt=c, pencil=True, skirt_len=15.6, pants=(196, 160, 136),
                    shoe=(20, 20, 22), heels=True, bun=True, coat_len=10.4, shirt=(226, 226, 220),
                    hair=rng.choice(HAIR[:3] + [(190, 150, 80)]))
    if kind == "f_casual":      # suéter, jean, colita, tote
        return dict(base, female=True, coat=rng.choice(TEE + [(200, 180, 160)]), shirt=(226, 226, 220),
                    pants=rng.choice(DENIM), shoe=rng.choice(SNEAK), sole=(236, 236, 230), coat_len=10.0,
                    ponytail=True, long_hair=False, bag=rng.choice([(196, 170, 120), (60, 50, 44), (150, 60, 50)]),
                    hair=rng.choice(HAIR[:3] + [(190, 150, 80), (150, 60, 40)]))
    # f_coat: piloto beige a la rodilla, pelo largo, medias y botas
    return dict(base, female=True, coat=rng.choice([(176, 150, 110), (120, 110, 100), (60, 64, 70)]),
                shirt=(226, 220, 210), pants=(36, 32, 34), shoe=(40, 28, 24), coat_len=15.8, long_hair=True,
                belt=rng.choice([None, (120, 96, 66)]), hair=rng.choice(HAIR[:3] + [(190, 150, 80), (150, 60, 40)]))


KINDS = ["m_suit", "f_casual", "m_casual", "f_suit", "m_jacket", "f_coat"]

# ------------------------------------------------ un auto dobla en la esquina y se pierde por la calle
def eff(t):
    return t if t < F0 else (F0 if t < F1 else t - (F1 - F0))


def tile_g(t):                                         # desplazamiento (sin módulo) del tile en el tiempo t
    return TILE0 + NEAR_SPEED * eff(t)


TURN_COL = (196, 150, 40)                              # un taxi amarillo, para que se distinga
T_IN, T_TURN, T_GONE = 11.6, 14.6, 19.6
LANE_Y = NEAR_LANE                                     # carril cercano (va hacia la derecha)
X_LANE = WR * .5                                       # carril de ida de la calle transversal
z_a = F * HC / (LANE_Y - VPY)
x_turn = VPX + F * X_LANE / z_a                        # en coordenadas del tile
scr_turn = (tile_g(T_TURN) + x_turn) % W
x_start = -40 + x_turn - scr_turn + NEAR_SPEED * (T_TURN - T_IN)
side = vehicle_side(76, TURN_COL, "taxi").transpose(Image.FLIP_LEFT_RIGHT)
rear = vehicle_end(24, TURN_COL, rear=True)
ka, kb = nid("turnA"), nid("turnB")
css.append(f"@keyframes {ka}{{0%,{pct(T_IN):.3f}%{{transform:translate({x_start:.1f}px,{LANE_Y}px);opacity:0}}"
           f"{pct(T_IN) + .001:.3f}%{{opacity:1;transform:translate({x_start:.1f}px,{LANE_Y}px)}}"
           f"{pct(T_TURN - .5):.3f}%{{opacity:1;transform:translate({x_turn - 3:.1f}px,{LANE_Y}px)}}"
           f"{pct(T_TURN):.3f}%{{opacity:1;transform:translate({x_turn:.1f}px,{LANE_Y}px)}}"
           f"{pct(T_TURN) + .001:.3f}%,100%{{opacity:0;transform:translate({x_turn:.1f}px,{LANE_Y}px)}}}}")
frames = [f"0%,{pct(T_TURN) - .001:.3f}%{{opacity:0;transform:translate({x_turn:.1f}px,{LANE_Y}px) scale(1)}}"]
kz = np.log(3200 / z_a) / (T_GONE - T_TURN)
for i in range(17):
    t = T_TURN + (T_GONE - T_TURN) * i / 16
    z = z_a * np.exp(kz * (t - T_TURN))
    haze_t = 1 - np.exp(-max(z - Z0, 0) / 1100)
    op = max(0.0, 1 - haze_t * 1.05) if i < 16 else 0
    frames.append(f"{pct(t):.3f}%{{opacity:{op:.2f};transform:translate({VPX + F * X_LANE / z:.2f}px,"
                  f"{VPY + F * HC / z:.2f}px) scale({z_a / z:.4f})}}")
frames.append(f"100%{{opacity:0;transform:translate({VPX:.1f}px,{VPY}px) scale(.02)}}")
css.append(f"@keyframes {kb}{{{''.join(frames)}}}")
turn_svg = (f'<g style="animation:{ka} {T}s infinite linear">{image(side, -side.width // 2, -side.height, up=1)}</g>'
            f'<g style="animation:{kb} {T}s infinite linear">{image(rear, -rear.width // 2, -rear.height, up=1)}</g>')
# va solo en la copia del tile que está en pantalla durante el giro (sin vuelta del tile en esa ventana)
assert int(tile_g(T_IN) // W) == int(tile_g(T_GONE) // W), "el tile da la vuelta durante el giro del taxi"
copy_off = round((scr_turn - (tile_g(T_TURN) % W) - x_turn) / W) * W
turn_svg = f'<g transform="translate({copy_off} 0)">{turn_svg}</g>'
body.append(f'<g style="{track(TILE0, NEAR_SPEED, W, 0, 0)}"><g style="{WALK_VIS}">{turn_svg}</g></g>')

# ------------------------------------------------ gente caminando por las veredas de la calle transversal
PREF, PH_Z0 = 18, 8.2                                    # alto del sprite y alto de una persona a la distancia z0
ZN, ZR = 104.0, 640.0                                    # tramo de vereda que recorren
peds = []
prng = random.Random(7)
for i in range(10):
    sd = (-1, 1)[i % 2]
    X = sd * (WR + 3.5 + prng.uniform(0, 6))
    away = prng.random() < .5
    vz = ZR / WALK * prng.choice([1, 1, 2])            # (se sortea igual para no alterar la secuencia aleatoria)
    vz = ZR / WALK                                       # una vuelta por ciclo: paso de caminata para todos
    zst = prng.uniform(0, ZR)
    look = crowd_look(KINDS[(i * 5) % len(KINDS)], prng)
    fr = [figure(PREF, 0, look), figure(PREF, 1, look)]
    flip = (sd > 0) != away                              # de perfil, mirando hacia donde camina
    FADE = 40.0                                          # tramo (en z) de aparición / desaparición

    def pose(q):
        z = ZN + (q if away else ZR - q)
        s_ = PH_Z0 * Z0 / z / PREF
        x = VPX + F * X / z - fr[0].width * s_ / 2
        y = VPY + F * HC / z - PREF * s_
        th = 1 - np.exp(-(z - Z0) / 1100)
        op = max(0.0, min(1.0, q / FADE, (ZR - q) / FADE)) * max(0.0, 1 - th * 1.1)
        return x, y, s_, op

    def frame_at(t, q, op=None):
        x, y, s_, o = pose(q)
        o = o if op is None else op
        return (t, f"transform:translate({x:.1f}px,{y:.1f}px) scale({s_:.3f});opacity:{o:.2f}")

    kf = [frame_at(t, (zst + vz * eff(t)) % ZR) for t in np.arange(0, T + .001, .5)]
    # vuelta del tramo: corte exacto e invisible (sin interpolar de punta a punta)
    e_w = (ZR - zst) / vz
    while e_w < WALK:
        t_w = e_w if e_w < F0 else e_w + (F1 - F0)
        kf += [frame_at(t_w - .002, ZR - 1e-6, 0.0), frame_at(t_w + .002, 0.0, 0.0)]
        e_w += ZR / vz
    kf.sort(key=lambda f: f[0])
    kf = [f"{pct(t):.3f}%{{{body_}}}" for t, body_ in kf]
    k = nid("cp")
    css.append(f"@keyframes {k}{{{''.join(kf)}}}")
    spr = sprite_group(fr, 0, 0, step=.5, freeze_img=frozen(fr[0]), flip=flip)
    peds.append(f'<g style="transform-origin:0 0;animation:{k} {T}s infinite linear">{spr}</g>')
cross_peds = "".join(peds)
cross_peds += f'<g transform="translate({-W} 0)">{cross_peds}</g>'
body.append(f'<g style="{track(TILE0, NEAR_SPEED, W, 0, 0)}">{cross_peds}</g>')

# tamaño real respecto de la gente (persona ~37 px = 1.75 m): auto ~4.6 m de largo
CSPAN, CMARG = W + 320, 160
LANES = [  # (base y, sentido, k: vueltas por ciclo, vehículos [(tipo, largo)])
    (AVE_TOP + 11, -1, -1, [("sedan", 64), ("bus", 150), ("van", 60), ("sedan", 66)]),
    (NEAR_LANE, 1, 2, [("sedan", 76), ("van", 72), ("sedan", 78), ("sedan", 74)]),
]


def taxi_screen_x(t):                          # posición en pantalla del taxi que dobla mientras se acerca
    return -40 + (scr_turn + 40) * (t - T_IN) / (T_TURN - T_IN)


def lane_phase(vehs, v):
    """Desfasaje del carril cercano que deja un hueco libre para el taxi que dobla."""
    gaps = CSPAN / len(vehs)
    for off in np.arange(0, gaps, 1.0):
        ok = True
        for t in np.arange(T_IN, T_TURN + .01, .1):
            tx = taxi_screen_x(t)
            for j in range(len(vehs)):
                cx = ((j * gaps + off + v * eff(t)) % CSPAN) - CMARG
                if abs(cx - tx) < 86:
                    ok = False; break
            if not ok:
                break
        if ok:
            return off
    return 0.0


for ly, dirn, k, vehs in LANES:
    v = k * CSPAN / WALK                       # mismo ritmo en cada carril: no se pisan entre sí
    gaps = CSPAN / len(vehs)
    phase = lane_phase(vehs, v) if dirn > 0 else 0.0
    for j, (kind, ln) in enumerate(vehs):
        col = {"bus": (204, 186, 96), "taxi": (214, 170, 40)}.get(kind) or random.choice(VEH_COLS)
        im = vehicle_side(ln, col, kind)
        if dirn > 0:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        x0 = j * gaps + phase
        spr = (f'<g style="{WALK_VIS}">{image(im, 0, -im.height + 2, up=1)}</g>'
               f'<g opacity="0" style="{FRZ_VIS}">{image(frozen(im), 0, -im.height + 2, up=1)}</g>')
        body.append(f'<g style="{track(x0, v, CSPAN, CMARG, ly)}">{spr}</g>')

# ============================================================ escena de infracción junto a nuestro cordón
# En el carril de estacionamiento (más cerca que el tránsito, más lejos que la gente de la vereda), de
# izquierda a derecha: el patrullero con la sirena, el oficial y un auto estacionado frente a un cartel de
# prohibido estacionar. El oficial, a escala de los autos (~1.75 m contra ~1.45 m), apoya un pie en el
# paragolpe trasero del infractor y escribe la multa. La escena cruza la costura del tile (x > W).
def officer_writing(h, k):
    """Policía mirando a la derecha, pie sobre un paragolpe, escribiendo en la libreta (cuadro k de 4)."""
    u = h / 24
    w = int(12 * u) + 6
    im = Image.new("RGBA", (w, h + 1), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx = w * .38
    navy, navy_d, skin, shoe = (30, 40, 74), (22, 30, 56), (196, 150, 118), (12, 12, 14)
    foot = h - 1
    d.polygon([(cx - 1.0 * u, 12.5 * u), (cx + .9 * u, 12.5 * u), (cx + .7 * u, foot - u), (cx - .9 * u, foot - u)], fill=navy_d)
    d.rectangle([cx - .9 * u, foot - u, cx + 1.4 * u, foot], fill=shoe)
    kx, ky = cx + 3.4 * u, 12.6 * u                                     # rodilla
    fx, fy = cx + 4.6 * u, foot - 6.2 * u                               # pie sobre el paragolpe
    d.polygon([(cx - .6 * u, 12.0 * u), (cx + .8 * u, 13.4 * u), (kx + .6 * u, ky + .6 * u), (kx - .2 * u, ky - .7 * u)], fill=navy)
    d.polygon([(kx - .4 * u, ky), (kx + .8 * u, ky), (fx + .6 * u, fy - u), (fx - .6 * u, fy - u)], fill=navy)
    d.rectangle([fx - .8 * u, fy - u, fx + 1.6 * u, fy], fill=shoe)
    d.polygon([(cx - 1.4 * u, 4.6 * u), (cx + 1.5 * u, 4.4 * u), (cx + 1.8 * u, 12.6 * u), (cx - 1.6 * u, 12.8 * u)], fill=navy)
    d.rectangle([cx - 1.7 * u, 11.6 * u, cx + 1.9 * u, 12.4 * u], fill=(14, 14, 16))                   # cinturón
    d.rectangle([cx + .5 * u, 6.0 * u, cx + 1.2 * u, 6.8 * u], fill=(230, 190, 70))                     # placa
    # libreta en la mano izquierda, a la altura del pecho
    px0, py0 = cx + 2.0 * u, 6.6 * u
    d.line([(cx + .2 * u, 5.2 * u), (cx + 1.0 * u, 8.6 * u), (px0 + .4 * u, py0 + 2.2 * u)], fill=navy_d, width=max(2, int(1.3 * u)))
    d.rectangle([px0, py0, px0 + 2.0 * u, py0 + 2.4 * u], fill=(236, 236, 226))
    d.line([(px0 + .1 * u, py0 + .2 * u), (px0 + 1.9 * u, py0 + .2 * u)], fill=(200, 60, 60), width=1)
    for j in range(min(k + 1, 3)):                                      # renglones que se van escribiendo
        d.line([(px0 + .3 * u, py0 + (.8 + j * .5) * u), (px0 + (1.1 + .25 * ((j + k) % 3)) * u, py0 + (.8 + j * .5) * u)],
               fill=(60, 70, 90), width=1)
    # mano derecha con la birome, se mueve sobre la libreta
    hx_, hy_ = px0 + [0.5, 1.2, 0.8, 1.5][k] * u, py0 + [1.0, 1.1, 1.6, 1.4][k] * u
    d.line([(cx + .6 * u, 5.0 * u), (cx + 1.6 * u, 8.0 * u), (hx_, hy_ + .4 * u)], fill=navy, width=max(2, int(1.3 * u)))
    d.ellipse([hx_ - .5 * u, hy_, hx_ + .5 * u, hy_ + .9 * u], fill=skin)
    d.line([(hx_, hy_), (hx_ + .5 * u, hy_ - .9 * u)], fill=(30, 30, 40), width=1)                     # birome
    # cabeza mirando la libreta
    hx, hy = cx + .7 * u, 2.5 * u
    d.rectangle([cx - .1 * u, 3.5 * u, cx + .9 * u, 4.7 * u], fill=skin)
    d.ellipse([hx - 1.5 * u, hy - 1.8 * u, hx + 1.5 * u, hy + 1.7 * u], fill=skin)
    d.rectangle([hx - 1.7 * u, hy - 2.4 * u, hx + 1.6 * u, hy - 1.0 * u], fill=(26, 34, 64))
    d.rectangle([hx - 1.7 * u, hy - 1.3 * u, hx + 1.6 * u, hy - 1.0 * u], fill=(10, 10, 12))
    d.rectangle([hx + 1.2 * u, hy - 1.1 * u, hx + 2.6 * u, hy - .7 * u], fill=(10, 10, 12))            # visera
    d.rectangle([hx + .4 * u, hy - 2.1 * u, hx + 1.0 * u, hy - 1.5 * u], fill=(220, 180, 60))
    d.rectangle([hx - 1.4 * u, hy - 1.0 * u, hx - .4 * u, hy + .4 * u], fill=(40, 30, 24))
    d.point((hx + .9 * u, hy + .2 * u), fill=(30, 24, 20))                                             # mirada baja
    return im


def no_parking_sign(h):
    """Cartel argentino de prohibido estacionar: círculo rojo con la E tachada, sobre un poste."""
    S = 4
    w = int(h * .32)
    big = Image.new("RGBA", (w * S, (h + 1) * S), (0, 0, 0, 0)); d = ImageDraw.Draw(big)
    cxs, r = w * S / 2, w * S * .46
    d.rectangle([cxs - S * .6, r * 1.6, cxs + S * .6, (h + 1) * S], fill=(70, 76, 74))                  # poste
    cy_ = r + S
    d.ellipse([cxs - r, cy_ - r, cxs + r, cy_ + r], fill=(206, 30, 40))
    d.ellipse([cxs - r * .76, cy_ - r * .76, cxs + r * .76, cy_ + r * .76], fill=(244, 244, 238))
    ex, ey, eh = cxs - r * .28, cy_ - r * .42, r * .84                                                   # la E
    d.rectangle([ex, ey, ex + r * .14, ey + eh], fill=(20, 20, 20))
    for yy_e in (0, .43, .86):
        d.rectangle([ex, ey + eh * yy_e - r * .02, ex + r * .5, ey + eh * yy_e + r * .12], fill=(20, 20, 20))
    d.line([(cxs - r * .54, cy_ - r * .54), (cxs + r * .54, cy_ + r * .54)], fill=(206, 30, 40), width=int(r * .2))
    return _finish(big, w, h + 1)


PCL, VCL = 80, 76                                        # largo del patrullero y del infractor
PCX = GX + GW + 8                                        # pasada la esquina (cruza la costura del tile)
PBASE = GROUND                                           # ruedas contra el cordón
pcar = vehicle_side(PCL, (26, 28, 32), "police").transpose(Image.FLIP_LEFT_RIGHT)   # mira a la derecha
car_h = PCL * .31
OH = round(car_h * 1.75 / 1.45)                          # alto del oficial a escala del auto
ofr = [officer_writing(OH, k) for k in range(4)]
p_y = PBASE - pcar.height + 2
o_x = PCX + PCL + 1                                      # entre el frente del patrullero y el infractor
o_y = PBASE - OH
foot_x = o_x + int(ofr[0].width * .38 + 4.6 * OH / 24 + 1.6 * OH / 24)
VCX = foot_x - 1                                         # paragolpe trasero del infractor bajo su pie
vcar = vehicle_side(VCL, (122, 34, 40), "sedan").transpose(Image.FLIP_LEFT_RIGHT)
v_y = PBASE - vcar.height + 2
sign = no_parking_sign(40)
SX, SY = VCX + VCL - 16, GROUND + 4 - sign.height        # en la vereda, junto al cordón, frente al infractor
_hh = PCL * .31 * VS
_tp = int(_hh * .14)
bar_y0, bar_y1 = p_y + (_tp - _hh * .1) / VS, p_y + (_tp + _hh * .03) / VS
bar_xb, bar_xm, bar_xr = PCX + PCL * (1 - .63), PCX + PCL * (1 - .55), PCX + PCL * (1 - .47)
defs.append('<radialGradient id="sirR"><stop offset="0" stop-color="#ff3040" stop-opacity=".75"/>'
            '<stop offset="1" stop-color="#ff3040" stop-opacity="0"/></radialGradient>'
            '<radialGradient id="sirB"><stop offset="0" stop-color="#3a70ff" stop-opacity=".75"/>'
            '<stop offset="1" stop-color="#3a70ff" stop-opacity="0"/></radialGradient>')
css.append("@keyframes sirR{0%,8%{opacity:1}9%,16%{opacity:0}17%,25%{opacity:1}26%,100%{opacity:0}}"
           "@keyframes sirB{0%,49%{opacity:0}50%,58%{opacity:1}59%,66%{opacity:0}67%,75%{opacity:1}76%,100%{opacity:0}}")
for k in range(4):                                       # escritura: 4 cuadros que se alternan
    a_, b_ = k * 25, (k + 1) * 25
    css.append(f"@keyframes wr{k}{{0%,100%{{opacity:0}}{a_}%,{b_ - .01}%{{opacity:1}}{b_}%{{opacity:0}}}}"
               if k else "@keyframes wr0{0%,24.99%{opacity:1}25%,100%{opacity:0}}")
bh_ = bar_y1 - bar_y0
siren = (f'<g style="animation:sirB .9s infinite steps(1)">'
         f'<ellipse cx="{(bar_xb + bar_xm) / 2:.1f}" cy="{bar_y0 + bh_ / 2:.1f}" rx="15" ry="9" fill="url(#sirB)"/>'
         f'<rect x="{bar_xb:.1f}" y="{bar_y0:.1f}" width="{bar_xm - bar_xb:.1f}" height="{bh_:.1f}" fill="#d4e0ff"/></g>'
         f'<g style="animation:sirR .9s infinite steps(1)">'
         f'<ellipse cx="{(bar_xm + bar_xr) / 2:.1f}" cy="{bar_y0 + bh_ / 2:.1f}" rx="15" ry="9" fill="url(#sirR)"/>'
         f'<rect x="{bar_xm:.1f}" y="{bar_y0:.1f}" width="{bar_xr - bar_xm:.1f}" height="{bh_:.1f}" fill="#ffd0d4"/></g>')
writing = "".join(f'<g opacity="{1 if k == 0 else 0}" style="animation:wr{k} 1.6s infinite steps(1)">'
                  f'{image(ofr[k], o_x, o_y, up=1)}</g>' for k in range(4))
scene = (f'<g style="{WALK_VIS}">{image(pcar, PCX, p_y, up=1)}{image(vcar, VCX, v_y, up=1)}{writing}{siren}'
         f'{image(sign, SX, SY, up=1)}</g>'
         f'<g opacity="0" style="{FRZ_VIS}">{image(frozen(pcar), PCX, p_y, up=1)}{image(frozen(vcar), VCX, v_y, up=1)}'
         f'{image(frozen(ofr[0]), o_x, o_y, up=1)}{image(frozen(sign), SX, SY, up=1)}</g>')
patrol = "".join(f'<g transform="translate({off} 0)">{scene}</g>' for off in (0, -W, -2 * W))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, W, 0, 0)}">{patrol}</g>')

# ============================================================ multitud (fila de atrás)
SPAN, MARGIN = W + 60, 30
crowd_back = []
for i in range(11):
    kind = KINDS[i % len(KINDS)]
    look = crowd_look(kind, random)
    h = random.randint(33, 37) if look.get("female") else random.randint(36, 40)
    k = random.choice([-1, 1, 1, 2, 2, 3])
    v = k * SPAN / WALK
    flip = k > 0                                   # los que van a la derecha miran a la derecha
    fr = [figure(h, 0, look), figure(h, 1, look)]
    s = sprite_group(fr, 0, -h, step=random.choice([.45, .5, .55]), freeze_img=frozen(fr[0]), flip=flip)
    crowd_back.append(f'<g style="{track(i * SPAN / 11 + random.uniform(-8, 8), v, SPAN, MARGIN, 102)}">{s}</g>')
body.extend(crowd_back)

# ============================================================ la mujer del vestido rojo (y el agente)
def woman_in_red(h, frame):
    """Silueta de reloj de arena: vestido rojo ajustado a la rodilla, pelo rubio largo, tacos."""
    u = h / 24
    w = int(12 * u) + 6
    im = Image.new("RGBA", (w, h + 1), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx = w / 2
    skin, skin_sh = (238, 204, 176), (214, 172, 146)
    red, red_hi, red_sh = (212, 22, 44), (244, 70, 84), (150, 10, 30)
    hair, hair_hi = (232, 192, 104), (250, 226, 150)
    sway = (.35 if frame == 0 else -.35) * u              # balanceo de cadera
    stride = (1.7 if frame == 0 else .35) * u
    foot = h - 1
    knee = 15.6 * u
    # piernas (de atrás primero), largas y finas, con tacos
    for sg, col in ((1, skin_sh), (-1, skin)):
        fx = cx + sway * .5 - sg * stride
        d.polygon([(cx + sway - .7 * u, knee - u), (cx + sway + .6 * u, knee - u), (fx + .35 * u, foot - 1.1 * u),
                   (fx - .35 * u, foot - 1.1 * u)], fill=col)
        d.polygon([(fx - 1.2 * u, foot - .6 * u), (fx + .3 * u, foot - 1.2 * u), (fx + .5 * u, foot - .4 * u),
                   (fx - 1.2 * u, foot)], fill=(176, 12, 34))                                   # zapato
        d.line([(fx + .45 * u, foot - .5 * u), (fx + .5 * u, foot)], fill=(120, 6, 24), width=1)  # taco
    # pelo largo por detrás (cae hasta media espalda y se mueve)
    hx, hy = cx - .2 * u, 2.3 * u
    flow = (.4 if frame == 0 else -.2) * u
    d.polygon([(hx - .2 * u, hy - 1.6 * u), (hx + 1.6 * u, hy - 1.2 * u), (hx + 2.3 * u + flow, hy + 3.5 * u),
               (hx + 2.0 * u + flow, hy + 6.8 * u), (hx + .9 * u + flow, hy + 6.2 * u), (hx + .3 * u, hy + 2.6 * u)], fill=hair)
    # vestido: hombros, busto, cintura marcada, cadera, falda a la rodilla
    body_pts = [(cx - 1.5 * u, 4.3 * u), (cx + 1.2 * u, 4.3 * u),               # hombros
                (cx + 1.0 * u, 6.2 * u), (cx + .9 * u + sway * .3, 8.6 * u),       # espalda y cintura
                (cx + 1.7 * u + sway, 10.8 * u), (cx + 1.4 * u + sway, knee),      # cadera y falda
                (cx - 1.3 * u + sway, knee), (cx - 1.5 * u + sway, 10.8 * u),
                (cx - .8 * u + sway * .3, 8.6 * u), (cx - 1.9 * u, 6.4 * u)]      # cintura y busto
    d.polygon(body_pts, fill=red)
    d.line([(cx - 1.6 * u, 5.0 * u), (cx - 1.8 * u, 6.3 * u)], fill=red_hi, width=1)           # brillo del busto
    d.line([(cx - .9 * u + sway * .3, 8.8 * u), (cx - 1.4 * u + sway, 10.8 * u)], fill=red_hi, width=1)
    d.line([(cx + .8 * u + sway * .3, 8.8 * u), (cx + 1.5 * u + sway, 10.8 * u)], fill=red_sh, width=1)
    d.line([(cx - .4 * u + sway, 12 * u), (cx - .2 * u + sway, knee)], fill=red_sh, width=1)  # pliegue
    # escote y hombros descubiertos
    d.polygon([(cx - 1.4 * u, 4.2 * u), (cx + 1.1 * u, 4.2 * u), (cx + .6 * u, 4.9 * u), (cx - 1.2 * u, 5.0 * u)], fill=skin)
    # brazo con balanceo y cartera chica
    sw = (-1.2 if frame == 0 else .6) * u
    d.line([(cx + .1 * u, 4.8 * u), (cx + sw * .5, 7.6 * u), (cx + sw, 10.2 * u)], fill=skin, width=max(1, int(.75 * u)))
    d.rectangle([cx + sw - .7 * u, 10.0 * u, cx + sw + .6 * u, 10.9 * u], fill=(30, 20, 24))
    # cabeza
    d.rectangle([cx - .55 * u, 3.4 * u, cx + .25 * u, 4.4 * u], fill=skin)                         # cuello fino
    d.ellipse([hx - 1.3 * u, hy - 1.7 * u, hx + 1.3 * u, hy + 1.6 * u], fill=skin)
    d.chord([hx - 1.5 * u, hy - 2.0 * u, hx + 1.6 * u, hy + 1.1 * u], 170, 360, fill=hair)
    d.polygon([(hx - 1.5 * u, hy - 1.1 * u), (hx - .4 * u, hy - 1.9 * u), (hx - .9 * u, hy - .2 * u)], fill=hair)  # flequillo
    d.line([(hx - .3 * u, hy - 1.7 * u), (hx + 1.2 * u, hy - 1.2 * u)], fill=hair_hi, width=1)
    d.point((hx - .9 * u, hy - .3 * u), fill=(40, 30, 40))                                         # ojo
    d.line([(hx - 1.2 * u, hy + .75 * u), (hx - .75 * u, hy + .75 * u)], fill=(200, 16, 40), width=1)  # labios
    return im


AGENT = dict(coat=(36, 40, 40), shirt=(226, 230, 222), skin=(214, 172, 140), hair=(28, 24, 20), tie=(8, 8, 8),
             shades=(6, 6, 8), earpiece=(200, 200, 190), pants=(30, 32, 34), coat_len=13.5)
WH, APPEAR, TURN = 50, 13.0, 18.6
WY = 110                                            # carril medio: entre la multitud y los protagonistas
wred = [woman_in_red(WH, 0), woman_in_red(WH, 1)]
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

# ============================================================ multitud (fila de adelante, fuera de foco)
for i in range(2):
    h = random.randint(86, 96)
    look = crowd_look(["m_casual", "f_coat"][i], random)
    dark = lambda c: tuple(int(16 + v * .12) for v in c) if c else c
    look = {k_: (dark(v_) if isinstance(v_, tuple) else v_) for k_, v_ in look.items()}
    fr = [figure(h, 0, look), figure(h, 1, look)]
    s = sprite_group(fr, 0, -h, step=.4, freeze_img=frozen(fr[0]), flip=True)
    FS = W + 420                                   # tramo largo: pasan de vez en cuando
    v = 2 * FS / WALK
    at_freeze = [10, 560][i]                       # fuera de cuadro al congelar: no tapan a los protagonistas
    body.append(f'<g opacity=".93" style="{track(at_freeze - v * F0, v, FS, 100, 140)}">{s}</g>')

# ============================================================ viñeta, franjas y tinte verde
defs.append('<radialGradient id="vig" cx=".5" cy=".5" r=".75"><stop offset=".55" stop-color="#000" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#000" stop-opacity=".55"/></radialGradient>')
body.append(rect(0, 0, W, H, "#2a6a3a", ' opacity=".10"'))
body.append(rect(0, 0, W, H, "url(#vig)"))
body.append(rect(0, 0, W, 6, "#000") + rect(0, H - 6, W, 6, "#000"))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{OUT_W}" height="{OUT_H}" '
       f'shape-rendering="crispEdges"><defs>{"".join(defs)}</defs><style>{"".join(css)}</style>'
       f'{rect(0, 0, W, H, "#05060c")}{"".join(body)}</svg>')
out = ROOT / "assets" / "lucas-himelfarb-construct.svg"
out.write_text(svg)
print(f"{out} ({len(svg) // 1024} KB)")
