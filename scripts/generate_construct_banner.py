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
import zlib
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
TW = 2 * W               # ancho del tramo de fondo que se repite (una sola esquina: cuadra larga)
NEAR_SPEED = TW / WALK   # px/s de la fachada cercana (cámara que sigue a los protagonistas, ~1 m/s)

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


def limb(d, p0, p1, w0, w1, col):
    """Segmento de extremidad con volumen: trapecio que se afina de w0 a w1, con extremos redondeados."""
    (x0, y0), (x1, y1) = p0, p1
    dx_, dy_ = x1 - x0, y1 - y0
    L_ = max(1e-6, np.hypot(dx_, dy_))
    nx_, ny_ = -dy_ / L_, dx_ / L_
    d.polygon([(x0 + nx_ * w0 / 2, y0 + ny_ * w0 / 2), (x1 + nx_ * w1 / 2, y1 + ny_ * w1 / 2),
               (x1 - nx_ * w1 / 2, y1 - ny_ * w1 / 2), (x0 - nx_ * w0 / 2, y0 - ny_ * w0 / 2)], fill=col)
    for (x, y), w_ in ((p0, w0), (p1, w1)):
        d.ellipse([x - w_ / 2, y - w_ / 2, x + w_ / 2, y + w_ / 2], fill=col)


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
LANE_FAR, LANE_NEAR = 99, 104                             # carriles de la vereda: van (derecha) / vienen (izquierda, como nosotros)
ZOOM = 1.25                                              # cámara más cerca: la escena se agranda hacia el horizonte
PERSP_K, HORIZON = .67, 46                               # alto de una persona = PERSP_K * (pies - HORIZON)
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

def stable_hash(x):
    """Hash estable entre ejecuciones (hash() de Python cambia en cada corrida con strings)."""
    return zlib.crc32(repr(x).encode())


# Escala de fachada a la distancia de la vereda de enfrente (una persona ~20 px = 1.75 m):
G_FLOOR, FLOOR_H, BAY = 32.0, 26.0, 18.0              # planta baja, piso y ancho de vano
BRICKS = [(150, 84, 66), (128, 74, 60), (110, 70, 58), (176, 150, 116), (164, 132, 100)]
CANOPY = [(40, 70, 52), (110, 36, 40), (30, 40, 70), (24, 26, 28)]


def facade(u, v, hb, base, aw, det, bid, kind="shop", corner_edge=False, width=None):
    """Fachada de un barrio de Nueva York de los 2000, a escala real, común a toda la ciudad.

    u: posición horizontal sobre la cara, v: altura sobre el piso, hb: altura del edificio.
    Cada edificio (bid) elige material y detalles con un hash estable: piedra caliza clara (la mayoría),
    ladrillo rojo/marrón u ocre con hiladas, o hierro fundido estilo SoHo. Planta baja de comercio
    (cartel sin letras, vidriera, puerta, toldo) o de hall (marquesina, puerta iluminada, número).
    Pisos con ventanas de guillotina, dinteles, alféizares, cortinas, aires acondicionados de ventana,
    molduras, escaleras de incendio en los de ladrillo y cornisa con dentículos.
    """
    hsh = stable_hash(bid)
    mat = ["stone", "stone", "stone", "brick", "brick", "tan", "iron"][hsh % 7]
    det = np.asarray(det, float)
    det = det if det.ndim == 3 else np.broadcast_to(det, u.shape)[..., None]
    if mat == "brick":
        base = np.array(BRICKS[hsh % 3], float) * (np.asarray(base, float).mean() / 200)
    elif mat == "tan":
        base = np.array(BRICKS[3 + hsh % 2], float) * (np.asarray(base, float).mean() / 200)
    elif mat == "iron":
        base = np.array([214, 208, 192], float) * (np.asarray(base, float).mean() / 200)
    base = np.asarray(base, float)
    c = np.broadcast_to(base, u.shape + (3,)).copy()
    trim = np.array([212, 208, 198]) * (base.mean() / 190) if mat in ("brick", "tan") else base * 1.1
    up = v >= G_FLOOR
    d = hb - v
    rr = (v - G_FLOOR) % FLOOR_H                               # altura dentro del piso
    fl = np.floor((v - G_FLOOR) / FLOOR_H)
    bx = u % BAY
    bay = np.floor(u / BAY)
    # material
    if mat in ("brick", "tan"):
        course = (v % 2.6) < .55
        joint = ((u + (np.floor(v / 2.6) % 2) * 3) % 6) < .5
        c = np.where(((course | joint) & up)[..., None], c * (1 - det * .25) + det * .25 * 205, c)
        c = c + (((np.floor(u / 6) * 7 + np.floor(v / 2.6) * 3) % 5)[..., None] - 2) * 2.5 * det   # ladrillos desparejos
    elif mat == "iron":
        col_ = up & (bx < 2.4)                                # columnas de hierro entre ventanales
        c = np.where(col_[..., None], c * .9, c)
    c = np.where((up & (rr < 1.2))[..., None], c * (1 - det) + trim * .94 * det, c)                 # moldura de piso
    # ventanas
    w0, w1 = (3.0, 15.0) if mat == "iron" else (4.5, 13.5)
    h0, h1 = 6.0, (22.0 if mat == "iron" else 20.0)
    win = up & (bx >= w0) & (bx < w1) & (rr >= h0) & (rr < h1) & (d > 10)
    wid = (fl * 31 + bay * 17 + hsh) % 23
    glass = np.zeros(u.shape + (3,)) + (44, 54, 60)
    glass = glass + ((rr - h0) / (h1 - h0))[..., None] * -10 + ((bx - w0) < 2)[..., None] * 14   # reflejo
    curtain = (wid % 5 == 1) & (rr > (h0 + h1) / 2)
    glass = np.where(curtain[..., None], [196, 186, 164], glass)
    blinds = (wid % 7 == 3) & ((rr % 1.5) < .6)
    glass = np.where(blinds[..., None], [176, 176, 170], glass)
    lit = wid == 11
    glass = np.where(lit[..., None], [206, 196, 150], glass)
    sash = (np.abs(rr - (h0 + h1) / 2) < .55) | ((bx - w0) < .6) | ((w1 - bx) < .6)            # marco de guillotina
    glass = np.where(sash[..., None], [222, 222, 214], glass)
    c = np.where(win[..., None], c * (1 - det) + glass * det, c)
    lint = up & (bx >= w0 - 1) & (bx < w1 + 1) & (rr >= h1) & (rr < h1 + 2.2) & (d > 10)
    sill = up & (bx >= w0 - 1.2) & (bx < w1 + 1.2) & (rr >= h0 - 1.6) & (rr < h0) & (d > 10)
    c = np.where((lint | sill)[..., None], c * (1 - det) + trim * det, c)
    ac = win & (wid % 6 == 2) & (bx >= w0 + 1.5) & (bx < w1 - 1.5) & (rr < h0 + 4)                 # aire acondicionado
    c = np.where(ac[..., None], c * (1 - det) + np.array([150, 152, 150]) * det, c)
    c = np.where((ac & ((bx % 1.2) < .4))[..., None], c - 26 * det, c)
    # escaleras de incendio (ladrillo), dos vanos por edificio y por cada 4 vanos en las caras largas
    if mat in ("brick", "tan"):
        fe_b = np.floor(u / (BAY * 4)) * BAY * 4 + BAY
        t_ = u - fe_b
        in_fe = up & (t_ >= 0) & (t_ < 2 * BAY) & (d > 14) & (v > G_FLOOR + 4)
        plat = in_fe & (rr >= 3) & (rr < 4.4)
        rail = in_fe & (((rr >= 11) & (rr < 11.8)) | (((t_ % 2.2) < .5) & (rr >= 4.4) & (rr < 11)))
        lad = in_fe & (np.abs((rr - 4) - (t_ - 4) * (FLOOR_H / 24)) < .9) & (t_ > 4) & (t_ < 28)
        fe = plat | rail | lad
        c = np.where(fe[..., None], c * (1 - det * .9) + np.array([28, 30, 32]) * det * .9, c)
    # planta baja
    gf = v < G_FLOOR
    e = u % 40
    if kind == "lobby":
        ston = np.array([206, 202, 192]) * (base.mean() / 190 if mat == "stone" else 1)
        c = np.where(gf[..., None], c * (1 - det) + ston * det, c)
        c = np.where((gf & ((v % 4) < .7))[..., None], c - 14 * det, c)                             # sillares
        door = gf & (e > 13) & (e < 27) & (v < 19)
        dc = np.where(((e > 14) & (e < 26) & (v < 18))[..., None], [214, 196, 140], [36, 40, 38])
        dc = np.where(((np.abs(e - 20) < .5) & (v < 18))[..., None], [90, 80, 60], dc)
        c = np.where(door[..., None], c * (1 - det) + dc * det, c)
        tran = gf & (e > 14) & (e < 26) & (v >= 19.5) & (v < 23)                                     # banderola
        c = np.where(tran[..., None], c * (1 - det) + np.array([196, 186, 140]) * det, c)
        can = gf & (e > 9) & (e < 31) & (v >= 24) & (v < 27.5)                                       # marquesina
        c = np.where(can[..., None], c * (1 - det) + np.array(CANOPY[hsh % 4]) * det, c)
        num = gf & (e > 30) & (e < 33) & (v > 12) & (v < 15)                                         # número
        c = np.where(num[..., None], c * (1 - det) + np.array([226, 210, 150]) * det, c)
    else:
        sign = gf & (v >= 24) & (v < 31)                                                             # marquesina de cartel
        sc = np.array(CANOPY[(hsh // 3) % 4]) + 10
        c = np.where(sign[..., None], c * (1 - det) + sc * det, c)
        glyph = sign & (v >= 26) & (v < 29) & (e > 6) & (e < 34) & (((e * 1.7 + hsh) % 3.4) < 1.6)   # (sin letras legibles)
        c = np.where(glyph[..., None], c * (1 - det) + np.array([226, 220, 196]) * det, c)
        frame = gf & (v < 24)
        fc = np.where(((v > 2) & (v < 22.5) & (e > 2) & (e < 38))[..., None], [64, 86, 80], [40, 46, 44])
        fc = np.where(((v > 2) & (v < 22.5) & ((e % 9) < .7) & (e > 2) & (e < 38))[..., None], [40, 46, 44], fc)
        fc = np.where(((e > 16) & (e < 24) & (v < 20))[..., None], [30, 36, 34], fc)                # puerta
        fc = np.where(((v > 12) & (v < 21) & (e > 4) & (e < 15) & ((u * 3 + v) % 7 < 1.2))[..., None], [190, 196, 180], fc)  # reflejo
        c = np.where(frame[..., None], c * (1 - det) + fc * det, c)
        if aw is not None:
            band = gf & (v >= 18) & (v < 24) & (e > 1) & (e < 39)
            stripe = (np.floor(u / 3) % 2)[..., None]
            c = np.where(band[..., None], c * (1 - det) + (np.asarray(aw) * (.85 + .3 * stripe)) * det, c)
    # cornisa con dentículos
    cor = (d < 5) & (d >= 0)
    c = np.where(cor[..., None], c * (1 - det) + trim * 1.08 * det, c)
    c = np.where((cor & (d > 1.5) & (d < 2.8) & ((u % 2.4) < 1.1))[..., None], c - 30 * det, c)
    if kind == "billboard_led":
        c = led_frame(c, u, v, det)
    elif kind.startswith("billboard"):
        design = {"billboard_soda": 0, "billboard_lime": 1, "billboard_fashion": 2}.get(kind, hsh % 3)
        c = billboard(c, u, v, det, design, width)
    elif kind == "grand":
        c = grand_entrance(c, u, v, det, base, width)
    if corner_edge:
        c = np.where((u < 1.2)[..., None], c + 14, c)   # arista de la esquina
    return np.clip(c, 0, 255)


LED_U0, LED_U1, LED_V0, LED_V1 = 4, 60, 36, 58       # pantalla LED (en la cara del edificio)


def led_frame(c, u, v, det):
    """Marco metálico, soportes y fondo apagado de la pantalla LED."""
    fr = (u >= LED_U0 - 2) & (u < LED_U1 + 2) & (v >= LED_V0 - 2) & (v < LED_V1 + 2)
    c = np.where(fr[..., None], c * (1 - det) + np.array([52, 54, 58]) * det, c)
    edge = fr & ((u < LED_U0 - 1.2) | (v >= LED_V1 + 1.2))
    c = np.where(edge[..., None], c * (1 - det) + np.array([120, 124, 128]) * det, c)
    inner = (u >= LED_U0) & (u < LED_U1) & (v >= LED_V0) & (v < LED_V1)
    c = np.where(inner[..., None], c * (1 - det) + np.array([16, 16, 18]) * det, c)
    brk = (v >= LED_V0 - 6) & (v < LED_V0 - 2) & ((np.abs(u - (LED_U0 + 8)) < 1) | (np.abs(u - (LED_U1 - 8)) < 1))
    return np.where(brk[..., None], c * (1 - det) + np.array([40, 42, 44]) * det, c)


def billboard(c, u, v, det, design, width):
    """Cartel publicitario gigante sobre los pisos altos (gráfica sin texto) con reflectores abajo."""
    W_ = width if width else 60.0
    u_ = u if width else u % 60
    u0, u1, v0, v1 = 3.0, W_ - 3.0, G_FLOOR + 3, G_FLOOR + 70
    pu, pv = (u_ - u0) / (u1 - u0), (v - v0) / 26.0                 # pv: 0..1 en la parte visible
    inside = (u_ >= u0) & (u_ < u1) & (v >= v0) & (v < v1)
    border = inside & ((u_ - u0 < 1.5) | (u1 - u_ < 1.5) | (v - v0 < 1.5))
    g_ = np.zeros(u.shape + (3,))
    if design == 0:                                                  # gaseosa: ola blanca y una lata
        g_[:] = (196, 30, 40)
        g_ = np.where((pv > .55)[..., None], [170, 22, 34], g_)
        wave = np.abs(pv - (.45 + .14 * np.sin(pu * 7))) < .09
        g_ = np.where(wave[..., None], [244, 240, 236], g_)
        can = (pu > .66) & (pu < .86) & (pv > .08) & (pv < .95)
        g_ = np.where(can[..., None], [214, 30, 44], g_)
        g_ = np.where((can & (pu > .69) & (pu < .72))[..., None], [250, 200, 200], g_)
        g_ = np.where((can & (pv > .38) & (pv < .5))[..., None], [240, 236, 232], g_)
        g_ = np.where(((pu > .66) & (pu < .86) & (pv >= .95) & (pv < 1.05))[..., None], [196, 198, 202], g_)
    elif design == 1:                                                # lima en rodaja (guiño a Limao)
        g_[:] = np.stack([226 - pv * 40, 220 - pv * 10, 70 + pv * 10], -1)
        cx, cy = .32 * (u1 - u0), .55 * 26.0
        du, dv = (u_ - u0) - cx, (v - v0) - cy
        r = np.hypot(du, dv)
        ang = np.arctan2(dv, du)
        R = 11.5
        g_ = np.where((r < R)[..., None], [60, 140, 40], g_)
        g_ = np.where((r < R - 1.6)[..., None], [196, 236, 110], g_)
        g_ = np.where(((r < R - 2.6) & (np.abs(np.sin(ang * 4)) > .2))[..., None], [226, 250, 150], g_)
        g_ = np.where((r < 1.4)[..., None], [240, 255, 210], g_)
        stripe = (pu > .62) & (pu < .92) & (((pv * 5) % 1) < .35) & (pv > .15) & (pv < .85)
        g_ = np.where(stripe[..., None], [60, 120, 40], g_)
    else:                                                            # moda: perfil sobre fondo oscuro
        g_[:] = (34, 34, 46)
        g_ = np.where((np.abs(pu - pv * .5 - .45) < .05)[..., None], [214, 90, 140], g_)
        hx, hy = .3 * (u1 - u0), .5 * 26.0
        du, dv = (u_ - u0) - hx, (v - v0) - hy
        head = (du / 7) ** 2 + (dv / 8.5) ** 2 < 1
        g_ = np.where(head[..., None], [226, 196, 170], g_)
        hair = ((du + 1.5) / 8) ** 2 + ((dv - 2) / 8.5) ** 2 < 1
        g_ = np.where((hair & ~head)[..., None] | (hair & (du > 2))[..., None], [28, 22, 26], g_)
        g_ = np.where(((np.hypot(du - 22, dv) > 6.5) & (np.hypot(du - 22, dv) < 8))[..., None], [220, 186, 90], g_)
    g_ = np.where(border[..., None], [30, 32, 34], g_)
    c = np.where(inside[..., None], c * (1 - det) + g_ * det, c)
    # reflectores bajo el cartel
    lamp = (v >= v0 - 3) & (v < v0) & ((((u_ - u0) % 14) < 1.2) | ((((u_ - u0) % 14) < 3) & (v >= v0 - 1)))
    lamp = lamp & (u_ >= u0) & (u_ < u1)
    c = np.where(lamp[..., None], c * (1 - det) + np.array([40, 42, 44]) * det, c)
    return c


def grand_entrance(c, u, v, det, base, width):
    """Entrada monumental Beaux-Arts: columnas acanaladas, entablamento, puerta en arco de bronce."""
    W_ = width if width else 40.0
    u_ = u if width else u % 40
    uc = W_ / 2
    stone = np.array([226, 222, 210], float)                    # siempre piedra clara
    zone = v < 60
    sc = np.broadcast_to(stone, u.shape + (3,)).copy()
    sc = np.where((((v % 6) < .6) | ((((u_ + (np.floor(v / 6) % 2) * 5) % 10) < .5)))[..., None], sc - 14, sc)
    steps = v < 3.4
    sc = np.where(steps[..., None], stone * (1 - .06 * np.floor(v))[..., None] + ((v % 1.15) < .35)[..., None] * -30, sc)
    DW = 9.0
    du = u_ - uc
    # puerta en arco
    arch_r = np.hypot(du, v - 30)
    opening = ((np.abs(du) < DW) & (v >= 3.4) & (v < 30)) | ((arch_r < DW) & (v >= 30))
    ring = ((np.abs(du) < DW + 2) & (v >= 3.4) & (v < 30) & ~opening) | ((arch_r < DW + 2) & (arch_r >= DW) & (v >= 30))
    sc = np.where(ring[..., None], stone + 18, sc)
    sc = np.where(((np.abs(du) < 1.6) & (arch_r >= DW - .5) & (arch_r < DW + 2.6) & (v > 36))[..., None], stone + 30, sc)  # clave
    bronze = np.zeros(u.shape + (3,)) + (146, 104, 50)
    panel = ((((v - 3.4) % 8.6) > 1.6) & (((v - 3.4) % 8.6) < 7.2)) & (np.abs(du) > 1.4) & (np.abs(du) < DW - 1.4)
    bronze = np.where(panel[..., None], [112, 78, 38], bronze)
    bronze = np.where((np.abs(du) < .5)[..., None], [70, 50, 26], bronze)
    bronze = np.where(((np.abs(du) > .6) & (np.abs(du) < 1.8) & (v > 14) & (v < 18))[..., None], [236, 200, 96], bronze)  # manijas
    fan = (v >= 26) & opening
    ang = np.arctan2(v - 30, du)
    fanc = np.where(((((ang / np.pi) * 9) % 1) < .18)[..., None] | ((arch_r < 2.4) & (v >= 30))[..., None],
                    [90, 70, 40], [226, 204, 140])
    fanc = np.where((np.abs(v - 26.5) < .6)[..., None], [90, 70, 40], fanc)
    door_c = np.where(fan[..., None], fanc, bronze)
    sc = np.where(opening[..., None], door_c, sc)
    # columnas acanaladas con base y capitel
    for off in (DW + 7, DW + 19):
        for sgn_ in (-1, 1):
            cx_ = uc + sgn_ * off
            if cx_ < 3 or cx_ > W_ - 3:
                continue
            dx_ = u_ - cx_
            shaft = (np.abs(dx_) < 2.6) & (v >= 5.4) & (v < 46)
            col_c = np.where((((dx_ + 2.6) % 1.3) < .4)[..., None], stone - 26, stone + 10)
            col_c = col_c + (dx_ < -1.2)[..., None] * 14 - (dx_ > 1.4)[..., None] * 22                  # volumen
            sc = np.where(shaft[..., None], col_c, sc)
            sc = np.where(((np.abs(dx_) < 3.6) & (v >= 3.4) & (v < 5.4))[..., None], stone + 6, sc)       # base
            sc = np.where(((np.abs(dx_) < 3.8) & (v >= 46) & (v < 50))[..., None], stone + 20, sc)       # capitel
            sc = np.where(((np.abs(np.abs(dx_) - 3) < .7) & (v >= 47) & (v < 49))[..., None], stone - 40, sc)  # volutas
    # faroles a los lados de la puerta
    for sgn_ in (-1, 1):
        lx_ = uc + sgn_ * (DW + 3.6)
        lam = (np.abs(u_ - lx_) < 1.3) & (v >= 22) & (v < 27)
        sc = np.where(lam[..., None], [236, 210, 140], sc)
        sc = np.where(((np.abs(u_ - lx_) < 1.6) & ((np.abs(v - 21.6) < .5) | (np.abs(v - 27.3) < .5)))[..., None], [30, 30, 32], sc)
    # entablamento: arquitrabe, friso con triglifos y cornisa con dentículos
    sc = np.where(((v >= 50) & (v < 52))[..., None], stone + 14, sc)
    sc = np.where(((v >= 52) & (v < 56) & ((u_ % 5) < 1.2))[..., None], stone - 24, sc)
    sc = np.where(((v >= 56) & (v < 58.5))[..., None], stone + 24, sc)
    sc = np.where(((v >= 56.6) & (v < 57.6) & ((u_ % 2.2) < 1))[..., None], stone - 30, sc)
    return np.where(zone[..., None], c * (1 - det) + sc * det, c)


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
    ratio = {"sedan": .31, "taxi": .31, "police": .31, "van": .42, "bus": .27}[kind]
    L, Hh = length * VS, length * ratio * VS
    top_pad = int(Hh * (.14 if kind in ("taxi", "police") else .02))
    im = Image.new("RGBA", (int(L), int(Hh) + top_pad + VS * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda pts: [(x * L, y * Hh + top_pad) for x, y in pts]
    hi, mid, lo, dark = _shade(body, 1.22), body, _shade(body, .78), _shade(body, .5)
    glass, glass_hi = (46, 58, 64), (120, 140, 140)
    # sombra en el piso
    d.ellipse([L * .02, Hh + top_pad - Hh * .06, L * .98, Hh + top_pad + Hh * .07], fill=(0, 0, 0, 120))
    if kind == "bus":                                   # colectivo de la MTA de Nueva York
        white, white_sh, mta_blue = (234, 236, 232), (196, 200, 198), (24, 70, 160)
        tint = (26, 30, 36)
        d.rounded_rectangle([0, top_pad, L - 1, Hh * .88 + top_pad], radius=Hh * .14, fill=white)   # techo redondeado
        d.rectangle(P([(.0, .66), (1, .88)])[0] + P([(1, .88)])[0], fill=white_sh)                    # zócalo
        d.rounded_rectangle(P([(.66, -.02), (.9, .05)])[0] + P([(.9, .05)])[0], radius=VS, fill=(210, 214, 212))  # A/A
        d.polygon(P([(.005, .2), (.05, .1), (.05, .56), (.005, .6)]), fill=tint)                       # parabrisas
        d.line(P([(.012, .25), (.04, .14)]), fill=(110, 130, 134), width=VS)
        d.rectangle(P([(.012, .05), (.1, .1)])[0] + P([(.1, .1)])[0], fill=(30, 24, 18))               # cartel de destino
        for lx_ in np.arange(.016, .096, .008):
            d.rectangle(P([(lx_, .062), (lx_ + .004, .088)])[0] + P([(lx_ + .004, .088)])[0], fill=(255, 150, 40))
        d.rectangle(P([(.12, .14), (.94, .5)])[0] + P([(.94, .5)])[0], fill=tint)                     # banda de ventanas
        for wx_ in np.arange(.2, .94, .085):
            d.line(P([(wx_, .14), (wx_, .5)]), fill=(70, 76, 80), width=max(1, VS // 2))
            d.line(P([(wx_ + .01, .17), (wx_ + .035, .17)]), fill=(90, 104, 110), width=max(1, VS // 2))
        for dx0, dx1 in ((.055, .115), (.5, .56)):                                                     # puertas
            d.rectangle(P([(dx0, .14), (dx1, .84)])[0] + P([(dx1, .84)])[0], fill=(44, 52, 58))
            d.line(P([((dx0 + dx1) / 2, .14), ((dx0 + dx1) / 2, .84)]), fill=(180, 184, 184), width=max(1, VS // 2))
            d.rectangle(P([(dx0, .5), (dx1, .54)])[0] + P([(dx1, .54)])[0], fill=(180, 184, 184))
        d.rectangle(P([(.0, .56), (1, .63)])[0] + P([(1, .63)])[0], fill=mta_blue)                    # franja azul MTA
        d.rectangle(P([(.0, .635), (1, .65)])[0] + P([(1, .65)])[0], fill=(120, 160, 220))
        for gx_ in np.arange(.9, .99, .012):                                                          # rejilla del motor
            d.line(P([(gx_, .68), (gx_, .84)]), fill=(120, 124, 124), width=max(1, VS // 2))
        d.line(P([(.0, .3), (-.01, .3)]), fill=(30, 30, 30), width=VS)                                 # brazo del espejo
        d.rectangle(P([(.0, .7), (.03, .78)])[0] + P([(.03, .78)])[0], fill=(250, 244, 200))           # faro
        d.rectangle(P([(.975, .6), (1, .7)])[0] + P([(1, .7)])[0], fill=(220, 30, 36))                 # luz trasera
        d.rectangle(P([(.0, .82), (1, .88)])[0] + P([(1, .88)])[0], fill=(54, 56, 58))                 # paragolpes/zócalo
        wheels = [(.16, .87), (.74, .87)]
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
near = np.zeros((H, TW, 4))
AVE_TOP = 74                                                          # base de los edificios (vereda de enfrente)
x = 0
# piedra caliza clara, como los monumentos de Washington DC
KIND_OVERRIDE = {0: "grand", 1: "billboard_led", 5: "grand", 7: "billboard_lime"}   # índices de edificios de frente
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
    yy_, xx_ = np.mgrid[0:AVE_TOP, x:min(TW, x + bw)].astype(float)
    kind = KIND_OVERRIDE.get(fi, kind)
    u_f = xx_ - x + .5 if kind == "grand" or kind.startswith("billboard") else xx_ - x + .5 - (bw % 40) / 2
    face = facade(u_f, AVE_TOP - yy_ - .5, AVE_TOP - top, c, aw, 1.0, ("f", fi), kind, width=bw)
    near[0:AVE_TOP, x:x + bw, :3] = face
    near[0:AVE_TOP, x:x + bw, 3] = 255
    near[0:AVE_TOP, x:x + 1, :3] = np.clip(c + 14, 0, 255)            # juntas entre edificios
    near[0:AVE_TOP, min(TW, x + bw) - 1, :3] = c * .78
    blds.append((x, bw, top, c.copy(), aw, kind))
    fi += 1
    x += bw
# segunda mitad del tramo (cuadra larga), con su propio generador para no alterar el resto
_main_random = random
random = random.Random(4242)
while x < TW:
    aw = None
    bw = random.randint(46, 78)
    bw = min(bw, TW - x) if TW - x < 40 else bw
    c = np.array(random.choice(FRONT_PAL), float)
    top = -random.randint(30, 110)
    kind = "lobby" if random.random() < .45 else "shop"
    aw = np.array(random.choice(AWN), float) if kind == "shop" and random.random() < .7 else None
    yy_, xx_ = np.mgrid[0:AVE_TOP, x:min(TW, x + bw)].astype(float)
    kind = KIND_OVERRIDE.get(fi, kind)
    u_f = xx_ - x + .5 if kind == "grand" or kind.startswith("billboard") else xx_ - x + .5 - (bw % 40) / 2
    face = facade(u_f, AVE_TOP - yy_ - .5, AVE_TOP - top, c, aw, 1.0, ("f", fi), kind, width=bw)
    near[0:AVE_TOP, x:x + bw, :3] = face
    near[0:AVE_TOP, x:x + bw, 3] = 255
    near[0:AVE_TOP, x:x + 1, :3] = np.clip(c + 14, 0, 255)
    near[0:AVE_TOP, min(TW, x + bw) - 1, :3] = c * .78
    blds.append((x, bw, top, c.copy(), aw, kind))
    fi += 1
    x += bw
random = _main_random
# avenida entre la vereda de enfrente y la nuestra, con bocacalle en la esquina
GX, GW, VY = 168, 92, 60                                              # hueco de la bocacalle
GC = GX + GW / 2
near[:AVE_TOP, GX:GX + GW, 3] = 0
cv = Image.new("RGBA", (TW, H), (0, 0, 0, 0)); cd = ImageDraw.Draw(cv)
cd.rectangle([0, AVE_TOP, TW, AVE_TOP + 3], fill=(150, 158, 140, 255))      # vereda de enfrente
cd.line([(0, AVE_TOP + 3), (TW, AVE_TOP + 3)], fill=(190, 196, 180, 255))  # cordón
cd.rectangle([0, AVE_TOP + 4, TW, GROUND - 1], fill=(64, 70, 68, 255))     # calzada
for lx in range(0, TW, 12):                                                # línea divisoria
    cd.line([(lx, AVE_TOP + 11), (lx + 6, AVE_TOP + 11)], fill=(210, 204, 150, 255))
cd.line([(0, GROUND - 1), (TW, GROUND - 1)], fill=(44, 50, 48, 255))
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


XW_N, XW_P, XW_W = 9, 7.0, 3.6                       # sendas: 9 rayas enteras, cada 7, de 3.6 de ancho


def zebra(Xs):
    """True donde cae una raya de senda: rayas enteras y centradas en la calzada (sin recortes)."""
    s0 = -((XW_N - 1) * XW_P + XW_W) / 2
    k = np.floor((Xs - s0) / XW_P)
    return (k >= 0) & (k < XW_N) & (((Xs - s0) - k * XW_P) < XW_W)


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
                corner_bid = ("f", next(i_ for i_, b_ in enumerate(blds) if b_ is corner[side]))
                setback, hb, ln = 0, AVE_TOP - top, rnd.uniform(110, 170)
            else:
                setback = rnd.choice([0, 0, 2, 5, 9])
                hb = rnd.choice([rnd.uniform(110, 180), rnd.uniform(160, 260), rnd.uniform(220, 420)])   # escala de NY
                col = np.array(rnd.choice(FRONT_PAL), float) * rnd.uniform(.94, 1.03)
                kind = "lobby" if rnd.random() < .45 else "shop"
                aw = np.array(rnd.choice(AWN), float) if kind == "shop" and rnd.random() < .6 else None
            boxes.append((side, WC + setback, z, z + ln, hb, col, corner_bid if z == Z0 else (side, bid), aw, kind))
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
        dash = (X < 1.4) & ((zg % 50) < 24) & (zg > Z0 + 8)
        g[dash] = (210, 204, 150)
        far_xw = (zg >= 86) & (zg < Z0 - .5) & zebra(sgn * X)                                    # senda de enfrente
        g[far_xw] = (226, 228, 214)
        stop_far = (zg >= 105.5) & (zg < 108.5) & (X < WR - .3)                              # línea de detención (ambos carriles)
        g[stop_far] = (236, 238, 228)
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
SW_ROWS = [GROUND + 3, H]                                               # una sola fila: losas de toda la profundidad
SLAB = 30                                                               # ancho de losa (~1.5 m)
grain = srng.normal(0, 1, (H, TW))
SW_OFF = 7                                                              # juntas alineadas: grilla, no ladrillo
for r0, r1 in zip(SW_ROWS, SW_ROWS[1:]):
    off_ = SW_OFF
    for sx0 in range(-off_, TW, SLAB):
        base_c = np.array([186, 186, 178]) + sr.uniform(-12, 10) + np.array([sr.uniform(-3, 3), 0, sr.uniform(-4, 2)])
        x0_, x1_ = max(0, sx0), min(TW, sx0 + SLAB)
        if x1_ <= x0_:
            continue
        blk = np.broadcast_to(base_c, (r1 - r0, x1_ - x0_, 3)).copy()
        blk += grain[r0:r1, x0_:x1_, None] * 3.2                        # grano del hormigón
        vy = np.linspace(0, 1, r1 - r0)[:, None, None]
        blk *= (1.02 - .05 * vy)                                         # leve desgaste hacia el borde
        near[r0:r1, x0_:x1_, :3] = blk
        if sx0 >= 0:                                                    # junta vertical: sombra + borde claro
            near[r0:r1, sx0, :3] = (100, 104, 98)
            if sx0 + 1 < TW:
                near[r0:r1, sx0 + 1, :3] = np.minimum(near[r0:r1, sx0 + 1, :3] + 16, 255)
        if sr.random() < .22:                                           # grieta
            cxk, cyk = sr.uniform(x0_ + 3, max(x0_ + 4, x1_ - 3)), r0 + 1
            for _ in range(int((r1 - r0) * 1.4)):
                cyk += sr.uniform(.4, 1); cxk += sr.uniform(-1.1, 1.1)
                if r0 <= int(cyk) < r1 and x0_ <= int(cxk) < x1_:
                    near[int(cyk), int(cxk), :3] = (124, 126, 120)
# cordón de granito con el borde iluminado y la cuneta en sombra
near[GROUND, :, :3] = (46, 50, 48)
near[GROUND + 1, :, :3] = (214, 214, 206)
near[GROUND + 2, :, :3] = (168, 170, 164) + grain[GROUND + 2, :, None] * 6
for gx_ in range(0, TW, 47):
    near[GROUND + 1:GROUND + 3, gx_, :3] = (120, 122, 118)
# chicles y manchas
for _ in range(90):
    gx_, gy_ = sr.randrange(0, TW), sr.randrange(GROUND + 4, H)
    near[gy_, gx_, :3] = np.array(near[gy_, gx_, :3]) * .7
for _ in range(5):
    ox_, oy_, rw = sr.randrange(0, TW), sr.randrange(GROUND + 4, GROUND + 12), sr.randrange(4, 9)
    yy_o, xx_o = np.ogrid[GROUND + 3:H, 0:TW]
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
    yy_m, xx_m = np.ogrid[0:H, 0:TW]
    m_ = ((xx_m - cx_) / rx) ** 2 + ((yy_m - cy_) / ry) ** 2
    near[(m_ < 1) & (yy_m >= GROUND + 3), :3] = (88, 90, 88)
    ring = (m_ >= .7) & (m_ < 1) & (yy_m >= GROUND + 3)
    near[ring, :3] = (60, 62, 60)
    grid_ = (m_ < .7) & ((xx_m + yy_m) % 3 == 0) & (yy_m >= GROUND + 3)
    near[grid_, :3] = (110, 112, 108)


diamond_plate(22, H - 12, 22, 9)
diamond_plate(GX + GW + 22, H - 13, 24, 10)
manhole(120, GROUND + 7, 6, 2.4)

# Intersección con una sola perspectiva: la de la calle del fondo. Cada punto del piso se ubica en
# coordenadas reales de esa calle (X a lo ancho, z en profundidad) y se proyecta igual que allá:
#   x = GC + X * (y - VPY) / HC,   z = F * HC / (y - VPY)
#  - la calzada de la calle del fondo sigue cruzando la avenida y nuestra vereda (|X| <= WR);
#  - sendas para cruzarla en la vereda de enfrente y en la nuestra: rayas a lo largo de la calle;
#  - sendas para cruzar la avenida a cada lado de la bocacalle (WR < |X| < WC): rayas a lo largo
#    de la avenida, apiladas en profundidad; la línea divisoria se corta en el cruce.
xs_all = np.arange(TW) + .5
XW_END = H + 50                                           # fin de las rayas de nuestra senda (antes de la franja negra)
for y_ in range(AVE_TOP + 4, H):
    f_ = (y_ + .5 - VPY) / HC
    Xr = (xs_all - GC) / f_                              # posición real a lo ancho de la calle del fondo
    aX = np.abs(Xr)
    z_ = F * HC / (y_ + .5 - VPY)
    road = aX <= WR
    if y_ < GROUND:                                      # sobre la avenida
        if y_ == AVE_TOP + 11:
            near[y_, aX < WC + 2, :3] = (64, 70, 68)     # la línea divisoria se corta en el cruce
        cross = (aX > WR + 1) & (aX < WC - 1) & ((z_ % 6) < 2.6) & (y_ > AVE_TOP + 5) & (y_ < GROUND - 1)
        near[y_, cross, :3] = (220, 222, 210)
        # líneas de detención a ambos lados del cruce, separadas de la senda y a lo ancho de toda la avenida
        sw_ = max(2.2, 2.6 / f_)                          # ancho suficiente para que la diagonal no se corte
        if AVE_TOP + 5 <= y_ < GROUND - 1:
            stop_av = (aX > WC + 5) & (aX <= WC + 5 + sw_)
            near[y_, stop_av, :3] = (236, 238, 228)
    else:                                                # nuestra vereda: en coordenadas de suelo (escala del cordón)
        f_ = (GROUND - VPY) / HC
        Xr = (xs_all - GC) / f_
        aX = np.abs(Xr)
        road = aX <= WR
        near[y_, road, :3] = (70, 76, 74)
        if GROUND + 2 <= y_ < XW_END:
            stripe = zebra(Xr)
            near[y_, stripe, :3] = (226, 228, 214)
        elif XW_END + 4 <= y_ < XW_END + 6:                # línea de detención, separada y a lo ancho de la calle
            near[y_, aX < WR - .5, :3] = (236, 238, 228)
        curb = (aX > WR) & (aX <= WR + max(.8, 1.2 / f_))
        near[y_, curb, :3] = (200, 202, 196)
# hidrante junto al cordón (después de las sendas: queda por delante)
HX, HB = GX - 30, GROUND + 3                             # pegado al cordón
hyd = Image.new("RGBA", (9, 14), (0, 0, 0, 0)); hd = ImageDraw.Draw(hyd)
hd.rectangle([2, 4, 6, 12], fill=(176, 40, 40)); hd.rectangle([1, 12, 7, 13], fill=(140, 30, 30))
hd.ellipse([2, 1, 6, 5], fill=(190, 48, 48)); hd.rectangle([3, 0, 5, 1], fill=(150, 34, 34))
hd.rectangle([0, 6, 8, 7], fill=(160, 36, 36)); hd.point((4, 3), fill=(230, 120, 110))
hd.rectangle([2, 5, 2, 11], fill=(214, 90, 84))
# (va en su propia capa, ordenada por profundidad; ver más abajo)
for _ in range(3):                                                      # postes / farolas
    px = random.choice([random.randrange(10, GX - 12), random.randrange(GX + GW + 8, W - 10)])
    near[30:GROUND + 2, px:px + 2, :3] = (40, 48, 44); near[30:GROUND + 2, px:px + 2, 3] = 255
    near[30:32, px - 4:px + 6, :3] = (40, 48, 44); near[30:32, px - 4:px + 6, 3] = 255
for px in (372, 468, 566):                                              # postes de la cuadra nueva
    near[30:GROUND + 2, px:px + 2, :3] = (40, 48, 44); near[30:GROUND + 2, px:px + 2, 3] = 255
    near[30:32, px - 4:px + 6, :3] = (40, 48, 44); near[30:32, px - 4:px + 6, 3] = 255


# ---------- vegetación
def street_tree(height, seed, young=True):
    """Plátano de sombra de vereda neoyorquina: tronco grueso de corteza manchada, ramas altas y copa
    ovalada, más alta que ancha y densa (dibujado a 4x)."""
    S4 = 4
    Hh = height * S4
    w_ = int(height * .5) * S4
    im = Image.new("RGBA", (w_, int(Hh) + S4), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    rt = random.Random(seed)
    cx_t, base = w_ / 2, Hh
    trunk_top = base - Hh * .5
    lean_ = rt.uniform(-.015, .015) * Hh
    limb(d, (cx_t, base), (cx_t + lean_, trunk_top), .1 * Hh, .07 * Hh, (96, 88, 74))      # tronco grueso
    d.polygon([(cx_t - .07 * Hh, base), (cx_t + .07 * Hh, base), (cx_t + .05 * Hh, base - .04 * Hh),
               (cx_t - .05 * Hh, base - .04 * Hh)], fill=(88, 80, 66))                      # base que se ensancha
    for _ in range(26):                                      # corteza manchada del plátano
        t_ = rt.uniform(.05, 1)
        yy_ = base - t_ * (base - trunk_top)
        xx_ = cx_t + lean_ * t_ + rt.uniform(-.035, .035) * Hh
        r_ = rt.uniform(.008, .018) * Hh
        d.ellipse([xx_ - r_, yy_ - r_ * 1.4, xx_ + r_, yy_ + r_ * 1.4],
                  fill=rt.choice([(176, 168, 140), (150, 146, 120), (70, 66, 56)]))
    d.line([(cx_t + .035 * Hh, base), (cx_t + .03 * Hh + lean_, trunk_top)], fill=(60, 54, 46), width=max(1, int(.012 * Hh)))
    for a_ in (-.55, -.2, .15, .5):                          # ramas altas
        ex, ey = cx_t + lean_ + np.sin(a_) * .2 * Hh, trunk_top - np.cos(a_) * .2 * Hh
        limb(d, (cx_t + lean_, trunk_top + .02 * Hh), (ex, ey), .045 * Hh, .02 * Hh, (88, 80, 68))
    greens = [(40, 74, 42), (54, 92, 50), (70, 112, 58), (88, 132, 66), (110, 150, 78)]
    ccx, ccy = cx_t + lean_, trunk_top - .2 * Hh               # copa ovalada (alta)
    RX, RY = .22 * Hh, .3 * Hh
    for _ in range(110):
        ang = rt.uniform(0, 2 * np.pi)
        rr = rt.uniform(0, 1) ** .55
        lx_, ly_ = ccx + np.cos(ang) * rr * RX, ccy + np.sin(ang) * rr * RY
        r_ = rt.uniform(.03, .055) * Hh
        lvl = (ccy + RY - ly_) / (2 * RY)                    # 0 abajo .. 1 arriba: más luz arriba
        g_ = greens[int(np.clip(lvl * 4 + rt.uniform(-1.2, .8), 0, 4))]
        d.ellipse([lx_ - r_, ly_ - r_, lx_ + r_, ly_ + r_], fill=g_)
    for _ in range(18):                                      # luz en el borde superior izquierdo
        ang = rt.uniform(np.pi * 1.05, np.pi * 1.6)
        lx_, ly_ = ccx + np.cos(ang) * RX * .8, ccy + np.sin(ang) * RY * .8
        r_ = rt.uniform(.015, .028) * Hh
        d.ellipse([lx_ - r_, ly_ - r_, lx_ + r_, ly_ + r_], fill=(146, 184, 100))
    for _ in range(10):                                      # huecos de sombra dentro de la copa
        lx_, ly_ = ccx + rt.uniform(-.6, .6) * RX, ccy + rt.uniform(-.2, .7) * RY
        r_ = rt.uniform(.012, .022) * Hh
        d.ellipse([lx_ - r_, ly_ - r_, lx_ + r_, ly_ + r_], fill=(30, 56, 34))
    return _finish(im, int(w_ / S4), int(im.height / S4))


def planter(width, seed):
    """Macetón de piedra con un arbusto (para las entradas monumentales)."""
    S4 = 4
    im = Image.new("RGBA", (width * S4, int(width * 1.3) * S4), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    W4, H4 = width * S4, im.height
    d.rectangle([W4 * .1, H4 * .55, W4 * .9, H4 - 1], fill=(196, 192, 182))
    d.rectangle([W4 * .05, H4 * .52, W4 * .95, H4 * .6], fill=(214, 210, 200))
    rp = random.Random(seed)
    for _ in range(18):
        x_, y_ = rp.uniform(.15, .85) * W4, rp.uniform(.12, .5) * H4
        r_ = rp.uniform(.12, .2) * W4
        d.ellipse([x_ - r_, y_ - r_, x_ + r_, y_ + r_], fill=rp.choice([(52, 96, 52), (70, 116, 60), (90, 136, 70)]))
    return _finish(im, width, int(H4 / S4))


def bake(im, x0, base_y):
    a_ = np.asarray(im, dtype=float)
    ys_, xs_ = np.nonzero(a_[..., 3] > 0)
    ty, tx = ys_ + int(base_y - im.height + 1), (xs_ + int(x0)) % TW
    ok = (ty >= 0) & (ty < H)
    near[ty[ok], tx[ok], :3] = a_[ys_[ok], xs_[ok], :3]
    near[ty[ok], tx[ok], 3] = 255


FAR_TREE_H = 56                                              # ~5 m a la distancia de la vereda de enfrente
for i_, tx_ in enumerate((296, 430, 560)):                # lejos de la pantalla LED y de la esquina
    t_im = street_tree(FAR_TREE_H, 50 + i_)
    bake(t_im, tx_ - t_im.width / 2, AVE_TOP + 2)
for bi_ in (0, 5):                                           # macetones a los lados de las entradas monumentales
    bx_, bw_ = blds[bi_][0], blds[bi_][1]
    for side_ in (-1, 1):
        pl = planter(7, 70 + bi_ * 2 + side_)
        bake(pl, bx_ + bw_ / 2 + side_ * 23 - pl.width / 2, AVE_TOP + 1)
# canteros de tierra con rejita para los árboles de nuestra vereda (en el piso: toman la perspectiva)
OUR_TREES = (24, 458, 612)
for tx_ in OUR_TREES:
    for x_ in range(tx_ - 7, tx_ + 8):
        xx_ = x_ % TW
        near[GROUND + 3:GROUND + 7, xx_, :3] = (66, 54, 44)
        if (x_ - tx_) % 3 == 0:
            near[GROUND + 3:GROUND + 7, xx_, :3] = (82, 68, 56)
    for x_ in range(tx_ - 8, tx_ + 9):                       # rejita de hierro alrededor
        near[GROUND + 2, x_ % TW, :3] = (30, 30, 32)
        near[GROUND + 7, x_ % TW, :3] = (30, 30, 32)
    for y_ in range(GROUND + 2, GROUND + 8):
        near[y_, (tx_ - 8) % TW, :3] = (30, 30, 32); near[y_, (tx_ + 8) % TW, :3] = (30, 30, 32)

# latas de gaseosa tiradas en la vereda de la cuadra larga (en el piso, toman la perspectiva)
CANS = [(20, 101, (200, 30, 40)), (62, 104, (40, 90, 190)), (118, 99, (200, 30, 40)), (330, 103, (210, 210, 214)),
        (402, 100, (40, 140, 70)), (455, 105, (200, 30, 40)), (512, 101, (210, 210, 214)), (602, 99, (40, 90, 190))]
for cx_c, cy_c, cc in CANS:
    near[cy_c + 2, cx_c:cx_c + 5, :3] = np.array(near[cy_c + 2, cx_c:cx_c + 5, :3]) * .72   # sombra
    near[cy_c:cy_c + 2, cx_c:cx_c + 4, :3] = cc                         # cuerpo de la lata (acostada)
    near[cy_c, cx_c:cx_c + 4, :3] = np.minimum(np.array(cc) + 60, 255)  # brillo
    near[cy_c:cy_c + 2, cx_c + 4, :3] = (210, 212, 216)                 # tapa de aluminio
    near[cy_c + 1, cx_c + 4, :3] = (150, 152, 156)
near_rgb = posterize(np.clip(near[..., :3], 0, 255), 64, 6)
near_full = np.dstack([near_rgb, near[..., 3]]).astype(np.uint8)
near_im = Image.fromarray(near_full, "RGBA")           # (completo, para inspección)
top_arr = near_full.copy(); top_arr[GROUND:, :, 3] = 0
tile = image(Image.fromarray(top_arr, "RGBA"), up=1) + image(Image.fromarray(top_arr, "RGBA"), x=-TW, up=1)
# franja de nuestra vereda (en coordenadas de suelo), que se dibuja aparte con perspectiva
band_im = Image.fromarray(near_full[GROUND:], "RGBA")
# fase del tile: al congelar, la senda queda justo debajo de Claude y Lucas (x ≈ 93..185)
TILE0 = 93 - GX - NEAR_SPEED * F0
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{tile}</g>')

# Pantalla LED de Coca-Cola (como las de Times Square): contenido animado en su propia capa
COKE_RED, WHITE_ = (230, 16, 34), (250, 250, 246)
SCRIPT = {   # letras cursivas en píxeles (alto de mayúscula 9, minúsculas 5)
    "C": ["..xxxx.", ".x....x", "x......", "x......", "x......", "x......", "x......", ".x....x", "..xxxx."],
    "o": [".xxx.", "x...x", "x...x", "x...x", ".xxx."], "c": [".xxx", "x...", "x...", "x...", ".xxx"],
    "a": [".xxx.", "x...x", "x...x", "x..xx", ".xx.x"],
    "l": ["x.", "x.", "x.", "x.", "x.", "x.", "x.", "x.", ".x"], "-": ["xxx"],
}


def wordmark(im, x0, base, col):
    d_ = ImageDraw.Draw(im)
    x = x0
    for ch in "Coca-Cola":
        g = SCRIPT[ch]
        y0 = base - len(g) + 1 if ch != "-" else base - 2
        for yy_, row in enumerate(g):
            for xx_, b in enumerate(row):
                if b == "x":
                    d_.point((x + xx_, y0 + yy_), fill=col)
        x += len(g[0]) + 1
    d_.line([(x0 + 2, base + 2), (x - 6, base + 2)], fill=col)   # subrayado con floritura
    d_.point((x - 5, base + 1), fill=col)


def ribbon(d_, wid, y_mid, col, amp=2.5, ph=0.0, th=2):
    for x in range(wid):
        y = y_mid + amp * np.sin(x / wid * 2 * np.pi * .9 + ph)
        d_.line([(x, y - th / 2), (x, y + th / 2)], fill=col)


def coke_frame(k):
    w_, h_ = LED_U1 - LED_U0, LED_V1 - LED_V0
    im = Image.new("RGB", (w_, h_), COKE_RED); d_ = ImageDraw.Draw(im)
    if k == 0:                                            # marca + cinta blanca
        wordmark(im, 2, 10, WHITE_)
        ribbon(d_, w_, 16.5, WHITE_)
    elif k in (1, 2):                                     # botella contorneada, burbujas y hielo
        ribbon(d_, w_, 19, WHITE_, amp=1.5, ph=1.5)       # la cinta va detrás de la botella
        bxc = 11                                          # centro de la botella
        widths = [3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 6, 5, 5, 5, 6, 7, 7, 7, 7, 6, 6]     # silueta contorneada, fila por fila
        for yy_, wd in enumerate(widths):
            x0_ = bxc - wd // 2
            d_.line([(x0_, yy_ + 1), (x0_ + wd - 1, yy_ + 1)], fill=(64, 28, 22))
            d_.point((x0_, yy_ + 1), fill=(150, 96, 76))                             # brillo del vidrio
        d_.rectangle([bxc - 1, 0, bxc + 1, 1], fill=(206, 206, 210))                 # tapita
        d_.line([(bxc + 1, 8), (bxc + 2, 10)], fill=(220, 180, 170))                 # reflejo del vidrio
        d_.line([(bxc + 2, 15), (bxc + 2, 18)], fill=(200, 150, 140))
        for ix, iy in [(22, 13), (28, 15), (25, 9)]:                                 # hielo
            d_.rectangle([ix, iy, ix + 4, iy + 4], fill=(196, 228, 240))
            d_.line([(ix, iy), (ix + 4, iy)], fill=(250, 255, 255))
            d_.point((ix + 1, iy + 2), fill=(250, 255, 255))
        rb = random.Random(3)
        for _ in range(12):                                                         # burbujas que suben
            bxx, byy = rb.randrange(18, 54), rb.randrange(0, 18)
            d_.point((bxx, (byy - (k - 1) * 4) % 18), fill=(255, 196, 200))
    else:                                                 # invertido: rojo sobre blanco
        d_.rectangle([0, 0, w_, h_], fill=WHITE_)
        wordmark(im, 2, 10, COKE_RED)
        ribbon(d_, w_, 16.5, COKE_RED)
    return im


def led(img):
    """Cada píxel como un LED: punto brillante con separación oscura (dibujado a 4x)."""
    a = np.asarray(img, dtype=float)
    big = np.repeat(np.repeat(a, 4, 0), 4, 1)
    mask = np.zeros((4, 4)); mask[:3, :3] = 1; mask[0, 0] = mask[0, 2] = mask[2, 0] = mask[2, 2] = .8
    m = np.tile(mask, (a.shape[0], a.shape[1]))[..., None]
    out = big * (.3 + .8 * m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


led_b = next(b for i_, b in enumerate(blds) if i_ == 1)
lx0, ly0 = led_b[0] + LED_U0, AVE_TOP - LED_V1
lw, lh = LED_U1 - LED_U0, LED_V1 - LED_V0
frames_led = [led(coke_frame(k)) for k in range(4)]
uri = lambda im_: png_uri(im_)
LCYC = 7.5
sched = [(0, 0, 2.5), (1, 2.5, 3.75), (2, 3.75, 5.0), (3, 5.0, 7.5)]     # (cuadro, desde, hasta)
led_imgs = []
for k, t0, t1 in sched:
    kk = nid("led")
    a0, a1 = t0 / LCYC * 100, t1 / LCYC * 100
    css.append(f"@keyframes {kk}{{0%,{max(a0 - .01, 0):.2f}%{{opacity:{1 if k == 0 else 0}}}{a0:.2f}%{{opacity:1}}"
               f"{a0 + 1.2:.2f}%{{opacity:.55}}{a0 + 2.4:.2f}%,{a1 - .01:.2f}%{{opacity:1}}{a1:.2f}%,100%{{opacity:0}}}}"
               if k else f"@keyframes {kk}{{0%,{a1 - .01:.2f}%{{opacity:1}}{a1:.2f}%,100%{{opacity:0}}}}")
    led_imgs.append(f'<image x="{lx0}" y="{ly0}" width="{lw}" height="{lh}" preserveAspectRatio="none" '
                    f'href="{uri(frames_led[k])}" style="animation:{kk} {LCYC}s infinite steps(1)"/>')
css.append("@keyframes ledscan{0%{transform:translateY(0)}100%{transform:translateY(22px)}}")
scan = (f'<clipPath id="ledclip"><rect x="{lx0}" y="{ly0}" width="{lw}" height="{lh}"/></clipPath>')
defs.append(scan)
glow = (f'<ellipse cx="{lx0 + lw / 2}" cy="{ly0 + lh / 2}" rx="{lw * .75}" ry="{lh * 1.1}" fill="#ff2030" opacity=".10"/>')
led_layer = (f'<g style="animation:walkvis {T}s infinite linear">{glow}{"".join(led_imgs)}'
             f'<g clip-path="url(#ledclip)"><rect x="{lx0}" y="{ly0 - 3}" width="{lw}" height="2" fill="#fff" opacity=".10" '
             f'style="animation:ledscan 2.2s infinite linear"/></g></g>'
             f'<g opacity="0" style="animation:frzvis {T}s infinite linear"><image x="{lx0}" y="{ly0}" width="{lw}" height="{lh}" preserveAspectRatio="none" '
             f'href="{uri(frames_led[0])}"/></g>')
led_layer = "".join(f'<g transform="translate({off} 0)">{led_layer}</g>' for off in (0, -TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{led_layer}</g>')

# Vereda con perspectiva: cada fila del suelo se escala por f(y) = (y - horizonte) / (cordón - horizonte)
# alrededor del centro de la pantalla y se desplaza f(y) veces más rápido (paralaje). Así las juntas y
# la senda fugan hacia el centro y se abren hacia abajo, como si la vereda siguiera bajo el cuadro.
# El ciclo cierra igual: en cada fila el desplazamiento por vuelta es exactamente un tile escalado.
band_id = nid("band")
defs.append(f'<g id="{band_id}">{image(band_im, 0, 0, up=1)}{image(band_im, -TW, 0, up=1)}</g>')
band_rows = []
for r_ in range(H - GROUND):
    y_r = GROUND + r_
    f_r = (y_r + .5 - HORIZON) / (GROUND + .5 - HORIZON)
    cid = nid("br")
    defs.append(f'<clipPath id="{cid}"><rect x="-{TW}" y="{y_r}" width="{3 * TW}" height="1.02"/></clipPath>')
    band_rows.append(f'<g clip-path="url(#{cid})"><g transform="translate({W / 2 * (1 - f_r):.3f} 0) scale({f_r:.4f} 1)">'
                     f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, GROUND)}"><use href="#{band_id}"/>'
                     f'<use href="#{band_id}" x="{TW}"/></g></g></g>')
body.append("".join(band_rows))

# ============================================================ figuras (dibujadas con PIL)
SKIN = [(232, 196, 166), (200, 150, 116), (160, 112, 82), (110, 76, 56), (240, 210, 186)]
HAIR = [(30, 24, 20), (70, 50, 32), (110, 84, 56), (150, 140, 130), (20, 18, 18)]
SUIT = [(58, 64, 62), (44, 50, 60), (84, 80, 72), (70, 76, 70), (36, 40, 40), (96, 98, 92)]


GAIT_N = 8                                               # cuadros por ciclo de marcha
GAIT_K = .5                                              # flexión máxima de rodilla en el balanceo (rad)


def gait_legs(phase, legL, A):
    """Ángulos de las dos piernas en la fase dada (0..1). Devuelve [(muslo, rodilla), ...] y la extensión vertical."""
    legs = []
    for i in (0, 1):
        a_ = 2 * np.pi * (phase + i / 2)
        th = A * np.sin(a_)                              # adelante = positivo
        kap = GAIT_K * max(0.0, np.cos(a_ + np.pi / 4)) ** 1.6   # flexiona al despegar y se estira al apoyar
        legs.append((th, kap))
    return legs


def figure(h, frame, look, pose="walk", phase=None, amp=None):
    """Persona de perfil mirando a la izquierda; ancla abajo-centro en (w/2, h).

    Marcha: con `phase` (0..1) dibuja un cuadro del ciclo: muslo y pantorrilla por pierna, la que avanza
    flexiona la rodilla y levanta el pie, la de apoyo queda recta y define la altura de la cadera (sube y
    baja con el paso); los brazos se balancean opuestos a las piernas. `amp` es el ángulo de cadera.
    """
    u = h / 24
    w = int(12 * u) + 4
    pad_ = int(2 * u) + 2                                    # margen abajo para el sube y baja
    im = Image.new("RGBA", (w, h + 1 + pad_), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w / 2
    coat, shirt, skin, hair = look["coat"], look.get("shirt", (200, 206, 196)), look["skin"], look["hair"]
    shoe = look.get("shoe", (18, 18, 18))
    pants = look.get("pants", tuple(max(0, c - 10) for c in coat))
    hip, foot = 12.5 * u, h - 1
    if phase is None:                                        # poses fijas heredadas
        phase = {0: .25, 1: 0.0}[frame] if pose == "walk" else 0.0
    A = (amp if amp is not None else .36) if pose == "walk" else .05
    legL = (foot - .6 * u) - hip
    th_l, sh_l = legL * .5, legL * .5
    legs = gait_legs(phase, legL, A)
    ext = [th_l * np.cos(t_) + sh_l * np.cos(t_ - k_) for t_, k_ in legs]
    drop = legL - max(ext)                                   # la cadera baja cuando las piernas se abren
    ankles = []
    # piernas
    fem = look.get("female")
    lw = .75 if fem else 1.0                                            # ancho de pierna
    for i, shade in ((1, .8), (0, 1.0)):                                # pierna de atrás primero
        col = tuple(int(c * shade) for c in pants)
        t_, k_ = legs[i]
        kx, ky = cx - th_l * np.sin(t_), hip + th_l * np.cos(t_)
        ax_, ay_ = kx - sh_l * np.sin(t_ - k_), ky + sh_l * np.cos(t_ - k_)
        ankles.append(ax_)
        limb(d, (cx, hip), (kx, ky), 2.9 * lw * u, 2.3 * lw * u, col)          # muslo
        limb(d, (kx, ky), (ax_, ay_), 2.2 * lw * u, 1.5 * lw * u, col)         # pantorrilla
        fy_ = ay_ + .6 * u
        tilt = max(0.0, k_ - .3) * .8 * u                               # la punta baja al despegar
        if look.get("boots"):                                           # caña de la bota sobre la pantorrilla
            bt = tuple(int(c * shade) for c in shoe)
            bx_, by_ = kx + (ax_ - kx) * .45, ky + (ay_ - ky) * .45
            limb(d, (bx_, by_), (ax_, ay_), 2.0 * u, 1.9 * u, bt)
            d.line([(bx_ - .9 * u, by_), (bx_ + .9 * u, by_)], fill=tuple(min(255, c + 28) for c in bt), width=1)  # borde de la caña
            if look.get("laces"):
                for j_ in (.3, .6):
                    lx_, ly_ = bx_ + (ax_ - bx_) * j_, by_ + (ay_ - by_) * j_
                    d.point((lx_ - .7 * u, ly_), fill=look["laces"])
        if look.get("heels"):
            d.polygon([(ax_ - 1.3 * u, fy_ - .4 * u + tilt), (ax_ + .3 * u, fy_ - 1.1 * u), (ax_ + .5 * u, fy_ - .3 * u),
                       (ax_ - 1.3 * u, fy_ + tilt)], fill=shoe)
        else:
            d.polygon([(ax_ - 1.6 * u, fy_ - u + tilt), (ax_ + .8 * u, fy_ - u), (ax_ + .8 * u, fy_), (ax_ - 1.6 * u, fy_ + tilt)],
                      fill=shoe)
            if look.get("sole"):
                d.line([(ax_ - 1.6 * u, fy_ + tilt - .2 * u), (ax_ + .8 * u, fy_ - .2 * u)], fill=look["sole"], width=max(1, int(.4 * u)))
            if look.get("boots"):                                       # suela gruesa de bota
                d.line([(ax_ - 1.7 * u, fy_ + tilt), (ax_ + .9 * u, fy_)], fill=(70, 70, 72), width=max(1, int(.5 * u)))
    stride = abs(ankles[0] - ankles[1]) / 2
    d.rounded_rectangle([cx - 1.8 * u, hip - 1.6 * u, cx + 1.7 * u, hip + 1.3 * u], radius=max(1, int(.8 * u)),
                        fill=pants)                                     # pelvis
    # brazo lejano (detrás del torso)
    arm_far = tuple(int(c * .62) for c in coat)
    arm_pose(d, cx + .3 * u, u, legs[0][0] * ARM_GAIN, arm_far, tuple(int(c * .8) for c in skin),
             (1.6 if fem else 2.0) * u, look.get("short_sleeves"))
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
    if look.get("inner"):                                               # tapado abierto: se ve la prenda de abajo
        inn = look["inner"]
        d.polygon([(cx - 2.0 * u, 4.4 * u), (cx - .7 * u, 4.4 * u), (cx - .8 * u, 12.2 * u), (cx - 2.2 * u, 12.2 * u)], fill=inn)
        if look.get("ribbed"):                                          # tejido acanalado del buzo
            for yy_r in np.arange(5.2, 12, 1.1):
                d.line([(cx - 2.0 * u, yy_r * u), (cx - .8 * u, yy_r * u)], fill=tuple(int(c * .82) for c in inn), width=1)
        if look.get("vest_buttons"):
            for yy_r in (6.4, 8.2, 10.0):
                d.point((cx - 1.5 * u, yy_r * u), fill=look["vest_buttons"])
        d.line([(cx - .7 * u, 4.4 * u), (cx - .8 * u, coat_len - .5 * u)],
               fill=tuple(min(255, int(c * 1.6 + 20)) for c in coat), width=max(1, int(.5 * u)))   # solapa con brillo
    if look.get("coat_sheen"):                                          # reflejo a lo largo de la espalda
        d.line([(cx + 1.2 * u, 5.0 * u), (cx + 1.7 * u + stride * .2, coat_len - 1.5 * u)],
               fill=tuple(min(255, int(c * 1.5 + 18)) for c in coat), width=max(1, int(.5 * u)))
    if look.get("collar_up"):                                           # cuello alto del tapado
        d.polygon([(cx - .9 * u, 3.0 * u), (cx + 1.5 * u, 3.0 * u), (cx + 1.7 * u, 4.8 * u), (cx - .8 * u, 4.8 * u)], fill=coat)
        d.line([(cx - .9 * u, 3.0 * u), (cx - .8 * u, 4.8 * u)], fill=tuple(min(255, int(c * 1.6 + 20)) for c in coat), width=1)
    if look.get("turtleneck"):
        d.rectangle([cx - 1.0 * u, 3.8 * u, cx + .5 * u, 4.6 * u], fill=look["turtleneck"])
    if not look.get("inner"):
        d.polygon([(cx - 1.9 * u, 4.4 * u), (cx - 0.9 * u, 4.4 * u), (cx - 1.4 * u, 7.5 * u)], fill=shirt)   # cuello
    if look.get("tie"):
        d.line([(cx - 1.5 * u, 5 * u), (cx - 1.7 * u, 8 * u)], fill=look["tie"], width=max(1, int(u * .6)))
    # brazo cercano (opuesto a la pierna cercana); el lejano se dibujó detrás del torso
    arm = tuple(int(c * .8 + 34) for c in coat)            # un poco más claro que el abrigo: se lee el balanceo
    aw_ = (1.7 if fem else 2.1) * u
    hand_x, hand_y = arm_pose(d, cx, u, -legs[0][0] * ARM_GAIN, arm, skin, aw_, look.get("short_sleeves"))
    swing = hand_x - cx
    if look.get("bag"):
        d.line([(hand_x, hand_y), (hand_x, hand_y + 1.1 * u)], fill=tuple(int(c * .6) for c in look["bag"]), width=1)
        d.rectangle([hand_x - 1.0 * u, hand_y + 1.1 * u, hand_x + 1.0 * u, hand_y + 3.3 * u], fill=look["bag"])
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
        sw_ = .3 * np.sin(2 * np.pi * phase) * u
        d.polygon([(hx + 1.0 * u, hy - 1.4 * u), (hx + 1.8 * u, hy - 1.1 * u), (hx + 2.4 * u + sw_, hy + 2.6 * u),
                   (hx + 1.7 * u + sw_, hy + 2.8 * u)], fill=hair)
    if look.get("bun"):
        d.ellipse([hx + .9 * u, hy - 2.1 * u, hx + 2.3 * u, hy - .7 * u], fill=hair)
    if look.get("ballcap"):
        d.chord([hx - 1.6 * u, hy - 2.3 * u, hx + 1.7 * u, hy + .4 * u], 180, 360, fill=look["ballcap"])
        d.rectangle([hx - 2.7 * u, hy - 1.05 * u, hx - .6 * u, hy - .6 * u], fill=tuple(int(c * .7) for c in look["ballcap"]))
    if look.get("shades"):
        d.rectangle([hx - 1.6 * u, hy - .5 * u, hx - .2 * u, hy + .2 * u], fill=look["shades"])
        d.point((hx - 1.2 * u, hy - .35 * u), fill=(190, 214, 220))                                # reflejo en el vidrio
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
    out_ = Image.new("RGBA", (w, h + 1), (0, 0, 0, 0))
    out_.paste(im.crop((0, 0, w, h + 1)), (0, int(round(drop))))
    return out_


ARM_GAIN = .55                                           # balanceo de brazos respecto de la cadera (contenido)


def arm_pose(d, sx, u, alpha, col, skin, width, short_sleeves=False):
    """Brazo desde el hombro con ángulo alpha (adelante = positivo) y codo levemente flexionado."""
    up_l, fo_l = 3.1 * u, 2.9 * u
    ex, ey = sx - up_l * np.sin(alpha), 5 * u + up_l * np.cos(alpha)
    beta = alpha + .25 + .35 * max(0.0, alpha)            # el antebrazo acompaña hacia adelante
    hx_, hy_ = ex - fo_l * np.sin(beta), ey + fo_l * np.cos(beta)
    if short_sleeves:
        limb(d, (sx, 5 * u), (ex, ey), width * 1.25, width * 1.05, col)
        limb(d, (ex, ey), (hx_, hy_), width * .95, width * .8, skin)
    else:
        limb(d, (sx, 5 * u), (ex, ey), width * 1.25, width * 1.05, col)       # brazo
        limb(d, (ex, ey), (hx_, hy_), width * 1.0, width * .85, col)          # antebrazo
    d.rectangle([hx_ - .45 * u, hy_ - .3 * u, hx_ + .45 * u, hy_ + .6 * u], fill=skin)  # mano (bloque chico, no '+')
    return hx_, hy_


def claude_figure(h, frame, pose="walk", facing_right=False, phase=None, amp=None):
    """Claude como Morpheus: abrigo largo, anteojos negros y cabeza de chispa naranja."""
    look = dict(coat=(34, 46, 38), shirt=(60, 74, 66), skin=(217, 119, 87), hair=(0, 0, 0, 0),
                pants=(40, 34, 30), coat_len=21, tie=None,
                inner=(92, 30, 40), vest_buttons=(200, 170, 90),                  # chaleco bordó
                coat_sheen=True, boots=True, shoe=(52, 34, 26))                   # tapado de cuero y botas marrones
    base = figure(h, frame, look, pose, phase=phase, amp=amp)
    u = h / 24
    pad = int(1.4 * u)                                   # espacio arriba para las puntas de la chispa
    im = Image.new("RGBA", (base.width, base.height + pad), (0, 0, 0, 0))
    im.paste(base, (0, pad))
    d = ImageDraw.Draw(im)
    w = im.width
    a_b = np.asarray(base)[..., 3] > 0
    top_row = int(np.argmax(a_b.any(1)))                 # la cabeza sube y baja con el paso
    hx, hy = w / 2 - .3 * u, 2.2 * u + pad - .3 * u + max(0, top_row - int(.2 * u))
    # borrar cabeza y dibujar chispa
    d.rectangle([hx - 3.4 * u, 0, hx + 3.4 * u, hy + 1.2 * u], fill=(0, 0, 0, 0))
    orange = (217, 119, 87, 255)
    for a in range(8):
        ang = a * np.pi / 4 + np.pi / 8
        r = 3.1 * u if a % 2 == 0 else 2.3 * u
        d.line([(hx, hy), (hx + np.cos(ang) * r, hy + np.sin(ang) * r)], fill=orange, width=max(1, int(.9 * u)))
    d.ellipse([hx - 1.2 * u, hy - 1.2 * u, hx + 1.2 * u, hy + 1.2 * u], fill=orange)
    d.ellipse([hx - 1.9 * u, hy - .75 * u, hx - .5 * u, hy + .55 * u], fill=(8, 10, 10, 255))     # lentes redondos
    d.ellipse([hx - 1.1 * u, hy - .55 * u, hx + .1 * u, hy + .45 * u], fill=(14, 16, 16, 255))
    d.point((hx - 1.5 * u, hy - .4 * u), fill=(150, 160, 160, 255))                               # reflejo
    d.line([(hx + .1 * u, hy - .2 * u), (hx + 1.4 * u, hy - .4 * u)], fill=(8, 10, 10, 255), width=1)
    if facing_right:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


LUCAS = dict(coat=(30, 32, 38), shirt=(30, 34, 40), skin=(214, 168, 136), hair=(58, 40, 28), curly=True,
             shades=(40, 56, 64), frame=(178, 184, 188), beard=(150, 112, 92), pants=(12, 12, 14), coat_len=21,
             inner=(56, 58, 64), ribbed=True, turtleneck=(56, 58, 64),            # buzo gris oscuro de cuello alto
             collar_up=True, coat_sheen=True,                                       # tapado negro de cuello parado
             boots=True, shoe=(14, 14, 16), laces=(80, 80, 84))                     # borcegos negros


def frozen(im):
    """Versión congelada. Desde que el congelado funde todo a negro (y sólo quedan Claude y Lucas), estas
    siluetas quedan tapadas: devuelve una imagen vacía, que image() omite, para no sumar peso al SVG."""
    return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


def frozen_green(im):
    """(Versión anterior) silueta de código verde."""
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


for i_ in range(GAIT_N):
    a0_, a1_ = i_ / GAIT_N * 100, (i_ + 1) / GAIT_N * 100
    css.append(f"@keyframes wk{i_}{{0%,100%{{opacity:0}}{a0_:.2f}%,{a1_ - .01:.2f}%{{opacity:1}}{a1_:.2f}%{{opacity:0}}}}"
               if i_ else f"@keyframes wk0{{0%,{a1_ - .01:.2f}%{{opacity:1}}{a1_:.2f}%,100%{{opacity:0}}}}")


def gait_params(h, world_speed, cadence=1.1, amax=.42):
    """Amplitud de cadera y duración del ciclo para que el pie de apoyo no patine.

    world_speed en px/s a la escala de la figura. Ciclo = dos pasos; paso = 2 * pierna * sin(A).
    """
    legL = (h - 1 - .6 * h / 24) - 12.5 * h / 24
    step = max(1e-3, abs(world_speed)) * cadence / 2
    A = float(np.clip(np.arcsin(min(.99, step / (2 * legL))), .12, amax))
    cyc = 2 * (2 * legL * np.sin(A)) / max(1e-3, abs(world_speed))
    return A, float(np.clip(cyc, .7, 2.2))


def walk_frames(fn, A):
    return [fn(i / GAIT_N, A) for i in range(GAIT_N)]


def sprite_group(frames, ox, oy, step=.5, freeze_img=None, flip=False, delay=None):
    """frames: cuadros de marcha (2 heredados o GAIT_N); freeze_img: cuadro durante el congelado."""
    fx = f' transform="translate({2 * ox + frames[0].width} 0) scale(-1 1)"' if flip else ""
    if len(frames) == 2:
        a, b = frames
        walk = (f'<g style="animation:fa {step}s infinite steps(1)">{image(a, ox, oy, up=1)}</g>'
                f'<g style="animation:fb {step}s infinite steps(1)">{image(b, ox, oy, up=1)}</g>')
    else:
        dl = -random.Random(len(css)).uniform(0, step) if delay is None else delay
        walk = "".join(f'<g opacity="{1 if i == 0 else 0}" style="animation:wk{i} {step:.3f}s {dl:.3f}s infinite steps(1)">'
                       f'{image(f, ox, oy, up=1)}</g>' for i, f in enumerate(frames))
    s = f'<g style="{WALK_VIS}">{walk}</g>'
    if freeze_img is not None:
        s += f'<g opacity="0" style="{FRZ_VIS}">{image(freeze_img, ox, oy, up=1)}</g>'
    return f'<g{fx}>{s}</g>'


# (la antigua capa verde del congelado se reemplazó por el fundido a negro con lluvia de código, más abajo;
#  esta lluvia vieja se sigue generando sin dibujarse para no alterar la secuencia aleatoria de la escena)
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
rain = None

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
T_IN, T_TURN, T_GONE = 15.0, 17.0, 19.8                 # con la esquina a la vista
LANE_Y = NEAR_LANE                                     # carril cercano (va hacia la derecha)
X_LANE = WR * .5                                       # carril de ida de la calle transversal
z_a = F * HC / (LANE_Y - VPY)
x_turn = VPX + F * X_LANE / z_a                        # en coordenadas del tile
scr_turn = (tile_g(T_TURN) + x_turn) % TW
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
assert int(tile_g(T_IN) // TW) == int(tile_g(T_GONE) // TW), "el tile da la vuelta durante el giro del taxi"
copy_off = round((scr_turn - (tile_g(T_TURN) % TW) - x_turn) / TW) * TW
turn_svg = f'<g transform="translate({copy_off} 0)">{turn_svg}</g>'
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}"><g style="{WALK_VIS}">{turn_svg}</g></g>')

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
    fr = walk_frames(lambda ph, a, look=look: figure(PREF, 0, look, phase=ph, amp=a), .3)
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
    spr = sprite_group(fr, 0, 0, step=1.15, freeze_img=frozen(fr[0]), flip=flip)
    peds.append(f'<g style="transform-origin:0 0;animation:{k} {T}s infinite linear">{spr}</g>')
cross_peds = "".join(peds)
cross_peds += f'<g transform="translate({-TW} 0)">{cross_peds}</g>'
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{cross_peds}</g>')

# ------------------------------------------------ gente en la vereda de enfrente (detrás del tránsito)
FAR_SW_Y = AVE_TOP + 2.5                                 # pies sobre la vereda de enfrente
FAR_H = round(PERSP_K * (FAR_SW_Y - HORIZON))            # ~20 px: misma regla de perspectiva
frng = random.Random(31)
FSPAN, FMARG = W + 60, 30
FSPAN_R = 457                                             # con k = 2: ~38 px/s relativo => ~11 px/s propios (~1 m/s)
far_peds = []
for i in range(7):
    look = crowd_look(KINDS[(i * 7 + 3) % len(KINDS)], frng)
    right = i % 2 == 0
    span_f = FSPAN_R if right else FSPAN
    k = 2 if right else 1
    v = k * span_f / WALK
    world = v - NEAR_SPEED
    A_, cyc_ = gait_params(FAR_H, world)
    fr = walk_frames(lambda ph, a, look=look: figure(FAR_H, 0, look, phase=ph, amp=a), A_)
    spr = sprite_group(fr, 0, -FAR_H, step=cyc_, freeze_img=frozen(fr[0]), flip=right)
    far_peds.append(f'<g style="{track(i * span_f / 7 + frng.uniform(-10, 10), v, span_f, FMARG, FAR_SW_Y + frng.uniform(-.6, .6))}">{spr}</g>')
body.extend(far_peds)

# ------------------------------------------------ un peatón dobla en la esquina y se va por la calle del fondo
# Viene por la vereda de enfrente, dobla en la esquina y sigue por la vereda izquierda de la calle del fondo,
# en perspectiva, hasta perderse en la bruma. En tiempo efectivo: se congela con el resto.
TA0, TTURN, TB1 = 16.0, 19.5, 23.5                       # tiempos efectivos: llega, dobla, desaparece
XS = -(WR + 6)                                            # vereda izquierda de la calle del fondo
z_t = F * HC / (FAR_SW_Y - VPY)
x_corner = VPX + F * XS / z_t                            # en coordenadas del tile
V_A = 11.0                                               # paso tranquilo por la vereda de enfrente (~1 m/s)
kz_t = np.log(1500 / z_t) / (TB1 - TTURN)
look_t = crowd_look("f_coat", random.Random(5))
A_t, cyc_t = gait_params(FAR_H, V_A)
fr_t = walk_frames(lambda ph, a: figure(FAR_H, 0, look_t, phase=ph, amp=a), A_t)
kf_t = []
for t in np.arange(0, T + .001, .2):
    e = eff(t)
    if e < TA0 - .2 or e > TB1:
        x, y, sc, op = x_corner, FAR_SW_Y, 1.0, 0.0
    elif e <= TTURN:
        x, y, sc, op = x_corner - V_A * (TTURN - e), FAR_SW_Y, 1.0, min(1.0, max(0.0, (e - TA0 + .2) / .2))
    else:
        z = z_t * np.exp(kz_t * (e - TTURN))
        sc = z_t / z
        x, y = VPX + F * XS / z, VPY + F * HC / z
        op = max(0.0, 1 - (1 - np.exp(-(z - Z0) / 1100)) * 1.15)
    kf_t.append(f"{pct(t):.2f}%{{transform:translate({x:.1f}px,{y:.1f}px) scale({sc:.3f});opacity:{op:.2f}}}")
kt_ = nid("turnP")
css.append(f"@keyframes {kt_}{{{''.join(kf_t)}}}")
# va en las dos copias del tramo: siempre está cerca de la esquina, así la otra copia queda fuera de cuadro
# y la vuelta del tramo no lo hace saltar
turner_one = (f'<g style="transform-origin:0 0;animation:{kt_} {T}s infinite linear">'
              f'{sprite_group(fr_t, -fr_t[0].width / 2, -FAR_H, step=cyc_t, freeze_img=frozen(fr_t[0]), flip=True)}</g>')
turner = "".join(f'<g transform="translate({off} 0)">{turner_one}</g>' for off in (0, -TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{turner}</g>')

# tamaño real respecto de la gente (persona ~37 px = 1.75 m): auto ~4.6 m de largo
CSPAN, CMARG = W + 320, 160
LANES = [  # (base y, sentido, k: vueltas por ciclo, vehículos [(tipo, largo)])
    (AVE_TOP + 11, -1, -1, [("sedan", 64), ("bus", 150), ("van", 60), ("sedan", 66)]),
    (NEAR_LANE, 1, 3, [("sedan", 76), ("van", 72), ("sedan", 78), ("sedan", 74)]),
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
        col = {"bus": (234, 236, 232), "taxi": (214, 170, 40)}.get(kind) or random.choice(VEH_COLS)   # el bus usa la librea MTA
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
patrol = "".join(f'<g transform="translate({off} 0)">{scene}</g>' for off in (0, -TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{patrol}</g>')

# ============================================================ contenedor de basura desbordado (cuadra larga)
def dumpster(length):
    """Contenedor de carga frontal (de los que levanta el camión), con la tapa abierta y desbordado de bolsas."""
    S_ = 4
    L_ = length * S_
    Hb_ = length * .55 * S_                                  # caja (~1.1 m de alto para ~1.9 m de largo)
    pad = int(Hb_ * .7)                                      # lugar para la montaña de bolsas
    im_ = Image.new("RGBA", (int(L_ + 14 * S_), int(Hb_ + pad + 2 * S_)), (0, 0, 0, 0)); d_ = ImageDraw.Draw(im_)
    ox, oy = 4 * S_, pad
    grn, grn_hi, grn_sh = (40, 78, 58), (66, 108, 82), (24, 48, 36)
    d_.ellipse([ox - 2 * S_, oy + Hb_ - S_, ox + L_ + 6 * S_, oy + Hb_ + 2.4 * S_], fill=(0, 0, 0, 110))      # sombra
    rng_b = random.Random(9)
    bags = [(70, 74, 78), (24, 26, 30), (214, 214, 206), (36, 38, 42), (150, 154, 156), (20, 22, 26)]
    for _ in range(26):                                      # bolsas que desbordan por arriba
        bx_ = ox + rng_b.uniform(-.05, 1.0) * L_
        by_ = oy + rng_b.uniform(-.62, .05) * Hb_
        rw_, rh_ = rng_b.uniform(.12, .22) * L_, rng_b.uniform(.22, .36) * Hb_
        col_b = rng_b.choice(bags)
        d_.ellipse([bx_ - rw_ / 2, by_ - rh_ / 2, bx_ + rw_ / 2, by_ + rh_ / 2], fill=col_b)
        d_.line([(bx_ - rw_ * .2, by_ - rh_ * .3), (bx_ + rw_ * .05, by_ - rh_ * .38)],
                fill=tuple(min(255, c_ + 60) for c_ in col_b), width=S_)                                     # brillo del plástico
        d_.polygon([(bx_ - S_, by_ - rh_ / 2), (bx_ + S_, by_ - rh_ / 2), (bx_, by_ - rh_ / 2 - 1.6 * S_)], fill=col_b)  # nudo
    # caja: algo más ancha arriba, con nervaduras y guías para las horquillas
    d_.polygon([(ox, oy), (ox + L_, oy), (ox + L_ - .04 * L_, oy + Hb_), (ox + .04 * L_, oy + Hb_)], fill=grn)
    d_.rectangle([ox - S_, oy - S_, ox + L_ + S_, oy + S_], fill=grn_hi)                                       # borde superior
    for rx_ in np.linspace(.12, .88, 5):
        d_.rectangle([ox + rx_ * L_ - S_ * .6, oy + 2 * S_, ox + rx_ * L_ + S_ * .6, oy + Hb_ - 2 * S_], fill=grn_sh)
    d_.rectangle([ox + .05 * L_, oy + .34 * Hb_, ox + .95 * L_, oy + .5 * Hb_], fill=grn_sh)                 # guías de las horquillas
    d_.rectangle([ox + .06 * L_, oy + .36 * Hb_, ox + .94 * L_, oy + .4 * Hb_], fill=(14, 22, 18))
    for ru in (.2, .8):                                                                                       # ruedas
        d_.ellipse([ox + ru * L_ - 2 * S_, oy + Hb_ - 1.5 * S_, ox + ru * L_ + 2 * S_, oy + Hb_ + 2.5 * S_], fill=(16, 16, 18))
    # tapas abiertas hacia atrás
    d_.polygon([(ox + .02 * L_, oy - S_), (ox + .5 * L_, oy - S_), (ox + .46 * L_, oy - .5 * Hb_), (ox + .05 * L_, oy - .42 * Hb_)],
               fill=(30, 34, 36))
    d_.polygon([(ox + .5 * L_, oy - S_), (ox + .98 * L_, oy - S_), (ox + .9 * L_, oy - .36 * Hb_), (ox + .54 * L_, oy - .44 * Hb_)],
               fill=(36, 40, 42))
    for bx_, sc_ in ((ox + L_ + 5 * S_, 1.0), (ox - 2 * S_, .8)):                                            # bolsas en el piso
        d_.ellipse([bx_ - 3.4 * S_ * sc_, oy + Hb_ - 6 * S_ * sc_, bx_ + 3.4 * S_ * sc_, oy + Hb_ + .6 * S_], fill=(26, 28, 32))
        d_.line([(bx_ - S_, oy + Hb_ - 4.6 * S_ * sc_), (bx_ + .6 * S_, oy + Hb_ - 5 * S_ * sc_)], fill=(90, 94, 100), width=S_)
    return _finish(im_, int(im_.width / S_), int(im_.height / S_))


dump = dumpster(38)
DX = 506                                                 # en la cuadra larga, contra el cordón
DY = GROUND + 2 - dump.height
dump_svg = (f'<g style="{WALK_VIS}">{image(dump, DX, DY, up=1)}</g>'
            f'<g opacity="0" style="{FRZ_VIS}">{image(frozen(dump), DX, DY, up=1)}</g>')
dump_svg = "".join(f'<g transform="translate({off} 0)">{dump_svg}</g>' for off in (0, -TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{dump_svg}</g>')

# ============================================================ árboles de nuestra vereda (capa propia)
OUR_TREE_H = 92                                              # ~5 m junto al cordón (la copa sale del cuadro)
trees_svg = ""
for i_, tx_ in enumerate(OUR_TREES):
    t_im = street_tree(OUR_TREE_H, 90 + i_)
    trees_svg += image(t_im, tx_ - t_im.width / 2, GROUND + 5 - t_im.height, up=1)
trees_svg = "".join(f'<g transform="translate({off} 0)">{trees_svg}</g>' for off in (0, -TW, TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{trees_svg}</g>')

# ============================================================ hidrante (capa propia)
# Por delante de los autos y sus sombras, por detrás de la gente de la vereda (pies más abajo).
hyd_svg = (f'<g style="{WALK_VIS}">{image(hyd, HX, HB - 14, up=1)}</g>'
           f'<g opacity="0" style="{FRZ_VIS}">{image(frozen(hyd), HX, HB - 14, up=1)}</g>')
hyd_svg = "".join(f'<g transform="translate({off} 0)">{hyd_svg}</g>' for off in (0, -TW))
body.append(f'<g style="{track(TILE0, NEAR_SPEED, TW, 0, 0)}">{hyd_svg}</g>')

# ============================================================ alguien paseando un perro (una sola vez)
def dog(h, phase):
    """Perro tipo beagle de perfil, mirando a la izquierda, trotando (patas en diagonal)."""
    S4 = 4
    L_ = int(h * 1.5) * S4
    im = Image.new("RGBA", (L_, int(h * 1.25) * S4), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    Hs_ = h * S4
    tan, white, black = (184, 128, 70), (232, 226, 212), (40, 32, 28)
    by_ = im.height - .55 * Hs_                              # línea del lomo
    bx0, bx1 = .3 * Hs_, 1.2 * Hs_
    for i_, (lx, sh) in enumerate(((.42, 1.0), (1.05, 1.0), (.5, .75), (1.12, .75))):   # patas: diagonal
        ph_ = phase + (0 if i_ in (0, 3) else .5)
        sw = np.sin(2 * np.pi * ph_) * .16 * Hs_
        lift = max(0.0, np.cos(2 * np.pi * ph_)) * .08 * Hs_
        col = white if sh == 1.0 else (196, 190, 178)
        limb(d, (lx * Hs_, by_ + .1 * Hs_), (lx * Hs_ - sw, im.height - 1 - lift), .1 * Hs_, .08 * Hs_, col)
    d.ellipse([bx0, by_ - .2 * Hs_, bx1, by_ + .25 * Hs_], fill=tan)                 # cuerpo
    d.ellipse([.55 * Hs_, by_ - .2 * Hs_, 1.0 * Hs_, by_ + .02 * Hs_], fill=black)    # montura negra
    d.ellipse([bx0 + .05 * Hs_, by_ + .02 * Hs_, .75 * Hs_, by_ + .25 * Hs_], fill=white)   # pecho y panza
    wag = np.sin(4 * np.pi * phase) * .08 * Hs_
    limb(d, (1.15 * Hs_, by_ - .1 * Hs_), (1.35 * Hs_ + wag * .3, by_ - .45 * Hs_ + wag), .07 * Hs_, .04 * Hs_, tan)  # cola
    d.ellipse([1.3 * Hs_ + wag * .3, by_ - .5 * Hs_ + wag, 1.38 * Hs_ + wag * .3, by_ - .42 * Hs_ + wag], fill=white)
    hx_, hy_ = .22 * Hs_, by_ - .32 * Hs_                    # cabeza y hocico
    d.ellipse([hx_ - .16 * Hs_, hy_ - .16 * Hs_, hx_ + .16 * Hs_, hy_ + .15 * Hs_], fill=tan)
    d.polygon([(hx_ - .1 * Hs_, hy_ - .02 * Hs_), (hx_ - .3 * Hs_, hy_ + .04 * Hs_), (hx_ - .3 * Hs_, hy_ + .14 * Hs_),
               (hx_ - .06 * Hs_, hy_ + .15 * Hs_)], fill=white)
    d.ellipse([hx_ - .33 * Hs_, hy_ + .02 * Hs_, hx_ - .26 * Hs_, hy_ + .08 * Hs_], fill=black)   # nariz
    d.ellipse([hx_ - .02 * Hs_, hy_ - .12 * Hs_, hx_ + .12 * Hs_, hy_ + .14 * Hs_], fill=(120, 76, 40))   # oreja caída
    d.ellipse([hx_ - .12 * Hs_, hy_ - .07 * Hs_, hx_ - .07 * Hs_, hy_ - .02 * Hs_], fill=black)   # ojo
    d.rectangle([hx_ + .06 * Hs_, hy_ + .1 * Hs_, hx_ + .14 * Hs_, hy_ + .18 * Hs_], fill=(200, 40, 40))  # collar
    return _finish(im, int(L_ / S4), int(im.height / S4))


DW_Y = LANE_FAR                                              # carril de enfrente: va hacia la derecha
DW_H = round(PERSP_K * (DW_Y - HORIZON))
DOG_H = max(6, round(DW_H * .26))                            # ~45 cm a la cruz
DW_WORLD = 18.0                                              # paseo tranquilo (~0.9 m/s)
DW_REL = DW_WORLD + NEAR_SPEED
DW_T0 = 2.5                                                  # entra por la izquierda
DW_T1 = DW_T0 + (W + 120) / DW_REL                           # sale por la derecha
dw_look = dict(crowd_look("f_coat", random.Random(77)), coat=(96, 70, 58), long_hair=True, hair=(150, 96, 56))
A_dw, cyc_dw = gait_params(DW_H, DW_WORLD)
dw_frames = walk_frames(lambda ph, a: figure(DW_H, 0, dw_look, phase=ph, amp=a), A_dw)
dog_frames = [dog(DOG_H, i / GAIT_N) for i in range(GAIT_N)]
cyc_dog = .42                                                # trote corto y rápido
walker = sprite_group(dw_frames, 0, -DW_H, step=cyc_dw, flip=True)
pup = sprite_group(dog_frames, 0, 0, step=cyc_dog, flip=True)
hand_x, hand_y = dw_frames[0].width * .7, -DW_H * .52        # mano que lleva la correa (mirando a la derecha)
dog_x = dw_frames[0].width + 10
leash = (f'<path d="M{hand_x:.1f} {hand_y:.1f} Q{(hand_x + dog_x) / 2:.1f} {-DOG_H * .4:.1f} '
         f'{dog_x + 2:.1f} {-DOG_H * .75:.1f}" stroke="#2a2420" stroke-width=".7" fill="none"/>')
kdw = nid("dw")
css.append(f"@keyframes {kdw}{{0%,{pct(DW_T0):.3f}%{{transform:translate(-60px,{DW_Y}px)}}"
           f"{pct(DW_T1):.3f}%,100%{{transform:translate({W + 60:.0f}px,{DW_Y}px)}}}}")
dog_walker = (f'<g style="transform:translate(-60px,{DW_Y}px);animation:{kdw} {T}s infinite linear">'
              f'<g style="{WALK_VIS}">{leash}</g>{walker}'
              f'<g transform="translate({dog_x} {-dog_frames[0].height + 1})">{pup}</g></g>')

# ============================================================ multitud: dos carriles de vereda
# Carril de enfrente: gente que va hacia la derecha (~1 m/s). Carril cercano (el nuestro): gente que va
# hacia la izquierda como Claude y Lucas pero más despacio (~0.5 m/s): ellos la pasan. Nadie está apurado.
SPAN, MARGIN = W + 60, 30
SPAN_FAR_LANE = 572                                       # con k = 2: ~47.7 px/s relativo => ~21 px/s propios
KEEP_OUR_SIDE = {1, 2, 4, 5, 7, 10}                 # menos gente de este lado: 4 en el carril de enfrente, 2 en el nuestro
crowd_back = []
for i in range(11):
    kind = KINDS[i % len(KINDS)]
    look = crowd_look(kind, random)
    h = random.randint(33, 37) if look.get("female") else random.randint(36, 40)
    k = random.choice([-1, 1, 1, 2, 2, 3])                     # (se sortea igual para no alterar la secuencia)
    near_lane = i % 3 == 2
    ly = LANE_NEAR if near_lane else LANE_FAR
    h = round(PERSP_K * (ly - HORIZON) * h / 38)               # escala de la regla de perspectiva
    span_i = SPAN if near_lane else SPAN_FAR_LANE
    k = 1 if near_lane else 2
    v = k * span_i / WALK                                       # relativo a la cámara (siempre hacia la derecha)
    world = v - NEAR_SPEED                                      # propio: + derecha, - izquierda
    flip = world > 0                                            # mira hacia donde camina
    A_, cyc_ = gait_params(h, world)
    fr = walk_frames(lambda ph, a, h=h, look=look: figure(h, 0, look, phase=ph, amp=a), A_)
    s = sprite_group(fr, 0, -h, step=cyc_, freeze_img=frozen(fr[0]), flip=flip)
    random.choice([.45, .5, .55])                               # (mismo consumo aleatorio que antes)
    x0_c = i * span_i / 11 + random.uniform(-8, 8)
    if i not in KEEP_OUR_SIDE:                         # (se generan igual para no alterar la secuencia aleatoria)
        continue
    crowd_back.append((ly, f'<g style="{track(x0_c, v, span_i, MARGIN, ly)}">{s}</g>'))
crowd_back.append((LANE_FAR, dog_walker))                    # paseador de perro (una sola vez por vuelta)
crowd_back = [g_ for _, g_ in sorted(crowd_back, key=lambda c: c[0])]   # el carril de enfrente se dibuja primero
body.extend(crowd_back)

# ============================================================ la mujer del vestido rojo (y el agente)
def woman_in_red(h, frame, phase=None, amp=.4):
    """Silueta de reloj de arena: vestido rojo ajustado a la rodilla, pelo rubio largo, tacos."""
    u = h / 24
    w = int(12 * u) + 6
    im = Image.new("RGBA", (w, h + 1), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx = w / 2
    skin, skin_sh = (238, 204, 176), (214, 172, 146)
    red, red_hi, red_sh = (212, 22, 44), (244, 70, 84), (150, 10, 30)
    hair, hair_hi = (232, 192, 104), (250, 226, 150)
    if phase is None:
        phase = {0: .25, 1: 0.0}[frame]
    sway = .35 * np.sin(2 * np.pi * phase) * u            # balanceo de cadera
    foot = h - 1
    knee = 15.6 * u
    hip_w = 11.5 * u
    legL = (foot - 1.0 * u) - hip_w
    legs = gait_legs(phase, legL, amp)
    ext = [legL * .5 * np.cos(t_) + legL * .5 * np.cos(t_ - k_) for t_, k_ in legs]
    drop = legL - max(ext)
    # piernas (de atrás primero), largas y finas, con tacos y rodilla
    for i_, col in ((1, skin_sh), (0, skin)):
        t_, k_ = legs[i_]
        hxp = cx + sway
        kx, ky = hxp - legL * .5 * np.sin(t_), hip_w + legL * .5 * np.cos(t_)
        fx, fy = kx - legL * .5 * np.sin(t_ - k_), ky + legL * .5 * np.cos(t_ - k_) + 1.0 * u
        limb(d, (hxp, hip_w), (kx, ky), 2.1 * u, 1.6 * u, col)
        limb(d, (kx, ky), (fx, fy - 1.0 * u), 1.5 * u, 1.0 * u, col)
        d.polygon([(fx - 1.2 * u, fy - .6 * u), (fx + .3 * u, fy - 1.2 * u), (fx + .5 * u, fy - .4 * u),
                   (fx - 1.2 * u, fy)], fill=(176, 12, 34))                                     # zapato
        d.line([(fx + .45 * u, fy - .5 * u), (fx + .5 * u, fy)], fill=(120, 6, 24), width=1)    # taco
    # pelo largo por detrás (cae hasta media espalda y se mueve)
    hx, hy = cx - .2 * u, 2.3 * u
    flow = .35 * np.sin(2 * np.pi * phase + 1) * u
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
    sw = .7 * np.sin(2 * np.pi * phase) * u               # opuesto a la pierna cercana (contenido)
    limb(d, (cx + .1 * u, 4.8 * u), (cx + sw * .5, 7.6 * u), 1.4 * u, 1.2 * u, skin)
    limb(d, (cx + sw * .5, 7.6 * u), (cx + sw, 10.2 * u), 1.2 * u, 1.0 * u, skin)
    d.rectangle([cx + sw - .7 * u, 10.0 * u, cx + sw + .6 * u, 10.9 * u], fill=(30, 20, 24))
    # cabeza
    d.rectangle([cx - .55 * u, 3.4 * u, cx + .25 * u, 4.4 * u], fill=skin)                         # cuello fino
    d.ellipse([hx - 1.3 * u, hy - 1.7 * u, hx + 1.3 * u, hy + 1.6 * u], fill=skin)
    d.chord([hx - 1.5 * u, hy - 2.0 * u, hx + 1.6 * u, hy + 1.1 * u], 170, 360, fill=hair)
    d.polygon([(hx - 1.5 * u, hy - 1.1 * u), (hx - .4 * u, hy - 1.9 * u), (hx - .9 * u, hy - .2 * u)], fill=hair)  # flequillo
    d.line([(hx - .3 * u, hy - 1.7 * u), (hx + 1.2 * u, hy - 1.2 * u)], fill=hair_hi, width=1)
    d.point((hx - .9 * u, hy - .3 * u), fill=(40, 30, 40))                                         # ojo
    d.line([(hx - 1.2 * u, hy + .75 * u), (hx - .75 * u, hy + .75 * u)], fill=(200, 16, 40), width=1)  # labios
    out_ = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out_.paste(im, (0, int(round(drop))))                 # el pie de apoyo toca el piso
    return out_


AGENT = dict(coat=(36, 40, 40), shirt=(226, 230, 222), skin=(214, 172, 140), hair=(28, 24, 20), tie=(8, 8, 8),
             shades=(6, 6, 8), earpiece=(200, 200, 190), pants=(30, 32, 34), coat_len=13.5)
WY = LANE_FAR
WH, APPEAR, TURN = round(PERSP_K * (WY - HORIZON)), 14.3, 18.6                                            # carril medio: entre la multitud y los protagonistas
W_WORLD = 47 - NEAR_SPEED                                # su velocidad propia (WV relativo menos la cámara)
A_w, cyc_w = gait_params(WH, W_WORLD)
wred = walk_frames(lambda ph, a: woman_in_red(WH, 0, phase=ph, amp=a), A_w)
wagt = walk_frames(lambda ph, a: figure(WH, 0, AGENT, phase=ph, amp=a), A_w)
# aparece de golpe (con un parpadeo) y cruza; al "mirá de nuevo" se vuelve agente
css.append(f"@keyframes wred{{0%,{pct(APPEAR) - .001:.3f}%{{opacity:0}}{pct(APPEAR):.3f}%{{opacity:1}}"
           f"{pct(APPEAR + .08):.3f}%{{opacity:.15}}{pct(APPEAR + .16):.3f}%,{pct(TURN) - .001:.3f}%{{opacity:1}}"
           f"{pct(TURN):.3f}%,100%{{opacity:0}}}}")
css.append(f"@keyframes wagt{{0%,{pct(TURN) - .001:.3f}%{{opacity:0}}{pct(TURN):.3f}%{{opacity:1}}{pct(TURN + .1):.3f}%{{opacity:.2}}"
           f"{pct(TURN + .2):.3f}%{{opacity:1}}{pct(TURN + .3):.3f}%{{opacity:.4}}{pct(TURN + .4):.3f}%,99.9%{{opacity:1}}100%{{opacity:0}}}}")
w_red = f'<g opacity="0" style="animation:wred {T}s infinite linear">{sprite_group(wred, 0, -WH, step=cyc_w, flip=True)}</g>'
w_agt = (f'<g opacity="0" style="animation:wagt {T}s infinite linear">'
         f'{sprite_group(wagt, 0, -WH, step=cyc_w, freeze_img=frozen(wagt[0]), flip=True)}</g>')
WV, WX0 = 47, -10                                   # más rápida que la cámara; pasa junto a Lucas ~17.6 s
x_at = lambda t: WX0 + WV * (t - APPEAR)
kw = nid("wk")
css.append(f"@keyframes {kw}{{0%,{pct(APPEAR):.3f}%{{transform:translate({WX0}px,{WY}px)}}"
           f"{pct(F0):.3f}%,{pct(F1):.3f}%{{transform:translate({x_at(F0):.1f}px,{WY}px)}}"
           f"100%{{transform:translate({x_at(F0) + WV * (T - F1):.1f}px,{WY}px)}}}}")
body.append(f'<g style="transform:translate({WX0}px,{WY}px);animation:{kw} {T}s infinite linear">{w_red}{w_agt}</g>')

# ============================================================ protagonistas
FEET = LANE_NEAR
PH = round(PERSP_K * (FEET - HORIZON))                  # misma regla que el resto de la gente


# ============================================================ protagonistas (diseño propio, a 4x)
HERO_STYLE = {
    "neo": dict(coat=(16, 16, 19), coat_hi=(96, 98, 110), inner=(40, 40, 46), inner_hi=(60, 60, 68), pants=(16, 16, 18),
                boot=(14, 14, 16), boot_hi=(60, 60, 66), skin=(212, 166, 134), skin_sh=(170, 124, 98),
                hair=(56, 38, 26), hair_hi=(92, 66, 46), flare=.05, hem=.92, collar="mao", buttons=False,
                lean=.075, arm=.5),                                   # decidido: inclinado hacia adelante
    "morpheus": dict(coat=(18, 22, 20), coat_hi=(88, 104, 96), inner=(96, 30, 42), inner_hi=(140, 52, 64),
                     pants=(34, 30, 28), boot=(54, 36, 26), boot_hi=(100, 72, 54), skin=(217, 119, 87),
                     skin_sh=(176, 88, 62), hair=None, hair_hi=None, flare=.08, hem=.93, collar="lapel", buttons=True,
                     lean=.02, arm=.28),                              # erguido y solemne
}


def _poly(d, pts, col):
    d.polygon([(float(x), float(y)) for x, y in pts], fill=col)


def _seg(d, p0, p1, w0, w1, col):
    limb(d, p0, p1, w0, w1, col)


HEADS = {
    # mira a la izquierda. H pelo, h brillo del pelo, S piel, s sombra, B barba, G lente, M marco, E oreja, K lente
    "neo": (["..HhHHH.",
             ".HhHHhHH",
             "HHHHHHhH",
             "HSHHHHHH",
             "GGGMMEHH",
             "SSSSsEHH",
             ".SSSSsH.",
             ".BBBBs..",
             "..BBss..",
             "...ss..."],
            {"H": (54, 36, 24), "h": (98, 70, 48), "S": (212, 166, 134), "s": (176, 130, 102), "B": (128, 92, 70),
             "G": (24, 32, 38), "M": (184, 190, 194), "E": (190, 140, 110)}),
    "morpheus": (["....o....",
                  ".o..o..o.",
                  "..o.o.o..",
                  "...oOo...",
                  "oKKkOOooo",
                  "...oOo...",
                  "..o.o.o..",
                  ".o..o..o.",
                  "....d....",
                  "...dd...."],
                 {"o": (232, 124, 88), "O": (252, 176, 138), "K": (12, 14, 14), "k": (60, 64, 64), "d": (176, 88, 62)}),
}


def claude_spark(size=13):
    """Asterisco tipo logo de Claude: rayos en cuña que se afinan, largos levemente desparejos."""
    S8 = 8
    R = size * S8 / 2
    big = Image.new("L", (size * S8, size * S8), 0); d8 = ImageDraw.Draw(big)
    c8 = size * S8 / 2
    rays = [1.0, .84, .96, .8, 1.0, .88, .93, .82]           # 8 rayos gruesos: se leen a 13 px
    for i_, ln in enumerate(rays):
        ang = i_ * 2 * np.pi / len(rays) + .13
        tip = (c8 + np.cos(ang) * R * ln, c8 + np.sin(ang) * R * ln)
        side = .6
        b1 = (c8 + np.cos(ang + side) * R * .25, c8 + np.sin(ang + side) * R * .25)
        b2 = (c8 + np.cos(ang - side) * R * .25, c8 + np.sin(ang - side) * R * .25)
        d8.polygon([b1, tip, b2], fill=255)
    d8.ellipse([c8 - R * .22, c8 - R * .22, c8 + R * .22, c8 + R * .22], fill=255)
    a = np.asarray(big.resize((size, size), Image.BOX), dtype=float) / 255
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px_ = out.load()
    yy_, xx_ = np.mgrid[0:size, 0:size]
    rr_ = np.hypot(xx_ - (size - 1) / 2, yy_ - (size - 1) / 2)
    for y_ in range(size):
        for x_ in range(size):
            if a[y_, x_] > .3:
                col = (240, 150, 114) if rr_[y_, x_] < size * .14 else ((217, 119, 87) if a[y_, x_] > .55 else (196, 102, 74))
                px_[x_, y_] = col + (255,)
    return out


def stamp_head(im, kind, neck_x, neck_y):
    if kind == "morpheus":                                   # cara de Claude: el logo, con lentes redondos
        sp = claude_spark(13)
        x0, y0 = int(round(neck_x - sp.width / 2 - .5)), int(round(neck_y - sp.height - 1))
        px_ = im.load()
        K_, k_ = (12, 14, 14, 255), (44, 48, 48, 255)
        for (xx_, yy_), col in (((3, 5), K_), ((4, 5), K_), ((3, 6), K_), ((4, 6), k_), ((5, 6), (90, 94, 94, 255)),
                                ((6, 5), K_), ((7, 5), K_), ((6, 6), K_), ((7, 6), k_), ((8, 6), (90, 94, 94, 255))):
            sp.putpixel((xx_, yy_), col)
        sp.putpixel((3, 5), (150, 160, 160, 255))                                                # reflejo
        im.alpha_composite(sp, (max(0, x0), max(0, y0)))
        for yy_ in range(2):                                                                     # cuello
            for xx_ in range(-1, 1):
                X_, Y_ = int(round(neck_x)) + xx_, int(round(neck_y)) - yy_
                if 0 <= X_ < im.width and 0 <= Y_ < im.height:
                    px_[X_, Y_] = (176, 88, 62, 255)
        return
    rows, pal = HEADS[kind]
    px_ = im.load()
    h_, w_ = len(rows), len(rows[0])
    x0 = int(round(neck_x - w_ / 2 + (0 if kind == "neo" else 0)))
    y0 = int(round(neck_y - h_ + 1))
    for yy_, row in enumerate(rows):
        for xx_, ch in enumerate(row):
            if ch != ".":
                X_, Y_ = x0 + xx_, y0 + yy_
                if 0 <= X_ < im.width and 0 <= Y_ < im.height:
                    px_[X_, Y_] = pal[ch] + (255,)


def hero(kind, H_, phase=0.0, amp=.36, facing_right=False):
    """Claude (Morpheus) o Lucas (Neo) de perfil, caminando hacia la izquierda. Dibujado a 4x y reducido."""
    st = HERO_STYLE[kind]
    S_ = 4
    Hs = H_ * S_
    pad_top = int(.08 * Hs)                                   # lugar para el pelo / la chispa
    Wc = int(.62 * Hs)
    img = Image.new("RGBA", (Wc, int(Hs + pad_top + .1 * Hs)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = Wc * .5
    Y = lambda f: pad_top + f * Hs                            # alto relativo desde la coronilla
    X = lambda f: cx + f * Hs                                 # + atrás, - adelante (mira a la izquierda)
    # --- marcha: muslo, rodilla, tobillo
    hip_y = Y(.52)
    th_l, sh_l = .205 * Hs, .225 * Hs
    legs = gait_legs(phase, th_l + sh_l, amp)
    ext = [th_l * np.cos(t) + sh_l * np.cos(t - k) for t, k in legs]
    drop = (th_l + sh_l) - max(ext)
    hip_y += drop
    Yd = lambda f: Y(f) + drop                                # el cuerpo baja con el paso
    pts = []
    for t, k in legs:
        kx, ky = cx - th_l * np.sin(t), hip_y + th_l * np.cos(t)
        ax, ay = kx - sh_l * np.sin(t - k), ky + sh_l * np.cos(t - k)
        pts.append(((kx, ky), (ax, ay), t, k))
    dark = lambda c, f=.7: tuple(int(v * f) for v in c)

    def leg(i, shade):
        (kx, ky), (ax, ay), t, k = pts[i]
        pc = dark(st["pants"], shade)
        _seg(d, (cx, hip_y), (kx, ky), .095 * Hs, .075 * Hs, pc)
        _seg(d, (kx, ky), (ax, ay), .072 * Hs, .058 * Hs, pc)
        # bota: caña + pie con punta hacia adelante y suela gruesa
        bc, bh = dark(st["boot"], shade), dark(st["boot_hi"], shade)
        bx0, by0 = kx + (ax - kx) * .55, ky + (ay - ky) * .55
        _seg(d, (bx0, by0), (ax, ay), .074 * Hs, .07 * Hs, bc)
        toe = max(0.0, k - .25) * .05 * Hs
        foot = [(ax + .035 * Hs, ay - .02 * Hs), (ax - .03 * Hs, ay - .03 * Hs), (ax - .115 * Hs, ay + .01 * Hs + toe),
                (ax - .12 * Hs, ay + .045 * Hs + toe), (ax + .04 * Hs, ay + .045 * Hs)]
        _poly(d, foot, bc)
        d.line([(ax - .12 * Hs, ay + .045 * Hs + toe), (ax + .04 * Hs, ay + .045 * Hs)], fill=(58, 58, 60), width=int(.016 * Hs))
        d.line([(ax - .09 * Hs, ay - .005 * Hs + toe * .6), (ax - .02 * Hs, ay - .028 * Hs)], fill=bh, width=max(1, int(.008 * Hs)))

    lean = st["lean"]
    XL = lambda f, yf: X(f) - (.52 - yf) * Hs * np.tan(lean)  # el torso se inclina como bloque desde la cadera

    def arm(alpha, col, hi, skin_c):
        sx, sy = XL(.005, .205), Yd(.205)
        up_l, fo_l = .165 * Hs, .15 * Hs
        ex, ey = sx - up_l * np.sin(alpha), sy + up_l * np.cos(alpha)
        beta = alpha + .3
        hx_, hy_ = ex - fo_l * np.sin(beta), ey + fo_l * np.cos(beta)
        _seg(d, (sx, sy), (ex, ey), .085 * Hs, .07 * Hs, col)
        _seg(d, (ex, ey), (hx_, hy_), .07 * Hs, .06 * Hs, col)
        d.line([(sx - .03 * Hs, sy + .02 * Hs), (ex - .028 * Hs, ey)], fill=hi, width=max(1, int(.01 * Hs)))
        d.ellipse([hx_ - .028 * Hs, hy_ - .01 * Hs, hx_ + .028 * Hs, hy_ + .05 * Hs], fill=skin_c)   # mano

    coat, coat_hi = st["coat"], st["coat_hi"]
    a_near = -legs[0][0] * st["arm"]
    hem_y = Yd(st["hem"])
    ank = [pts[0][1][0], pts[1][1][0]]
    front = 0 if ank[0] < ank[1] else 1                       # la pierna que va adelante (x menor)
    back = 1 - front
    shade = {0: 1.0, 1: .72}                                  # 0 = pierna cercana, 1 = lejana
    # brazo lejano y pierna de atrás (queda dentro del tapado)
    arm(-a_near, dark(coat, .6), dark(coat_hi, .6), st["skin_sh"])
    leg(back, shade[back])
    # el tapado: una sola pieza que cae detrás de la pierna trasera y se arrastra como capa
    fl = st["flare"] * Hs
    trail = (.06 + .03 * np.sin(2 * np.pi * phase) ** 2) * Hs
    hem_front_x = (ank[front] + ank[back]) / 2 + .015 * Hs
    skirt = [(XL(-.07, .43), Yd(.43)), (XL(.085, .43), Yd(.43)), (ank[back] + .07 * Hs + fl + trail, hem_y - .012 * Hs),
             (hem_front_x, hem_y + .008 * Hs)]
    _poly(d, skirt, coat)
    for f_ in (.3, .55, .78):                                 # reflejos alargados del cuero
        p0_ = (skirt[0][0] + (skirt[1][0] - skirt[0][0]) * f_, skirt[0][1])
        p1_ = (skirt[3][0] + (skirt[2][0] - skirt[3][0]) * f_, skirt[3][1])
        d.line([(p0_[0] + (p1_[0] - p0_[0]) * .1, p0_[1] + (p1_[1] - p0_[1]) * .1),
                (p0_[0] + (p1_[0] - p0_[0]) * .75, p0_[1] + (p1_[1] - p0_[1]) * .75)],
               fill=dark(coat_hi, .5 if f_ != .55 else .7), width=max(1, int(.01 * Hs)))
    d.line([skirt[3], skirt[2]], fill=dark(coat, .7), width=max(1, int(.01 * Hs)))          # ruedo
    d.line([skirt[0], skirt[3]], fill=dark(coat_hi, .8), width=max(1, int(.01 * Hs)))       # canto de la abertura
    # la pierna que avanza sale por la abertura del tapado
    leg(front, shade[front])
    torso = [(XL(-.035, .165), Yd(.165)), (XL(.06, .175), Yd(.175)), (XL(.085, .25), Yd(.25)), (XL(.08, .45), Yd(.45)),
             (XL(-.085, .45), Yd(.45)), (XL(-.1, .3), Yd(.3)), (XL(-.09, .22), Yd(.22))]
    _poly(d, torso, coat)
    d.line([(XL(.075, .2), Yd(.2)), (XL(.075, .44), Yd(.44))], fill=dark(coat, .7), width=max(1, int(.016 * Hs)))
    d.line([(XL(.035, .2), Yd(.2)), (XL(.045, .42), Yd(.42))], fill=dark(coat_hi, .55), width=max(1, int(.01 * Hs)))
    inner = [(XL(-.07, .17), Yd(.17)), (XL(-.03, .17), Yd(.17)), (XL(-.045, .44), Yd(.44)), (XL(-.085, .44), Yd(.44)),
             (XL(-.095, .28), Yd(.28))]
    _poly(d, inner, st["inner"])
    d.line([(XL(-.088, .24), Yd(.24)), (XL(-.083, .42), Yd(.42))], fill=st["inner_hi"], width=max(1, int(.012 * Hs)))
    if kind == "morpheus":                                                            # chaleco con botones y camisa
        for yy in (.27, .32, .37, .42):
            d.ellipse([XL(-.072, yy) - .008 * Hs, Yd(yy) - .008 * Hs, XL(-.072, yy) + .008 * Hs, Yd(yy) + .008 * Hs],
                      fill=(206, 172, 96))
        _poly(d, [(XL(-.07, .17), Yd(.17)), (XL(-.035, .17), Yd(.17)), (XL(-.055, .24), Yd(.24))], (40, 44, 46))
    else:                                                                             # suéter: tejido
        for yy in np.arange(.2, .44, .035):
            d.line([(XL(-.09, yy), Yd(yy)), (XL(-.05, yy), Yd(yy))], fill=dark(st["inner"], .8), width=max(1, int(.006 * Hs)))
    d.line([(XL(-.03, .17), Yd(.17)), (XL(-.045, .45), Yd(.45))], fill=coat_hi, width=max(1, int(.016 * Hs)))
    if st["buttons"]:                                                                 # cruzado: doble fila
        for yy in (.28, .34, .40):
            for xx in (-.02, .015):
                d.ellipse([XL(xx, yy) - .007 * Hs, Yd(yy) - .007 * Hs, XL(xx, yy) + .007 * Hs, Yd(yy) + .007 * Hs], fill=(14, 18, 16))
    # panel delantero del tapado: angosto, va por delante de la pierna que avanza
    fpanel = [(XL(-.09, .43), Yd(.43)), (XL(-.05, .43), Yd(.43)), (ank[front] - .012 * Hs, hem_y),
              (ank[front] - .07 * Hs, hem_y + .006 * Hs)]
    _poly(d, fpanel, coat)
    d.line([fpanel[0], fpanel[3]], fill=coat_hi, width=max(1, int(.012 * Hs)))
    arm(a_near, tuple(min(255, int(v * 1.15 + 6)) for v in coat), coat_hi, st["skin"])
    if st["collar"] == "mao":
        _poly(d, [(XL(-.035, .125), Yd(.125)), (XL(.045, .12), Yd(.12)), (XL(.06, .185), Yd(.185)), (XL(-.04, .185), Yd(.185))], coat)
        d.line([(XL(-.035, .125), Yd(.125)), (XL(-.04, .185), Yd(.185))], fill=coat_hi, width=max(1, int(.012 * Hs)))
    else:
        _poly(d, [(XL(-.04, .16), Yd(.16)), (XL(.05, .15), Yd(.15)), (XL(.06, .2), Yd(.2)), (XL(-.06, .24), Yd(.24))],
              dark(coat, 1.15))
    neck_x, neck_y = XL(-.01, .165) / S_, Yd(.165) / S_                   # base del cuello (en píxeles finales)
    out = _finish(img, int(Wc / S_), int(img.height / S_)).copy()
    stamp_head(out, kind, neck_x, neck_y)
    # recortar abajo para que la suela toque el piso (la imagen termina en la suela)
    a = np.asarray(out)[..., 3] > 0
    bottom = int(np.nonzero(a.any(1))[0].max()) + 1
    out = out.crop((0, 0, out.width, bottom))
    if facing_right:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    return out


def protagonist(frames, idle, ox, cyc, delay=0.0):
    walk = "".join(f'<g opacity="{1 if i == 0 else 0}" style="animation:wk{i} {cyc:.3f}s {delay:.3f}s infinite steps(1)">'
                   f'{image(f, ox, FEET - f.height + 1, up=1)}</g>' for i, f in enumerate(frames))
    return (f'<g style="{WALK_VIS}">{walk}</g>'
            f'<g opacity="0" style="{FRZ_VIS}">{image(idle, ox, FEET - idle.height + 1, up=1)}</g>')


# Cada uno con su andar (los dos a la velocidad de la cámara, sin patinar): Claude un poco más alto, con
# zancadas largas y lentas; Lucas con pasos más cortos y frecuentes. Así no caminan sincronizados.
PH_C = round(PH * 1.05)
A_c, cyc_c = gait_params(PH_C, NEAR_SPEED, cadence=1.3, amax=.5)
A_p, cyc_p = gait_params(PH, NEAR_SPEED, cadence=1.0, amax=.5)
cl = walk_frames(lambda ph, a: hero("morpheus", PH_C, ph, a), A_c)
cl_idle = hero("morpheus", PH_C, 0.0, .04, facing_right=True)
lu = walk_frames(lambda ph, a: hero("neo", PH, ph, a), A_p)
lu_idle = hero("neo", PH, 0.0, .04)
CLX, LUX = 98, 146
# ============================================================ congelado: fundido a negro y lluvia de código
# Todo se apaga a negro (rápido, empezando un instante antes del congelado), cae la lluvia de código de
# Matrix y sólo quedan Claude y Lucas. Al terminar, la lluvia se apaga y la calle vuelve.
css.append(f"@keyframes blackout{{0%,{pct(F0 - .6):.3f}%{{opacity:0}}{pct(F0):.3f}%,{pct(F1):.3f}%{{opacity:1}}"
           f"{pct(F1 + .7):.3f}%,100%{{opacity:0}}}}")
css.append(f"@keyframes rainvis{{0%,{pct(F0 - .25):.3f}%{{opacity:0}}{pct(F0 + .5):.3f}%,{pct(F1 - .5):.3f}%{{opacity:1}}"
           f"{pct(F1 + .1):.3f}%,100%{{opacity:0}}}}")
body.append(rect(-60, -80, W + 120, H + 160, "#000", f' opacity="0" style="animation:blackout {T}s infinite linear"'))
rrng = random.Random(1999)
RAIN_GLYPHS = [c for c in MINI if c not in " .,'"]
STEP_Y = 6
rain2 = []
for cx_ in range(-12, W + 12, 5):
    n = rrng.randint(9, 18)
    dur = rrng.uniform(1.6, 3.6)
    dl = -rrng.uniform(0, dur)
    k = nid("mr")
    span_ = (n * STEP_Y) + H + 80
    css.append(f"@keyframes {k}{{from{{transform:translateY({-n * STEP_Y - 40}px)}}to{{transform:translateY({H + 40}px)}}}}")
    glyphs = []
    for j in range(n):                                   # j = 0 arriba (cola) ... n-1 abajo (punta)
        g = mini_px(rrng.choice(RAIN_GLYPHS), cx_, j * STEP_Y)
        if rrng.random() < .5:                           # algunos invertidos: se parecen más a katakana
            g = {(2 * cx_ + 2 - x, y) for x, y in g}
        t_ = j / max(1, n - 1)
        if j == n - 1:
            col, op = "#e6ffe9", 1.0
        elif j == n - 2:
            col, op = "#8dffa6", .95
        else:
            col, op = "#22c55a", .12 + .75 * t_ ** 1.4
        glyphs.append(f'<g fill="{col}" opacity="{op:.2f}">{runs(g)}</g>')
    rain2.append(f'<g style="animation:{k} {dur:.2f}s {dl:.2f}s infinite linear">{"".join(glyphs)}</g>')
body.append(f'<g opacity="0" style="animation:rainvis {T}s infinite linear">{"".join(rain2)}</g>')

body.append(f'<ellipse cx="{CLX + cl[0].width / 2}" cy="{FEET}" rx="8" ry="1.6" fill="#000" opacity=".22"/>')
body.append(protagonist(cl, cl_idle, CLX, cyc_c))

# ------------------------------------------------ Lucas se queda de pie, distraído con la mujer de rojo
# Camina junto a Claude; al verla venir de frente se frena (queda quieto en el mundo, así que la cámara,
# que sigue a Claude, lo deja atrás); cuando ella pasa se da vuelta y la sigue con la mirada hasta que se
# vuelve agente y todo se congela. Al descongelarse apura el paso y recupera su lugar junto a Claude.
T_STOP = 17.2                                            # se frena
T_TURN = APPEAR + (LUX + 10 + NEAR_SPEED * 0) / 1 * 0    # (se recalcula abajo)
T_BACK = 31.6                                            # vuelve a su lugar
# instante en que ella lo pasa: su x = WX0 + WV (t - APPEAR); la de él = LUX + NEAR_SPEED (t - T_STOP)
T_TURN = (LUX - WX0 + WV * APPEAR - NEAR_SPEED * T_STOP) / (WV - NEAR_SPEED)
LAG = NEAR_SPEED * (F0 - T_STOP)                         # cuánto queda atrás al congelarse
V_CATCH = NEAR_SPEED + LAG / (T_BACK - F1)               # paso apurado para alcanzar a Claude
A_f, cyc_f = gait_params(PH, V_CATCH, cadence=.95, amax=.5)
lu_fast = walk_frames(lambda ph, a: hero("neo", PH, ph, a), A_f)
lu_stand_l = hero("neo", PH, 0.0, .04)                   # parado mirándola venir
lu_stand_r = hero("neo", PH, 0.0, .04, facing_right=True)   # se da vuelta y la sigue con la mirada
css.append(f"@keyframes lucx{{0%,{pct(T_STOP):.3f}%{{transform:translateX(0)}}{pct(F0):.3f}%,{pct(F1):.3f}%"
           f"{{transform:translateX({LAG:.2f}px)}}{pct(T_BACK):.3f}%,100%{{transform:translateX(0)}}}}")
css.append(f"@keyframes lwalk{{0%,{pct(T_STOP) - .001:.3f}%{{opacity:1}}{pct(T_STOP):.3f}%,{pct(T_BACK) - .001:.3f}%{{opacity:0}}"
           f"{pct(T_BACK):.3f}%,100%{{opacity:1}}}}")
lstand_l = window(T_STOP, T_TURN)
lstand_r = window(T_TURN, F1)
lfast = window(F1, T_BACK)


def frames_svg(frames, cyc, delay=0.0):
    return "".join(f'<g opacity="{1 if i == 0 else 0}" style="animation:wk{i} {cyc:.3f}s {delay:.3f}s infinite steps(1)">'
                   f'{image(f, LUX, FEET - f.height + 1, up=1)}</g>' for i, f in enumerate(frames))


lucas_svg = (f'<ellipse cx="{LUX + lu[0].width / 2}" cy="{FEET}" rx="8" ry="1.6" fill="#000" opacity=".22"/>'
             f'<g style="animation:lwalk {T}s infinite linear">{frames_svg(lu, cyc_p, -cyc_p * .4)}</g>'
             f'<g opacity="0" style="{lstand_l}">{image(lu_stand_l, LUX, FEET - lu_stand_l.height + 1, up=1)}</g>'
             f'<g opacity="0" style="{lstand_r}">{image(lu_stand_r, LUX, FEET - lu_stand_r.height + 1, up=1)}</g>'
             f'<g opacity="0" style="{lfast}">{frames_svg(lu_fast, cyc_f)}</g>')
body.append(f'<g style="animation:lucx {T}s infinite linear">{lucas_svg}</g>')

# ============================================================ cámara más cerca
scene_svg = "".join(body)
body = [f'<g transform="translate({W / 2} {HORIZON}) scale({ZOOM}) translate({-W / 2} {-HORIZON})">{scene_svg}</g>']

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
