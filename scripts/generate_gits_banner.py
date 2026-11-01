#!/usr/bin/env python3
"""Genera assets/gits-banner.svg: retrato pixel art + ciudad lluviosa con
helicópteros policiales, inspirado en la estética de Ghost in the Shell (1995).

Uso:  pip install pillow && python3 scripts/generate_gits_banner.py
La foto se toma de assets/avatar.png (tu avatar de GitHub).
"""
import random
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H, S = 200, 80, 4
random.seed(1995)

# ---------------------------------------------------------------- utilidades
css, body = [], []

def runs(pixels):
    rows = {}
    for x, y in pixels:
        rows.setdefault(y, []).append(x)
    out = []
    for y, xs in sorted(rows.items()):
        xs.sort(); start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x; continue
            out.append(f'<rect x="{start}" y="{y}" width="{prev-start+1}" height="1"/>')
            if x is not None: start = prev = x
    return "".join(out)

def g(pixels, color, attrs=""):
    return f'<g fill="{color}"{attrs}>{runs(pixels)}</g>' if pixels else ""

def sprite(rows, pal, ox=0, oy=0):
    return "".join(g({(ox+x, oy+y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == k}, col)
                   for k, col in pal.items())

def rect(x, y, w, h, fill, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{extra}/>'

# ---------------------------------------------------------------- fuentes
BIG = {  # 5x7
 "A":["01110","10001","10001","11111","10001","10001","10001"],
 "B":["11110","10001","10001","11110","10001","10001","11110"],
 "C":["01110","10001","10000","10000","10000","10001","01110"],
 "E":["11111","10000","10000","11110","10000","10000","11111"],
 "F":["11111","10000","10000","11110","10000","10000","10000"],
 "H":["10001","10001","10001","11111","10001","10001","10001"],
 "I":["01110","00100","00100","00100","00100","00100","01110"],
 "L":["10000","10000","10000","10000","10000","10000","11111"],
 "M":["10001","11011","10101","10101","10001","10001","10001"],
 "R":["11110","10001","10001","11110","10100","10010","10001"],
 "S":["01111","10000","10000","01110","00001","00001","11110"],
 "U":["10001","10001","10001","10001","10001","10001","01110"],
 "D":["11110","10001","10001","10001","10001","10001","11110"],
 "G":["01110","10001","10000","10111","10001","10001","01111"],
 "N":["10001","11001","10101","10011","10001","10001","10001"],
 "O":["01110","10001","10001","10001","10001","10001","01110"],
 "T":["11111","00100","00100","00100","00100","00100","00100"],
 " ":["00000"]*7,
}
SMALL = {  # 3x5
 "A":["010","101","111","101","101"], "B":["110","101","110","101","110"],
 "C":["011","100","100","100","011"], "D":["110","101","101","101","110"],
 "E":["111","100","110","100","111"], "G":["011","100","101","101","011"],
 "I":["111","010","010","010","111"], "K":["101","101","110","101","101"],
 "L":["100","100","100","100","111"], "N":["110","101","101","101","101"],
 "O":["010","101","101","101","010"], "R":["110","101","110","101","101"],
 "S":["011","100","010","001","110"], "T":["111","010","010","010","010"],
 "V":["101","101","101","101","010"], "Y":["101","101","010","010","010"],
 "W":["101","101","111","111","101"], "H":["101","101","111","101","101"], "/":["001","001","010","100","100"],
 ">":["100","010","001","010","100"], ",":["000","000","000","010","100"],
 ".":["000","000","000","000","010"], ":":["000","010","000","010","000"],
 "1":["010","110","010","010","111"], "9":["111","101","111","001","111"],
 "5":["111","100","111","001","111"], " ":["000"]*5,
}

def text_px(txt, font, x0, y0, adv):
    return [{(x0+i*adv+x, y0+y) for y, r in enumerate(font[c]) for x, b in enumerate(r) if b == "1"}
            for i, c in enumerate(txt)]

# ---------------------------------------------------------------- cielo / niebla
for i, col in enumerate(["#07110f", "#0a1714", "#0d1d19", "#11241f", "#152c25", "#1a3429", "#1f3c2e", "#24442f"]):
    body.append(rect(0, i * 10, W, H, col))

# ---------------------------------------------------------------- ciudad (3 capas)
css.append("@keyframes bl{0%,49%{opacity:1}50%,100%{opacity:0}}"
           "@keyframes fl{0%,70%{opacity:1}71%,100%{opacity:.15}}"
           "@keyframes neon{0%,100%{opacity:1}3%{opacity:.3}6%{opacity:1}52%{opacity:1}53%{opacity:.4}54%{opacity:1}}")
LAYERS = [  # (color, alto min, alto max, prob ventana, colores ventana)
    ("#16302a", 30, 55, .25, ["#2f5d4f"]),
    ("#0f221d", 18, 42, .35, ["#4f8f6f", "#3c7a66"]),
    ("#08130f", 8, 28, .45, ["#9fe8c8", "#e8d48b", "#6fd6c6"]),
]
for li, (col, hmin, hmax, pw, wcols) in enumerate(LAYERS):
    x = -random.randint(0, 6)
    while x < W:
        bw, bh = random.randint(6, 15), random.randint(hmin, hmax)
        top = H - bh
        body.append(rect(x, top, bw, bh, col))
        if random.random() < .35:  # torre / antena
            body.append(rect(x + bw // 2, top - 5, 1, 5, col))
            if li == 2 or random.random() < .5:
                body.append(rect(x + bw // 2, top - 6, 1, 1, "#ff3b4e",
                                 f' style="animation:bl {random.uniform(1.2,2.4):.1f}s infinite steps(1)"'))
        if li == 2 and random.random() < .3:  # cartel de neón vertical
            nc = random.choice(["#ff4fa3", "#3df2ff", "#c6ff5a"])
            body.append(rect(x + 1, top + 2, 2, random.randint(5, 9), nc,
                             f' opacity=".85" style="animation:neon {random.uniform(4,8):.1f}s {random.uniform(0,3):.1f}s infinite"'))
        for wy in range(top + 2, H - 1, 2 if li == 2 else 3):
            for wx in range(x + 1, x + bw - 1, 2):
                if random.random() < pw * .5:
                    a = (f' style="animation:fl {random.uniform(3,10):.1f}s {random.uniform(0,6):.1f}s infinite steps(1)"'
                         if random.random() < .15 else "")
                    body.append(rect(wx, wy, 1, 1, random.choice(wcols), a))
        x += bw + random.randint(-1, 2)

# niebla baja
body.append('<defs><linearGradient id="fog" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#5fa88a" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#5fa88a" stop-opacity=".22"/></linearGradient>'
            '<linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#eaffe8" stop-opacity=".5"/>'
            '<stop offset=".7" stop-color="#eaffe8" stop-opacity="0"/></linearGradient></defs>')
body.append(rect(0, 50, W, 30, "url(#fog)"))

# ---------------------------------------------------------------- texto HUD (izquierda)
HUD = "#7dffc4"
lines = [
    (text_px("GHOST DETECTED", BIG, 8, 12, 6), "#e8fff4"),
    (text_px("> CYBERBRAIN SYNC", SMALL, 8, 23, 4), HUD),
    (text_px("> DIVING INTO THE NET", SMALL, 8, 30, 4), HUD),
]
CYCLE, idx = 14, 0
total = sum(len(l) for l, _ in lines)
for chars, color in lines:
    for px in chars:
        if px:
            on = 4 + idx * 50 / total
            css.append(f"@keyframes t{idx}{{0%,{on:.2f}%{{opacity:0}}{on+.01:.2f}%,94%{{opacity:1}}94.01%,100%{{opacity:0}}}}")
            st = f' style="animation:t{idx} {CYCLE}s infinite"'
            body.append(g({(x+1, y+1) for x, y in px}, "#020605", st) + g(px, color, st))
        idx += 1
body.append(rect(8 + 21 * 4 + 1, 30, 3, 5, HUD, ' style="animation:bl 1s infinite steps(1)"'))
# barra de sincronización: aparece tras escribir la línea 2 y se llena
sync_on = 4 + 31 * 50 / total
css.append(f"@keyframes syncv{{0%,{sync_on:.1f}%{{opacity:0}}{sync_on+.1:.1f}%,94%{{opacity:1}}94.1%,100%{{opacity:0}}}}"
           f"@keyframes syncf{{0%,{sync_on:.1f}%{{transform:scaleX(0)}}{sync_on+20:.1f}%,100%{{transform:scaleX(1)}}}}")
bx, by, bw = 8 + 17 * 4 + 2, 23, 30
body.append(f'<g style="animation:syncv {CYCLE}s infinite">'
            + g({(bx + i, by) for i in range(bw)} | {(bx + i, by + 4) for i in range(bw)}
                | {(bx, by + j) for j in range(5)} | {(bx + bw - 1, by + j) for j in range(5)}, HUD, ' opacity=".7"')
            + f'<rect x="{bx+2}" y="{by+2}" width="{bw-4}" height="1" fill="#e8fff4" '
              f'style="transform-origin:{bx+2}px {by}px;animation:syncf {CYCLE}s infinite steps(13)"/></g>')
# esquinas HUD
for x, y, sx, sy in [(4, 6, 1, 1), (110, 6, -1, 1), (4, 38, 1, -1), (110, 38, -1, -1)]:
    body.append(g({(x, y), (x + sx, y), (x + 2*sx, y), (x, y + sy), (x, y + 2*sy)}, HUD, ' opacity=".6"'))

# ---------------------------------------------------------------- retrato pixelado
PX, PY, PS = 128, 7, 64
img = Image.open(ROOT / "assets" / "avatar.png").convert("RGB")
img = img.crop((110, 20, 450, 360)).resize((PS, PS), Image.BOX)
q = img.quantize(40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
pal, data = q.getpalette(), list(q.get_flattened_data() if hasattr(q, "get_flattened_data") else q.getdata())
by_color = {}
for i, ci in enumerate(data):
    r, gg, b = pal[ci*3:ci*3+3]
    # leve tinte verdoso tipo GITS
    r, gg, b = int(r*.92), min(255, int(gg*1.02 + 6)), int(b*.98)
    by_color.setdefault(f"#{r:02x}{gg:02x}{b:02x}", set()).add((PX + i % PS, PY + i // PS))
portrait = "".join(g(px, c) for c, px in by_color.items())

css.append("@keyframes glitch{0%,90%,100%{transform:translate(0,0)}91%{transform:translate(2px,0)}"
           "92%{transform:translate(-1px,0)}93%{transform:translate(1px,0)}94%{transform:translate(0,0)}}"
           "@keyframes slice{0%,90%,94%,100%{opacity:0}91%,93%{opacity:1}}"
           "@keyframes scan{from{transform:translateY(0)}to{transform:translateY(64px)}}"
           "@keyframes glint{0%,70%{transform:translateX(-6px);opacity:0}72%{opacity:1}80%{transform:translateX(14px);opacity:0}100%{opacity:0}}")
body.append(rect(PX - 2, PY - 2, PS + 4, PS + 4, "#020605"))
body.append(f'<g style="animation:glitch 7s infinite steps(1)">{portrait}</g>')
# barras de glitch
for yy, ww, dx in [(PY + 18, 30, 3), (PY + 40, 44, -2), (PY + 52, 20, 4)]:
    body.append(rect(PX + random.randint(0, PS - ww), yy, ww, 1, "#3df2ff",
                     f' opacity=".6" style="animation:slice 7s infinite steps(1)"'))
# brillo en los anteojos (destello que cruza)
body.append(f'<clipPath id="lens"><rect x="{PX+22}" y="{PY+30}" width="18" height="6"/></clipPath>'
            f'<g clip-path="url(#lens)"><g style="animation:glint 5s 1s infinite linear">'
            + g({(PX+22+i, PY+35-i) for i in range(6)} | {(PX+23+i, PY+35-i) for i in range(5)}, "#e8fff4", ' opacity=".8"')
            + "</g></g>")
# scanlines + línea de barrido
body.append(f'<g fill="#000" opacity=".18">' + "".join(rect(PX, PY + y, PS, 1, "#000") for y in range(0, PS, 2)) + "</g>")
body.append(f'<clipPath id="pc"><rect x="{PX}" y="{PY}" width="{PS}" height="{PS}"/></clipPath>'
            f'<g clip-path="url(#pc)"><rect x="{PX}" y="{PY-3}" width="{PS}" height="2" fill="{HUD}" opacity=".18" '
            f'style="animation:scan 3s infinite linear"/></g>')
# marco HUD
fr = set()
for i in range(PS + 6):
    for xx, yy in [(PX - 3 + i, PY - 3), (PX - 3 + i, PY + PS + 2)]:
        if i < 8 or i > PS - 3: fr.add((xx, yy))
    for xx, yy in [(PX - 3, PY - 3 + i), (PX + PS + 2, PY - 3 + i)]:
        if i < 8 or i > PS - 3: fr.add((xx, yy))
body.append(g(fr, HUD))
rec = text_px("REC", SMALL, PX + PS - 14, PY + 2, 4)
body.append(f'<g style="animation:bl 1.2s infinite steps(1)">'
            + rect(PX + PS - 19, PY + 3, 3, 3, "#ff3b4e") + g(set().union(*rec), "#ff3b4e") + "</g>")

# ---------------------------------------------------------------- helicópteros policiales
HELI = [
    "......RRRRRRRRRRRRRRRRRRRRRR..",
    "..............mm..............",
    "...........hhhhhhhh...........",
    "T.........hbbbbbbbbbgg........",
    "TT.......bbbbbbbbbbbggg.......",
    "TTbbbbbbbbbbbbbbbbbbgggg......",
    "T....l....wwwwwwwwwwwwwww.....",
    "..........bbbbbbbbbbbbbbL.....",
    "............s.......s.........",
    "..........sssssssssssssss.....",
]
HPAL = {"b": "#3a504d", "h": "#7a9a92", "w": "#d8e6df", "g": "#3df2ff", "m": "#1a2624", "s": "#1a2624", "T": "#9fb8ae"}
HW = len(HELI[0])
css.append("@keyframes rA{0%,49%{opacity:1}50%,100%{opacity:0}}"
           "@keyframes rB{0%,49%{opacity:0}50%,100%{opacity:1}}"
           "@keyframes red{0%,49%{opacity:1}50%,100%{opacity:.1}}"
           "@keyframes blu{0%,49%{opacity:.1}50%,100%{opacity:1}}"
           "@keyframes sweep{0%,100%{transform:rotate(-14deg)}50%{transform:rotate(16deg)}}"
           "@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(2px)}}")

def helicopter():
    rotor = "#b9d2c8"
    s = [sprite(HELI[1:], HPAL, 0, 1)]
    s.append('<g style="animation:rA .12s infinite steps(1)">' + g({(x, 0) for x, c in enumerate(HELI[0]) if c == "R"}, rotor) + "</g>")
    s.append('<g style="animation:rB .12s infinite steps(1)">' + g({(x, 0) for x in range(11, 19)}, rotor) + "</g>")
    s.append(rect(5, 7, 1, 1, "#ff3b4e", ' style="animation:red .5s infinite steps(1)"'))
    s.append(rect(24, 8, 1, 1, "#3d7bff", ' style="animation:blu .5s infinite steps(1)"'))
    s.append(rect(13, 3, 2, 1, "#ff3b4e", ' style="animation:red .5s infinite steps(1)"'))
    s.append(rect(15, 3, 2, 1, "#3d7bff", ' style="animation:blu .5s infinite steps(1)"'))
    beam = ('<g style="transform-origin:23px 9px;animation:sweep 4s infinite ease-in-out" opacity=".45">'
            '<polygon points="22,9 24,9 38,60 8,60" fill="url(#beam)"/></g>')
    return beam + "".join(s)

def flight(y, scale, dur, delay, ltr, dim):
    span = W + 40 / scale
    fx = (f"translate({-30}px,{y}px)", f"translate({W + 10}px,{y}px)") if ltr else \
         (f"translate({W + 10}px,{y}px)", f"translate({-30}px,{y}px)")
    name = f"fly{len(css)}"
    css.append(f"@keyframes {name}{{from{{transform:{fx[0]}}}to{{transform:{fx[1]}}}}}")
    flip = "" if ltr else f"translate({HW*scale} 0) scale(-1 1) "
    return (f'<g style="animation:{name} {dur}s {delay}s infinite linear;transform:translate(-40px,{y}px)">'
            f'<g style="animation:bob 1.6s infinite ease-in-out">'
            f'<g transform="{flip}scale({scale})" opacity="{dim}">{helicopter()}</g></g></g>')

body.append(flight(30, .55, 24, 0, False, .5))     # lejano
body.append(flight(18, .8, 18, 8, False, .8))      # medio
body.append(flight(2, 1.1, 13, 2, True, 1))        # cercano

# ---------------------------------------------------------------- lluvia
css.append(f"@keyframes rain{{from{{transform:translate(0,0)}}to{{transform:translate(-12px,{H}px)}}}}")
for layer, (n, op, dur) in enumerate([(70, .22, .9), (45, .35, .6)]):
    drops = []
    for _ in range(n):
        x, y = random.randrange(0, W + 12), random.randrange(0, H)
        for dx, dy in [(0, 0), (12, -H)]:
            drops.append(f'<rect x="{x+dx}" y="{y+dy}" width="1" height="{2+layer}"/>')
    body.append(f'<g fill="#bfe8d6" opacity="{op}" style="animation:rain {dur}s infinite linear">{"".join(drops)}</g>')

# ---------------------------------------------------------------- salida
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W*S}" height="{H*S}" '
       f'shape-rendering="crispEdges"><style>{"".join(css)}</style>{"".join(body)}</svg>')
out = ROOT / "assets" / "gits-banner.svg"
out.write_text(svg)
print(f"{out} ({len(svg)//1024} KB)")
