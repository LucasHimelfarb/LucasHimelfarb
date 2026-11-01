#!/usr/bin/env python3
"""Genera assets/pixel-banner.svg: una escena pixel art animada para el README.

Uso:  python3 scripts/generate_banner.py
Edita TEXT_LINES, colores o la semilla y vuelve a ejecutar.
"""
import random
from pathlib import Path

W, H, S = 160, 56, 5          # ancho/alto en pixeles "art" y escala
CYCLE = 12                     # segundos del ciclo de escritura
TEXT_LINES = [("HOLA, SOY LUCAS!", "#ffd166", 7), ("BIENVENIDO A MI PERFIL", "#8ecae6", 17)]
random.seed(7)

FONT = {  # 5x7
 "A":["01110","10001","10001","11111","10001","10001","10001"],
 "B":["11110","10001","10001","11110","10001","10001","11110"],
 "C":["01110","10001","10000","10000","10000","10001","01110"],
 "D":["11110","10001","10001","10001","10001","10001","11110"],
 "E":["11111","10000","10000","11110","10000","10000","11111"],
 "F":["11111","10000","10000","11110","10000","10000","10000"],
 "H":["10001","10001","10001","11111","10001","10001","10001"],
 "I":["01110","00100","00100","00100","00100","00100","01110"],
 "L":["10000","10000","10000","10000","10000","10000","11111"],
 "M":["10001","11011","10101","10101","10001","10001","10001"],
 "N":["10001","11001","10101","10011","10001","10001","10001"],
 "O":["01110","10001","10001","10001","10001","10001","01110"],
 "P":["11110","10001","10001","11110","10000","10000","10000"],
 "R":["11110","10001","10001","11110","10100","10010","10001"],
 "S":["01111","10000","10000","01110","00001","00001","11110"],
 "U":["10001","10001","10001","10001","10001","10001","01110"],
 "V":["10001","10001","10001","10001","10001","01010","00100"],
 "Y":["10001","10001","01010","00100","00100","00100","00100"],
 "!":["00100","00100","00100","00100","00100","00000","00100"],
 ",":["00000","00000","00000","00000","00110","00100","01000"],
 " ":["00000"]*7,
}

css, body = [], []

def rects(pixels, color, cls=None, extra=""):
    """pixels: set de (x,y). Une tramos horizontales para reducir tamaño."""
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
    c = f' class="{cls}"' if cls else ""
    return f'<g fill="{color}"{c}{extra}>{"".join(out)}</g>'

# --- cielo en bandas ---
for i, col in enumerate(["#0b132b", "#101a36", "#162042", "#1c2650", "#232d5c"]):
    body.append(f'<rect x="0" y="{i*8}" width="{W}" height="{H}" fill="{col}"/>')

# --- estrellas titilantes ---
css.append("@keyframes tw{0%,100%{opacity:1}50%{opacity:.15}}")
for _ in range(45):
    x, y = random.randrange(W), random.randrange(34)
    if 6 <= x <= 142 and 5 <= y <= 25:  # no tapar el texto
        continue
    d, dl = random.uniform(1.5, 4), random.uniform(0, 4)
    c = random.choice(["#ffffff", "#cdd6f4", "#ffd166"])
    body.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{c}" '
                f'style="animation:tw {d:.1f}s {dl:.1f}s infinite steps(2)"/>')

# --- luna ---
moon = ["..XXXX..", ".XXXXXX.", "XXXXoXXX", "XXXXXXXX", "XXoXXXXX", "XXXXXXoX", ".XXXXXX.", "..XXXX.."]
mx, my = 146, 4
body.append(rects({(mx+x, my+y) for y, r in enumerate(moon) for x, ch in enumerate(r) if ch == "X"}, "#f1faee"))
body.append(rects({(mx+x, my+y) for y, r in enumerate(moon) for x, ch in enumerate(r) if ch == "o"}, "#c9d1d9"))

# --- estrella fugaz ---
css.append("@keyframes shoot{0%,80%{transform:translate(0,0);opacity:0}"
           "82%{opacity:1}95%{transform:translate(-60px,24px);opacity:0}100%{opacity:0}}")
body.append('<g style="animation:shoot 9s 2s infinite linear">'
            + rects({(120, 2)}, "#ffffff") + rects({(121, 1), (122, 1)}, "#8ecae6")
            + rects({(123, 0)}, "#3a86ff") + "</g>")

# --- texto que se escribe letra por letra ---
idx, total = 0, sum(len(t) for t, _, _ in TEXT_LINES)
step = 55 / total  # % del ciclo que tarda en aparecer cada letra (todo el texto en ~55%)
for text, color, ty in TEXT_LINES:
    tx = (W - 12 - len(text) * 6) // 2 + 2
    for i, ch in enumerate(text):
        g = FONT[ch]
        px = {(tx + i*6 + x, ty + y) for y, r in enumerate(g) for x, b in enumerate(r) if b == "1"}
        if not px: idx += 1; continue
        on = 5 + idx * step
        css.append(f"@keyframes t{idx}{{0%,{on:.2f}%{{opacity:0}}{on+0.01:.2f}%,92%{{opacity:1}}92.01%,100%{{opacity:0}}}}")
        body.append(rects({(x+1, y+1) for x, y in px}, "#05070f", extra=f' style="animation:t{idx} {CYCLE}s infinite"'))
        body.append(rects(px, color, extra=f' style="animation:t{idx} {CYCLE}s infinite"'))
        idx += 1
# cursor parpadeante tras la última línea
last, _, ly = TEXT_LINES[-1]
cx = (W - 12 - len(last) * 6) // 2 + 2 + len(last) * 6
css.append("@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:0}}")
body.append(f'<rect x="{cx}" y="{ly}" width="4" height="7" fill="#8ecae6" style="animation:blink 1s infinite steps(1)"/>')

# --- skyline ---
GROUND = 50
css.append("@keyframes win{0%,60%{opacity:1}61%,100%{opacity:.1}}")
x = 0
while x < W:
    bw, bh = random.randint(8, 16), random.randint(10, 22)
    col = random.choice(["#14213d", "#1a2547", "#111a33"])
    body.append(f'<rect x="{x}" y="{GROUND-bh}" width="{bw}" height="{bh}" fill="{col}"/>')
    if random.random() < .3:  # antena
        body.append(f'<rect x="{x+bw//2}" y="{GROUND-bh-3}" width="1" height="3" fill="{col}"/>')
        body.append(f'<rect x="{x+bw//2}" y="{GROUND-bh-4}" width="1" height="1" fill="#ef476f" style="animation:blink 1.6s infinite steps(1)"/>')
    for wy in range(GROUND - bh + 2, GROUND - 2, 3):
        for wx in range(x + 2, x + bw - 2, 3):
            if random.random() < .45:
                anim = (f' style="animation:win {random.uniform(3,9):.1f}s {random.uniform(0,5):.1f}s infinite steps(1)"'
                        if random.random() < .25 else "")
                body.append(f'<rect x="{wx}" y="{wy}" width="1" height="1" fill="#ffd166"{anim}/>')
    x += bw + random.randint(0, 2)

# --- suelo ---
body.append(f'<rect x="0" y="{GROUND}" width="{W}" height="{H-GROUND}" fill="#2b2d42"/>')
body.append(f'<rect x="0" y="{GROUND}" width="{W}" height="1" fill="#4a4e69"/>')
for gx in range(2, W, 7):
    body.append(f'<rect x="{gx}" y="{GROUND+3}" width="3" height="1" fill="#3d405b"/>')

# --- personaje caminando ---
PAL = {"h": "#3d2c2e", "s": "#f4c095", "e": "#1b1b1b", "b": "#ef476f", "p": "#3a86ff", "f": "#f1faee", "l": "#9aa0a6"}
TOP = ["..hhhh..", ".hhhhhh.", ".hhsses.", "..sssss.", "..bbbb.."]
FRAMES = [TOP + [".sbbbbs.", "..bbbb..", "..pppp..", ".pp..pp.", ".ff..ff."],
          TOP + ["..bbbbs.", "..bbbb..", "..pppp..", "...pp...", "...ff..."]]
css.append("@keyframes walk{from{transform:translateX(-12px)}to{transform:translateX(172px)}}"
           "@keyframes fA{0%,49%{opacity:1}50%,100%{opacity:0}}"
           "@keyframes fB{0%,49%{opacity:0}50%,100%{opacity:1}}")
char = []
for n, fr in enumerate(FRAMES):
    parts = "".join(rects({(x, GROUND - 10 + y) for y, r in enumerate(fr) for x, ch in enumerate(r) if ch == k}, c)
                    for k, c in PAL.items())
    char.append(f'<g style="animation:{"fA" if n == 0 else "fB"} .5s infinite steps(1)">{parts}</g>')
body.append(f'<g style="animation:walk 14s infinite linear">{"".join(char)}</g>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W*S}" height="{H*S}" '
       f'shape-rendering="crispEdges"><style>{"".join(css)}</style>{"".join(body)}</svg>')
out = Path(__file__).resolve().parent.parent / "assets" / "pixel-banner.svg"
out.write_text(svg)
print(f"{out} ({len(svg)//1024} KB)")
