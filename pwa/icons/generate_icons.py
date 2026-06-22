"""
Genera iconos PNG para el manifest de la PWA.
Requiere: pip install Pillow
Uso: python pwa/icons/generate_icons.py
"""

from PIL import Image, ImageDraw, ImageFont
import os

ICON_DIR = os.path.dirname(__file__)
SIZES = [192, 512]

BG_COLOR     = (26, 26, 46)    # #1a1a2e
ACCENT_COLOR = (0, 255, 136)   # #00ff88
TEXT_COLOR   = (255, 255, 255)


def make_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), BG_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    # Círculo decorativo
    margin  = size * 0.1
    circle_bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(circle_bbox, outline=ACCENT_COLOR, width=max(2, size // 40))

    # Grieta estilizada (línea en zigzag)
    cx, cy = size / 2, size / 2
    r = size * 0.28
    points = [
        (cx - r,        cy),
        (cx - r * 0.3,  cy - r * 0.5),
        (cx,            cy + r * 0.2),
        (cx + r * 0.3,  cy - r * 0.4),
        (cx + r,        cy),
    ]
    draw.line(points, fill=ACCENT_COLOR, width=max(2, size // 32))

    return img


def main():
    for size in SIZES:
        icon = make_icon(size)
        path = os.path.join(ICON_DIR, f"icon-{size}.png")
        icon.save(path, "PNG")
        print(f"Generado: {path}")


if __name__ == "__main__":
    main()
