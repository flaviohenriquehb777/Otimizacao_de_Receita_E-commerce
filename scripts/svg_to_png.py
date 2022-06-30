from pathlib import Path
import sys

try:
    import cairosvg
except Exception:
    print("[ERROR] cairosvg não instalado")
    sys.exit(1)


def convert(svg_path: Path, png_path: Path, scale: float = 1.0):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)
    print('[OK]', svg_path, '->', png_path)


def main():
    targets = [
        (Path('docs/badges/pages.svg'), Path('docs/badges/pages.png'), 1.0),
        (Path('docs/badges/license-mit.svg'), Path('docs/badges/license-mit.png'), 1.0),
    ]
    for svg, png, scale in targets:
        if not svg.exists():
            print('[SKIP] Não encontrado:', svg)
            continue
        convert(svg, png, scale)


if __name__ == '__main__':
    main()