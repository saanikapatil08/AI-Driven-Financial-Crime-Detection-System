#!/usr/bin/env python3
"""Generates a QR code for the deployed Streamlit app URL."""

import sys
from pathlib import Path


def generate_qr_code(url: str, output_path: str = "assets/qr_code.png"):
    try:
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qrcode[pil]"])
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)

    try:
        img = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer(),
                            fill_color="#2c3e50", back_color="white")
    except Exception:
        img = qr.make_image(fill_color="#2c3e50", back_color="white")

    img.save(output_path)
    print(f"QR code saved to: {output_path}")
    print(f"URL: {url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_qr.py <URL>")
        print("Example: python generate_qr.py https://your-app.streamlit.app")
        sys.exit(1)
    generate_qr_code(sys.argv[1])
