#!/usr/bin/env python3
"""
Generate a QR code for the deployed Streamlit app.
Usage: python generate_qr.py "https://your-app-name.streamlit.app"
"""

import sys
from pathlib import Path

def generate_qr_code(url: str, output_path: str = "assets/qr_code.png"):
    try:
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
    except ImportError:
        print("Installing qrcode library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qrcode[pil]"])
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

    # Create output directory
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Generate QR code with styling
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create styled image
    try:
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=RoundedModuleDrawer(),
            fill_color="#2c3e50",
            back_color="white"
        )
    except Exception:
        # Fallback to basic QR
        img = qr.make_image(fill_color="#2c3e50", back_color="white")

    img.save(output_path)
    print(f"QR code saved to: {output_path}")
    print(f"URL encoded: {url}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_qr.py <YOUR_STREAMLIT_APP_URL>")
        print("Example: python generate_qr.py https://financial-crime-detection.streamlit.app")
        sys.exit(1)

    url = sys.argv[1]
    generate_qr_code(url)
