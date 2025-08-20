import base64
import io
import numpy as np
from PIL import Image

def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def resize_keep_aspect(img: Image.Image, max_side=512) -> Image.Image:
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR)

def overlay_mask(rgb: Image.Image, mask_bin: np.ndarray, color=(255, 40, 40), alpha=0.45) -> Image.Image:
    """
    rgb: PIL RGB image
    mask_bin: HxW boolean/0-1 numpy array (same size as rgb)
    """
    rgb = rgb.convert("RGBA")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask_bin.astype("uint8") * 255), mode="L").resize(rgb.size, Image.NEAREST)

    color_img = Image.new("RGBA", rgb.size, color + (0,))
    # put color only where mask==1, then set alpha
    color_img.putalpha(mask_img)
    # blend on top of original
    out = Image.alpha_composite(rgb, Image.blend(Image.new("RGBA", rgb.size, (0, 0, 0, 0)), color_img, alpha))
    return out.convert("RGB")
