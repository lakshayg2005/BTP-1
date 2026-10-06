"""Daily-life degradations: what happens to an image on WhatsApp, Instagram or a screenshot.

Training on these makes the detector robust to real-world content (motivated by DailyBench, arXiv 2607.24016).
"""
import io
import random

from PIL import Image, ImageFilter


def jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.BICUBIC)


def whatsapp_like(img: Image.Image) -> Image.Image:
    # WhatsApp caps the long side around 1600px and re-encodes at moderate JPEG quality
    w, h = img.size
    s = min(1.0, 1600 / max(w, h))
    return jpeg(resize(img, s), random.randint(60, 80))


def screenshot_like(img: Image.Image) -> Image.Image:
    # rescale to a screen size, slight blur from display scaling, then PNG-like (no JPEG)
    img = resize(img, random.uniform(0.5, 0.9))
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.6)))


def random_daily_life(img: Image.Image) -> Image.Image:
    ops = [
        lambda x: x,
        lambda x: jpeg(x, random.randint(50, 95)),
        lambda x: resize(x, random.uniform(0.4, 0.9)),
        whatsapp_like,
        screenshot_like,
        lambda x: jpeg(resize(x, random.uniform(0.5, 0.9)), random.randint(55, 85)),
    ]
    return random.choice(ops)(img.convert("RGB"))
