"""The image generator for stress tests: a class-conditional DCGAN in the layout of P5's (P5).

Trained on P5's SULAND crops of two surrogate-mine types, it makes images of types the photo screener never
learned, to test whether the screener answers "unknown" rather than a confident wrong category. Generated images
never train the screener and never enter a decision: P5 found that a classifier trained on generated mines found
almost none of the real ones.
"""
import io
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TYPES = ["PFM-1", "PMA-2"]
SETTINGS = {"nz": 100, "width": 32, "epochs": 40, "batch": 64, "learning_rate": 0.0002, "beta1": 0.5,
            "threads": 2, "jpeg_quality": 95}  # P5's settings


class Generator(nn.Module):
    """Random numbers and a one-hot type in, a 64 x 64 RGB image in [-1, 1] out."""

    def __init__(self, nz=SETTINGS["nz"], width=SETTINGS["width"], types=len(TYPES)):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + types, w * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(w * 8), nn.ReLU(True),
            nn.ConvTranspose2d(w * 8, w * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(w * 4), nn.ReLU(True),
            nn.ConvTranspose2d(w * 4, w * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(w * 2), nn.ReLU(True),
            nn.ConvTranspose2d(w * 2, w, 4, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z, onehot):
        return self.net(torch.cat([z, onehot], dim=1)[:, :, None, None])


class Discriminator(nn.Module):
    """A 64 x 64 RGB image and its one-hot type in, one real-or-generated logit out."""

    def __init__(self, width=SETTINGS["width"], types=len(TYPES)):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3 + types, w, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w, w * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(w * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w * 2, w * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(w * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w * 4, w * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(w * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w * 8, 1, 4, 1, 0, bias=False))

    def forward(self, image, onehot):
        maps = onehot[:, :, None, None].expand(-1, -1, image.shape[2], image.shape[3])
        return self.net(torch.cat([image, maps], dim=1)).view(-1)


def init_weights(module):
    """DCGAN initialisation: convolution weights from N(0, 0.02); batch-norm scales from N(1, 0.02), shifts at 0."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def to_unit_range(images):
    """8-bit RGB images (n, 64, 64, 3) to a float tensor (n, 3, 64, 64) in [-1, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    return tensor.contiguous(memory_format=torch.channels_last)


def to_8bit(tensor):
    return ((tensor.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def train(images, types, settings=None, seed=0, verbose=True):
    """Train the GAN on 8-bit crops with types 0 and 1; returns the generator (in evaluation mode) and the history."""
    s = {**SETTINGS, **(settings or {})}
    torch.set_num_threads(s["threads"])        # fixed, so results do not depend on the computer's core count
    torch.use_deterministic_algorithms(True)
    x, y = to_unit_range(images), torch.from_numpy(np.asarray(types))
    torch.manual_seed(seed)
    generator = Generator().to(memory_format=torch.channels_last)
    discriminator = Discriminator().to(memory_format=torch.channels_last)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    opt_g = torch.optim.Adam(generator.parameters(), lr=s["learning_rate"], betas=(s["beta1"], 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=s["learning_rate"], betas=(s["beta1"], 0.999))
    bce = nn.BCEWithLogitsLoss()
    order_gen = torch.Generator().manual_seed(seed)
    flip_gen = torch.Generator().manual_seed(seed + 1)
    noise_gen = torch.Generator().manual_seed(seed + 2)
    history, started, batch = [], time.perf_counter(), s["batch"]
    for epoch in range(1, s["epochs"] + 1):
        generator.train()
        discriminator.train()
        order = torch.randperm(len(x), generator=order_gen)
        totals, steps = {"D loss": 0.0, "G loss": 0.0, "D(real)": 0.0, "D(generated)": 0.0}, 0
        for start in range(0, len(x) - batch + 1, batch):
            index = order[start:start + batch]
            flip = torch.rand(batch, generator=flip_gen) < 0.5
            real = torch.where(flip[:, None, None, None], x[index].flip(3), x[index])
            onehot = F.one_hot(y[index], len(TYPES)).float()
            fake = generator(torch.randn(batch, s["nz"], generator=noise_gen), onehot)
            opt_d.zero_grad()
            real_logit, fake_logit = discriminator(real, onehot), discriminator(fake.detach(), onehot)
            d_loss = bce(real_logit, torch.ones_like(real_logit)) + bce(fake_logit, torch.zeros_like(fake_logit))
            d_loss.backward()
            opt_d.step()
            opt_g.zero_grad()
            logit = discriminator(fake, onehot)
            g_loss = bce(logit, torch.ones_like(logit))
            g_loss.backward()
            opt_g.step()
            totals["D loss"] += d_loss.item()
            totals["G loss"] += g_loss.item()
            totals["D(real)"] += torch.sigmoid(real_logit).mean().item()
            totals["D(generated)"] += torch.sigmoid(fake_logit).mean().item()
            steps += 1
        history.append({"epoch": epoch, **{k: v / steps for k, v in totals.items()},
                        "seconds": time.perf_counter() - started})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            h = history[-1]
            print(f"  epoch {epoch:>2}: D loss {h['D loss']:.3f}, G loss {h['G loss']:.3f}, {h['seconds']:.0f} s",
                  flush=True)
    generator.eval()
    return generator, pd.DataFrame(history).set_index("epoch")


def jpeg_round_trip(images, quality=SETTINGS["jpeg_quality"]):
    """Store 8-bit images as JPEGs and read them back, as the real crops were stored."""
    out = []
    for image in images:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        out.append(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")))
    return np.stack(out)


def generate(generator, counts, seed):
    """counts[i] images of type i, from seeded random inputs, stored as the real crops are; returns (images, types)."""
    types = np.repeat(np.arange(len(TYPES)), counts)
    z = torch.randn(len(types), SETTINGS["nz"], generator=torch.Generator().manual_seed(seed))
    onehot = F.one_hot(torch.from_numpy(types), len(TYPES)).float()
    with torch.no_grad():
        images = np.concatenate([to_8bit(generator(z[i:i + 256], onehot[i:i + 256])) for i in range(0, len(types), 256)])
    return jpeg_round_trip(images), types


def save(path, generator, history):
    torch.save({"state_dict": generator.state_dict(), "history": history.reset_index().to_dict("list"),
                "types": TYPES, "settings": SETTINGS}, path)


def load(path):
    saved = torch.load(path, weights_only=False)
    generator = Generator().to(memory_format=torch.channels_last)
    generator.load_state_dict(saved["state_dict"])
    generator.eval()
    return generator, pd.DataFrame(saved["history"]).set_index("epoch")
