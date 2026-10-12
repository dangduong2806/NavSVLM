"""InternVL tiling from the supplied original data.py (policy is configurable)."""
from PIL import Image
import numpy as np
import torch


def image_tiles(image, *, image_size=448, max_tiles=6, use_thumbnail=True):
    image = image.convert("RGB")
    width, height = image.size
    ratios = set((i, j) for n in range(1, max_tiles + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_tiles)
    ratios = sorted(ratios, key=lambda pair: pair[0] * pair[1])
    best, best_diff = (1, 1), float("inf")
    for ratio in ratios:
        difference = abs(width / height - ratio[0] / ratio[1])
        if difference < best_diff:
            best_diff, best = difference, ratio
        elif difference == best_diff and width * height > 0.5 * image_size ** 2 * ratio[0] * ratio[1]:
            best = ratio
    resized = image.resize((image_size * best[0], image_size * best[1]))
    tiles = []
    for i in range(best[0] * best[1]):
        x, y = (i % best[0]) * image_size, (i // best[0]) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    values = []
    for tile in tiles:
        tile = tile.resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(tile, copy=True)).permute(2, 0, 1).float() / 255.0
        values.append((tensor - mean) / std)
    return torch.stack(values)
