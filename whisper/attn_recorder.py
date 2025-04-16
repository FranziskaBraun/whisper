import os.path
from pathlib import Path
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
import imageio as iio


class AttentionRecorder:
    """
    Collects one heat‑map frame per decoder step and writes a video per layer.
    """

    def __init__(self,
                 n_layers: int,
                 out_dir: str | Path = "attn_videos",
                 fps: int = 10):
        self.frames: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

    def add(self, layer: int, attn: torch.Tensor) -> None:
        """
        Store a single (T_dec, T_enc) attention matrix for the given layer.
        Expected shape after mean‑over‑heads: (T_dec, T_enc)
        """
        # normalize to [0,1] and bring to CPU / numpy once
        a = attn.float()
        a = (a - a.min()) / (a.max() - a.min() + 1e-5)
        self.frames[layer].append(a.cpu().numpy())

    def save(self):
        for layer_idx, layer_frames in enumerate(self.frames):
            if not layer_frames:
                continue

            # --- NEU: Zielgröße berechnen --------------------------------
            max_h = max(m.shape[0] for m in layer_frames)  # größte T_dec
            max_w = max(m.shape[1] for m in layer_frames)  # i. d. R. 1500

            # make sure dimensions are even – libx264 requires width & height % 2 == 0
            if max_h % 2:
                max_h += 1
            if max_w % 2:
                max_w += 1
            # --------------------------------------------------------------

            vid_path = self.out_dir / f"cross_layer_{layer_idx:02d}.mp4"
            with iio.get_writer(
                    vid_path,
                    fps=self.fps,
                    codec="libx264",
                    macro_block_size=None,  # Warnung wegen 16er‑Raster unterdrücken
            ) as writer:
                for mat in layer_frames:
                    h, w = mat.shape

                    # --- NEU: nach unten mit letztem Wert auffüllen --------
                    # pad bottom rows if necessary
                    if h < max_h:
                        pad = np.tile(mat[-1:, :], (max_h - h, 1))
                        mat = np.vstack([mat, pad])
                    elif h == max_h and h % 2:  # odd height, add one row
                        mat = np.vstack([mat, mat[-1:, :]])

                    # pad right‑hand columns if necessary
                    if w < max_w:
                        pad = np.tile(mat[:, -1:], (1, max_w - w))
                        mat = np.hstack([mat, pad])
                    elif w == max_w and w % 2:  # odd width, add one column
                        mat = np.hstack([mat, mat[:, -1:]])
                    # -------------------------------------------------------

                    img = (plt.cm.viridis(mat)[:, :, :3] * 255).astype(np.uint8)
                    writer.append_data(img)
            print(f"[AttentionRecorder] wrote {vid_path}")


RECORDER: AttentionRecorder | None = None


def get_recorder() -> AttentionRecorder | None:
    return RECORDER
