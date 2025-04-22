import os.path
from pathlib import Path
from typing import List

import numpy as np
import torch
import plotly.graph_objects as go
import imageio as iio
from matplotlib import pyplot as plt


class AttentionRecorder:
    """Collect one attention *row* per decoder step and create an interactive
    Plotly visualisation **including an averaged “ALL” layer**.

    A single call of :py:meth:`add` receives the mean–over–heads cross–attention
    matrix of shape *(T_dec,T_enc)*.  We store only the **last decoder row**
    (i.e. the token that has just been generated).  That keeps the HTML file
    small while still showing which encoder positions each new token attends
    to.
    """

    def __init__(self, n_layers: int, out_dir: str | Path = "attn_videos", fps: int = 10):
        self.frames: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

    # ---------------------------------------------------------------------
    # Data collection ------------------------------------------------------
    # ---------------------------------------------------------------------
    def add(self, layer: int, attn: torch.Tensor):
        """Record the last decoder row of *attn* (shape (T_dec,T_enc))."""
        a = attn.float()
        a = (a - a.min()) / (a.max() - a.min() + 1e-5)
        last_row = a[-1:].cpu().numpy()  # shape (1,T_enc)
        self.frames[layer].append(last_row)

    # ---------------------------------------------------------------------
    # HTML visualisation ---------------------------------------------------
    # ---------------------------------------------------------------------
    def _build_tensor(self) -> tuple[np.ndarray, int, int, int]:
        """Stack the recorded rows into a 4‑D tensor.

        Returns
        -------
        tensor_all : ndarray  (n_layers+1, n_steps,  max_dec,  n_enc)
            Complete stack **including** the averaged "ALL" layer at index
            *n_layers*.
        n_steps     : int     – number of decoder steps (time axis)
        max_dec     : int     – equals *n_steps* (one row per token)
        n_enc       : int     – encoder length (usually1500)
        """
        if not any(self.frames):
            raise RuntimeError("[AttentionRecorder] no data collected – call add() first")

        n_layers = len(self.frames)
        n_steps = max(len(fr) for fr in self.frames)
        max_dec = n_steps  # one row per token ⇒ square
        n_enc = max(mat.shape[1] for fr in self.frames for mat in fr)

        tensor = np.zeros((n_layers, n_steps, max_dec, n_enc), dtype=np.float32)

        for l, layer_frames in enumerate(self.frames):
            for t in range(n_steps):
                rows = [layer_frames[i] for i in range(min(t + 1, len(layer_frames)))]
                mat = np.vstack(rows)  # (t+1, n_enc)
                pad = max_dec - mat.shape[0]
                if pad:
                    mat = np.vstack([mat, np.zeros((pad, n_enc), dtype=mat.dtype)])
                tensor[l, t] = mat

        # averaged layer "ALL" ------------------------------------------------
        avg_tensor = tensor.mean(axis=0, keepdims=True)  # (1, …)
        tensor_all = np.concatenate([tensor, avg_tensor], axis=0)  # (L+1, …)
        return tensor_all, n_steps, max_dec, n_enc

    def save_html(
            self,
            tokens_text: list[str],
            filename: str = "attention_interactive.html",
            row_px: int = 22,
    ):
        """Schreibt ein interaktives HTML ohne Zeit‑Slider (alles t‑auf einmal)."""
        tensor_all, n_steps, max_dec, n_enc = self._build_tensor()
        n_layers_all = tensor_all.shape[0]

        # y‑Beschriftungen auf volle Decoder‑Länge auffüllen
        tokens_pad = tokens_text + [tokens_text[-1]] * max(0, max_dec - len(tokens_text))

        # ------------------------------------------------------------------
        # Initiale Figur – wir zeigen gleich t = n_steps‑1
        # ------------------------------------------------------------------
        t_last = n_steps - 1
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=tensor_all[0, t_last],  # vollständige Matrix
            colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="Attention"),
        ))

        fig.update_layout(
            height=140 + row_px * max_dec,
            xaxis=dict(title=f"Encoder‑Token(0…{n_enc - 1})"),
            yaxis=dict(title="Decoder‑Token",
                       tickmode="array",
                       tickvals=list(range(max_dec)),
                       ticktext=tokens_pad,
                       autorange="reversed"),
            title=f"Cross‑Attention • Layer0 • allet(0…{t_last})",
            autosize=True,
        )

        # ------------------------------------------------------------------
        # Dropdown zum Umschalten der Layer (inkl. „ALL“)
        # ------------------------------------------------------------------
        buttons = []
        for l in range(n_layers_all):
            label = "ALL" if l == n_layers_all - 1 else f"Layer{l}"
            buttons.append({
                "label": label,
                "method": "update",
                "args": [
                    {"z": [tensor_all[l, t_last]]},  # vollständige Matrix
                    {"title": f"Cross‑Attention • {label} • allet(0…{t_last})"},
                ],
            })

        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.1,
                "showactive": True,
            }]
        )

        # ------------------------------------------------------------------
        # Export
        # ------------------------------------------------------------------
        out_path = self.out_dir / filename
        fig.write_html(out_path, include_plotlyjs="cdn", config={"responsive": True})
        print(f"[AttentionRecorder] wrote {out_path}")
        self.frames = [[] for _ in range(len(self.frames))]  # reset for next run

    # ---------------------------------------------------------------------
    # (optional) MP4 writer – unchanged from the original version ----------
    # ---------------------------------------------------------------------
    def save(self):
        """Write legacy per‑layer MP4s (kept for compatibility)."""
        for layer_idx, layer_frames in enumerate(self.frames):
            if not layer_frames:
                continue

            max_w = max(m.shape[1] for m in layer_frames)
            vid_path = self.out_dir / f"cross_layer_{layer_idx:02d}.mp4"
            with iio.get_writer(vid_path, fps=self.fps, codec="libx264", macro_block_size=None) as writer:
                for row in layer_frames:
                    mat = np.vstack([row, row])  # duplicate row so height ≥2
                    if mat.shape[1] < max_w:
                        pad = np.tile(mat[:, -1:], (1, max_w - mat.shape[1]))
                        mat = np.hstack([mat, pad])
                    img = (plt.cm.viridis(mat)[:, :, :3] * 255).astype(np.uint8)
                    writer.append_data(img)
            print(f"[AttentionRecorder] wrote {vid_path}")


# -------------------------------------------------------------------------
# global singleton ---------------------------------------------------------
# -------------------------------------------------------------------------
RECORDER: AttentionRecorder | None = None


def get_recorder() -> AttentionRecorder | None:  # noqa: N802 – keep legacy name
    return RECORDER
