from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.io import ensure_dir, save_numpy, write_json


class ArtifactLogger:
    """Saves large intermediate tensors to the artifacts/ subfolder of a run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.artifact_dir = Path(run_dir) / "artifacts"
        ensure_dir(self.artifact_dir)

    def save_option_vectors(self, episode_id: str, turn: int, vectors: np.ndarray) -> str:
        fname = f"episode_{episode_id}_turn_{turn}_option_vectors.npy"
        path = self.artifact_dir / fname
        save_numpy(vectors, path)
        return str(path)

    def save_hypothesis_vectors(self, episode_id: str, turn: int, vectors: np.ndarray, ids: list[str]) -> str:
        fname = f"episode_{episode_id}_turn_{turn}_hyp_vectors.npy"
        save_numpy(vectors, self.artifact_dir / fname)
        write_json(ids, self.artifact_dir / f"episode_{episode_id}_turn_{turn}_hyp_ids.json")
        return str(self.artifact_dir / fname)

    def save_belief_snapshot(self, episode_id: str, turn: int, mu: np.ndarray, sigma: np.ndarray) -> str:
        fname = f"episode_{episode_id}_turn_{turn}_belief.npz"
        path = self.artifact_dir / fname
        np.savez(path, mu=mu, sigma=sigma)
        return str(path)
