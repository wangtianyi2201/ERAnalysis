import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if pd.isna(value):
        return np.zeros(0, dtype=np.float32)
    if isinstance(value, str):
        try:
            parsed = eval(value, {"__builtins__": {}}, {})
        except Exception:
            parsed = []
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return np.asarray(parsed, dtype=np.float32)
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def normalize_event_df(event_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "coors.Type": "coors.type",
        "coors.properties.basesName": "coors.properties.basedName",
        "locsId": "locIds",
        "locs": "loc",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in event_df.columns}
    if rename_map:
        event_df = event_df.rename(columns=rename_map)

    required = ["coors.longitudes", "coors.latitudes"]
    for column in required:
        if column not in event_df.columns:
            raise ValueError(f"event_df missing required column '{column}'")

    return event_df


def build_gt_dict(
    gt_df: pd.DataFrame,
    event_df: pd.DataFrame,
    report_key: str = "report_id",
    track_key: str = "track_id",
) -> Dict[int, int]:
    event_df = event_df.reset_index(drop=True)
    gt_dict: Dict[int, int] = {}

    if report_key in gt_df.columns:
        for _, row in gt_df.iterrows():
            rid = row[report_key]
            if isinstance(rid, (int, np.integer)) and 0 <= rid < len(event_df):
                gt_dict[int(rid)] = int(row[track_key])

    if gt_dict:
        return gt_dict

    for key in ["entityId", "recordAt", "locIds", "loc"]:
        if key in gt_df.columns and key in event_df.columns:
            event_map = {str(v): idx for idx, v in enumerate(event_df[key].astype(str))}
            for _, row in gt_df.iterrows():
                value = str(row.get(key, ""))
                if value in event_map:
                    gt_dict[event_map[value]] = int(row[track_key])
            if gt_dict:
                return gt_dict

    return gt_dict


def build_trajectories(sensor_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    trajectories: Dict[int, np.ndarray] = {}

    if {"trackId", "latitude", "longitude", "heading", "speed", "timestamp"}.issubset(sensor_df.columns):
        sensor_df = sensor_df.sort_values(["trackId", "timestamp"])
        for track_id, group in sensor_df.groupby("trackId"):
            trajectories[int(track_id)] = group[
                ["latitude", "longitude", "heading", "speed", "timestamp"]
            ].to_numpy(dtype=np.float32)
        return trajectories

    if {"trajectory_id", "x", "y", "vx", "vy", "t"}.issubset(sensor_df.columns):
        sensor_df = sensor_df.sort_values(["trajectory_id", "t"])
        for track_id, group in sensor_df.groupby("trajectory_id"):
            vx = group["vx"].to_numpy(dtype=np.float32)
            vy = group["vy"].to_numpy(dtype=np.float32)
            heading = np.rad2deg(np.arctan2(vy, vx + 1e-6)).astype(np.float32)
            speed = np.sqrt(vx ** 2 + vy ** 2).astype(np.float32)
            trajectory = np.column_stack(
                [
                    group["x"].to_numpy(dtype=np.float32),
                    group["y"].to_numpy(dtype=np.float32),
                    heading,
                    speed,
                    group["t"].to_numpy(dtype=np.float32),
                ]
            ).astype(np.float32)
            trajectories[int(track_id)] = trajectory
        return trajectories

    raise ValueError(
        "Unsupported sensor schema. Expected either "
        "['trackId','latitude','longitude','heading','speed','timestamp'] or "
        "['trajectory_id','x','y','vx','vy','t']."
    )


def split_sub_trajectories(
    trajectories: Dict[int, np.ndarray],
    min_len: int = 6,
    max_len: int = 18,
    stride: int = 2,
) -> List[dict]:
    sub_tracks: List[dict] = []
    for track_id, track in trajectories.items():
        n_points = len(track)
        if n_points < min_len:
            continue
        for window in range(min_len, min(max_len, n_points) + 1):
            for start in range(0, n_points - window + 1, stride):
                segment = track[start : start + window]
                sub_tracks.append(
                    {
                        "track_id": track_id,
                        "segment": segment,
                        "t_start": float(segment[0, -1]),
                        "t_end": float(segment[-1, -1]),
                    }
                )
    return sub_tracks


def unwrap_heading_rad(angle_rad: np.ndarray) -> np.ndarray:
    if len(angle_rad) == 0:
        return angle_rad
    return np.unwrap(angle_rad)


def finite_diff(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values)
    diffs = np.diff(values, axis=0)
    if values.ndim == 1:
        return np.concatenate([[0.0], diffs], axis=0).astype(np.float32)
    zero = np.zeros((1, values.shape[1]), dtype=np.float32)
    return np.concatenate([zero, diffs], axis=0).astype(np.float32)


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim == 2 else 1), dtype=np.float32)
    if len(seq) == target_len:
        return seq.astype(np.float32)

    idx = np.linspace(0, len(seq) - 1, target_len)
    left = np.floor(idx).astype(int)
    right = np.ceil(idx).astype(int)
    weight = (idx - left).astype(np.float32)

    if seq.ndim == 1:
        seq = seq[:, None]
        squeeze = True
    else:
        squeeze = False

    out = (1.0 - weight[:, None]) * seq[left] + weight[:, None] * seq[right]
    out = out.astype(np.float32)
    if squeeze:
        return out[:, 0]
    return out


def compute_curvature(delta_xy: np.ndarray) -> np.ndarray:
    if len(delta_xy) == 0:
        return np.zeros(0, dtype=np.float32)
    headings = np.arctan2(delta_xy[:, 1], delta_xy[:, 0] + 1e-6)
    headings = unwrap_heading_rad(headings)
    return finite_diff(headings).astype(np.float32)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) + 1e-6
    return ((values - median) / (1.4826 * mad)).astype(np.float32)


def build_track_timestep_features(segment: np.ndarray) -> np.ndarray:
    lat_lon = segment[:, :2].astype(np.float32)
    heading_deg = segment[:, 2].astype(np.float32)
    speed = segment[:, 3].astype(np.float32)
    timestamp = segment[:, 4].astype(np.float32)

    origin = lat_lon[0:1]
    rel_pos = lat_lon - origin
    delta_xy = finite_diff(lat_lon)
    step_dist = np.linalg.norm(delta_xy, axis=1).astype(np.float32)
    heading_rad = np.deg2rad(heading_deg)
    heading_rad = unwrap_heading_rad(heading_rad)
    delta_heading = finite_diff(heading_rad)
    delta_speed = finite_diff(speed)
    delta_time = finite_diff(timestamp)
    curvature = compute_curvature(delta_xy)
    radial_dist = np.linalg.norm(rel_pos, axis=1).astype(np.float32)

    features = np.column_stack(
        [
            rel_pos[:, 0],
            rel_pos[:, 1],
            delta_xy[:, 0],
            delta_xy[:, 1],
            step_dist,
            speed,
            delta_speed,
            np.cos(heading_rad),
            np.sin(heading_rad),
            delta_heading,
            curvature,
            delta_time,
            radial_dist,
        ]
    ).astype(np.float32)

    return features


def build_report_timestep_features(row: pd.Series) -> np.ndarray:
    lons = safe_array(row["coors.longitudes"])
    lats = safe_array(row["coors.latitudes"])
    n_points = min(len(lons), len(lats))
    if n_points == 0:
        return np.zeros((1, 13), dtype=np.float32)

    lat_lon = np.column_stack([lats[:n_points], lons[:n_points]]).astype(np.float32)
    origin = lat_lon[0:1]
    rel_pos = lat_lon - origin
    delta_xy = finite_diff(lat_lon)
    step_dist = np.linalg.norm(delta_xy, axis=1).astype(np.float32)
    heading_rad = np.arctan2(delta_xy[:, 1], delta_xy[:, 0] + 1e-6).astype(np.float32)
    heading_rad = unwrap_heading_rad(heading_rad)
    speed = step_dist.copy()
    delta_speed = finite_diff(speed)
    delta_heading = finite_diff(heading_rad)
    curvature = compute_curvature(delta_xy)
    delta_time = np.ones(len(lat_lon), dtype=np.float32)
    delta_time[0] = 0.0
    radial_dist = np.linalg.norm(rel_pos, axis=1).astype(np.float32)

    features = np.column_stack(
        [
            rel_pos[:, 0],
            rel_pos[:, 1],
            delta_xy[:, 0],
            delta_xy[:, 1],
            step_dist,
            speed,
            delta_speed,
            np.cos(heading_rad),
            np.sin(heading_rad),
            delta_heading,
            curvature,
            delta_time,
            radial_dist,
        ]
    ).astype(np.float32)

    return features


def fit_feature_stats(sequences: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(sequences, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_sequences(
    sequences: Sequence[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> List[np.ndarray]:
    return [((seq - mean) / std).astype(np.float32) for seq in sequences]


def append_global_context(sequences: Sequence[np.ndarray]) -> List[np.ndarray]:
    enriched = []
    for seq in sequences:
        path_extent = np.max(seq[:, :2], axis=0) - np.min(seq[:, :2], axis=0)
        path_extent = np.repeat(path_extent[None, :], len(seq), axis=0)
        seq_len = np.full((len(seq), 1), float(len(seq)), dtype=np.float32)
        progress = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)[:, None]
        enriched.append(np.concatenate([seq, path_extent, seq_len, progress], axis=1).astype(np.float32))
    return enriched


@dataclass
class SequenceBundle:
    track_sequences: List[np.ndarray]
    report_sequences: List[np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    feature_dim: int


def build_sequence_bundle(sub_tracks: Sequence[dict], event_df: pd.DataFrame) -> SequenceBundle:
    track_sequences = [build_track_timestep_features(item["segment"]) for item in sub_tracks]
    report_sequences = [build_report_timestep_features(row) for _, row in event_df.iterrows()]

    raw_sequences = track_sequences + report_sequences
    mean, std = fit_feature_stats(raw_sequences)
    track_sequences = standardize_sequences(track_sequences, mean, std)
    report_sequences = standardize_sequences(report_sequences, mean, std)

    track_sequences = append_global_context(track_sequences)
    report_sequences = append_global_context(report_sequences)

    feature_dim = track_sequences[0].shape[1] if track_sequences else report_sequences[0].shape[1]
    return SequenceBundle(
        track_sequences=track_sequences,
        report_sequences=report_sequences,
        mean=mean,
        std=std,
        feature_dim=feature_dim,
    )


class SequenceTripletDataset(Dataset):
    def __init__(
        self,
        triplets: Sequence[Tuple[int, int, int]],
        report_sequences: Sequence[np.ndarray],
        track_sequences: Sequence[np.ndarray],
    ):
        self.triplets = list(triplets)
        self.report_sequences = report_sequences
        self.track_sequences = track_sequences

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        report_idx, pos_idx, neg_idx = self.triplets[idx]
        return (
            self.report_sequences[report_idx],
            self.track_sequences[pos_idx],
            self.track_sequences[neg_idx],
        )


def pad_batch(sequences: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    feat_dim = sequences[0].shape[1]
    batch = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        batch[idx, :seq_len] = seq
        mask[idx, :seq_len] = 1.0
    return torch.from_numpy(batch), torch.from_numpy(mask)


def triplet_collate_fn(batch):
    report_seqs, pos_seqs, neg_seqs = zip(*batch)
    report_batch, report_mask = pad_batch(report_seqs)
    pos_batch, pos_mask = pad_batch(pos_seqs)
    neg_batch, neg_mask = pad_batch(neg_seqs)
    return report_batch, report_mask, pos_batch, pos_mask, neg_batch, neg_mask


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn = self.score(sequence).squeeze(-1)
        attn = attn.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(attn, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class GRUSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.pool = AttentionPooling(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, emb_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, hidden = self.gru(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if sequence.size(1) < x.size(1):
            pad_len = x.size(1) - sequence.size(1)
            pad = torch.zeros(
                sequence.size(0),
                pad_len,
                sequence.size(2),
                device=sequence.device,
                dtype=sequence.dtype,
            )
            sequence = torch.cat([sequence, pad], dim=1)

        pooled = self.pool(sequence, mask)
        hidden = hidden.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)
        last_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)
        embedding = self.head(torch.cat([pooled, last_hidden], dim=-1))
        return F.normalize(embedding, dim=-1)


def create_triplets_gt(
    sub_tracks: Sequence[dict],
    report_sequences: Sequence[np.ndarray],
    track_sequences: Sequence[np.ndarray],
    gt_dict: Dict[int, int],
    num_pos: int = 6,
    num_neg: int = 18,
) -> List[Tuple[int, int, int]]:
    del report_sequences, track_sequences
    track_to_sub: Dict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(sub_tracks):
        track_to_sub[int(item["track_id"])].append(idx)

    all_indices = np.arange(len(sub_tracks))
    triplets: List[Tuple[int, int, int]] = []
    for report_id, true_track in gt_dict.items():
        positives = track_to_sub.get(int(true_track), [])
        if not positives:
            continue
        pos_sample = random.sample(positives, min(num_pos, len(positives)))
        neg_candidates = [idx for idx in all_indices if int(sub_tracks[idx]["track_id"]) != int(true_track)]
        if not neg_candidates:
            continue
        neg_sample = random.sample(neg_candidates, min(num_neg, len(neg_candidates)))
        for pos_idx in pos_sample:
            for neg_idx in neg_sample:
                triplets.append((int(report_id), int(pos_idx), int(neg_idx)))
    return triplets


def train_gru_triplet(
    report_sequences: Sequence[np.ndarray],
    track_sequences: Sequence[np.ndarray],
    triplets: Sequence[Tuple[int, int, int]],
    feature_dim: int,
    epochs: int = 18,
    batch_size: int = 64,
    hidden_dim: int = 96,
    emb_dim: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
):
    if not triplets:
        raise ValueError("No triplets generated from ground truth.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SequenceTripletDataset(triplets, report_sequences, track_sequences)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn,
    )

    report_encoder = GRUSequenceEncoder(feature_dim, hidden_dim=hidden_dim, emb_dim=emb_dim).to(device)
    track_encoder = GRUSequenceEncoder(feature_dim, hidden_dim=hidden_dim, emb_dim=emb_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(report_encoder.parameters()) + list(track_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=0.25,
        reduction="mean",
    )

    for epoch in range(epochs):
        report_encoder.train()
        track_encoder.train()
        total_loss = 0.0

        for report_batch, report_mask, pos_batch, pos_mask, neg_batch, neg_mask in loader:
            report_batch = report_batch.to(device)
            report_mask = report_mask.to(device)
            pos_batch = pos_batch.to(device)
            pos_mask = pos_mask.to(device)
            neg_batch = neg_batch.to(device)
            neg_mask = neg_mask.to(device)

            optimizer.zero_grad()
            report_emb = report_encoder(report_batch, report_mask)
            pos_emb = track_encoder(pos_batch, pos_mask)
            neg_emb = track_encoder(neg_batch, neg_mask)
            loss = loss_fn(report_emb, pos_emb, neg_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(report_encoder.parameters()) + list(track_encoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch + 1}: loss={total_loss / max(len(loader), 1):.4f}")

    return track_encoder, report_encoder


def encode_all_sequences(
    encoder: GRUSequenceEncoder,
    sequences: Sequence[np.ndarray],
    batch_size: int = 256,
    device: Optional[str] = None,
) -> torch.Tensor:
    device = device or next(encoder.parameters()).device
    encoder.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            batch, mask = pad_batch(batch_sequences)
            batch = batch.to(device)
            mask = mask.to(device)
            outputs.append(encoder(batch, mask).cpu())
    return torch.cat(outputs, dim=0)


def match_reports_topk(
    track_encoder: GRUSequenceEncoder,
    report_encoder: GRUSequenceEncoder,
    sub_tracks: Sequence[dict],
    track_sequences: Sequence[np.ndarray],
    report_sequences: Sequence[np.ndarray],
    top_k: int = 5,
    device: Optional[str] = None,
) -> Dict[int, List[dict]]:
    track_emb = encode_all_sequences(track_encoder, track_sequences, device=device)
    report_emb = encode_all_sequences(report_encoder, report_sequences, device=device)
    distances = torch.cdist(report_emb, track_emb)

    report_topk: Dict[int, List[dict]] = {}
    for report_id in range(distances.size(0)):
        row = distances[report_id].numpy()
        order = np.argsort(row)[:top_k]
        report_topk[report_id] = [
            {
                "track_id": int(sub_tracks[sub_idx]["track_id"]),
                "sub_idx": int(sub_idx),
                "score": float(row[sub_idx]),
            }
            for sub_idx in order
        ]
    return report_topk


def aggregate_report_topk(report_topk: Dict[int, List[dict]]) -> Dict[int, List[dict]]:
    aggregated: Dict[int, List[dict]] = {}
    for report_id, candidates in report_topk.items():
        best_by_track: Dict[int, dict] = {}
        for item in candidates:
            track_id = int(item["track_id"])
            if track_id not in best_by_track or item["score"] < best_by_track[track_id]["score"]:
                best_by_track[track_id] = item
        aggregated[report_id] = sorted(best_by_track.values(), key=lambda x: x["score"])
    return aggregated


def evaluate_topk(report_to_tracks: Dict[int, List[dict]], gt_dict: Dict[int, int], k: int = 5) -> float:
    hits = 0
    total = 0
    for report_id, candidates in report_to_tracks.items():
        true_track = gt_dict.get(report_id)
        if true_track is None:
            continue
        total += 1
        top_tracks = [int(item["track_id"]) for item in candidates[:k]]
        if int(true_track) in top_tracks:
            hits += 1
    recall = hits / total if total else 0.0
    print(f"Recall@{k}: {recall:.3f} ({hits}/{total})")
    return recall


def run_pipeline_gru_feature_rich(
    sensor_df: pd.DataFrame,
    event_df: pd.DataFrame,
    gt_dict: Dict[int, int],
    top_k: int = 5,
    min_len: int = 6,
    max_len: int = 18,
    stride: int = 2,
    epochs: int = 18,
    hidden_dim: int = 96,
    emb_dim: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    device: Optional[str] = None,
):
    set_seed(seed)
    event_df = normalize_event_df(event_df).reset_index(drop=True)
    trajectories = build_trajectories(sensor_df)
    sub_tracks = split_sub_trajectories(
        trajectories,
        min_len=min_len,
        max_len=max_len,
        stride=stride,
    )
    sequence_bundle = build_sequence_bundle(sub_tracks, event_df)
    triplets = create_triplets_gt(
        sub_tracks,
        sequence_bundle.report_sequences,
        sequence_bundle.track_sequences,
        gt_dict,
    )

    track_encoder, report_encoder = train_gru_triplet(
        sequence_bundle.report_sequences,
        sequence_bundle.track_sequences,
        triplets,
        feature_dim=sequence_bundle.feature_dim,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        lr=lr,
        device=device,
    )

    report_topk = match_reports_topk(
        track_encoder,
        report_encoder,
        sub_tracks,
        sequence_bundle.track_sequences,
        sequence_bundle.report_sequences,
        top_k=top_k,
        device=device,
    )
    report_to_tracks = aggregate_report_topk(report_topk)
    return {
        "sub_tracks": sub_tracks,
        "report_to_tracks": report_to_tracks,
        "track_encoder": track_encoder,
        "report_encoder": report_encoder,
        "sequence_bundle": sequence_bundle,
        "triplets": triplets,
    }


def export_topk_csv(report_to_tracks: Dict[int, List[dict]], output_path: str, top_k: int = 5) -> None:
    rows = []
    for report_id, candidates in report_to_tracks.items():
        for rank, item in enumerate(candidates[:top_k], start=1):
            rows.append(
                {
                    "report_id": report_id,
                    "rank": rank,
                    "track_id": int(item["track_id"]),
                    "score": float(item["score"]),
                    "sub_idx": int(item["sub_idx"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)
