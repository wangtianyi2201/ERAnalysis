```python
import pandas as pd

from gru_feature_pipeline import (
    build_gt_dict,
    evaluate_topk,
    export_topk_csv,
    run_pipeline_gru_feature_rich,
)

sensor_df = pd.read_parquet("sensor.parquet")
event_df = pd.read_parquet("event.parquet")
gt_df = pd.read_parquet("ground_truth.parquet")

gt_dict = build_gt_dict(gt_df, event_df)

artifacts = run_pipeline_gru_feature_rich(
    sensor_df,
    event_df,
    gt_dict,
    top_k=5,
    min_len=6,
    max_len=18,
    stride=2,
    epochs=18,
    hidden_dim=96,
    emb_dim=128,
    batch_size=64,
    lr=3e-4,
)

recall = evaluate_topk(artifacts["report_to_tracks"], gt_dict, k=5)
print("GRU feature-rich Recall@5:", recall)

export_topk_csv(
    artifacts["report_to_tracks"],
    "report_topk_gru_feature_rich.csv",
    top_k=5,
)
```

Gợi ý tuning nhanh:

```python
artifacts = run_pipeline_gru_feature_rich(
    sensor_df,
    event_df,
    gt_dict,
    top_k=5,
    min_len=8,
    max_len=24,
    stride=3,
    epochs=24,
    hidden_dim=128,
    emb_dim=160,
    batch_size=48,
    lr=2e-4,
)
```
