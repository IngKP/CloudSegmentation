import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from benchmark_models import percentile, select_model


def test_percentile_and_deterministic_selection():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
    rows = [
        {"id": "unet_ch_01_seed42", "archived_iou_mean": 0.70, "cpu": {"p50_ms": 12}},
        {"id": "arch_resnet34_seed42", "archived_iou_mean": 0.71, "cpu": {"p50_ms": 4}},
        {"id": "garg_benchmark_seed42", "archived_iou_mean": 0.65, "cpu": {"p50_ms": 2}},
    ]
    assert select_model(rows) == "arch_resnet34_seed42"


def test_two_channel_model_is_preferred_when_within_speed_gate():
    rows = [
        {"id": "unet_ch_01_seed42", "archived_iou_mean": 0.70, "cpu": {"p50_ms": 5}},
        {"id": "arch_resnet34_seed42", "archived_iou_mean": 0.71, "cpu": {"p50_ms": 4}},
    ]
    assert select_model(rows) == "unet_ch_01_seed42"
