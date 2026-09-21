"""Loading state dicts written by earlier versions."""

import torch
from conftest import make_model

from cliffordip.train.checkpoints import migrate_checkpoint_file, migrate_state_dict


def test_grade_three_bias_entries_are_dropped():
    sd = {
        "layer.w0": torch.zeros(2, 2),
        "layer.b0": torch.zeros(2, 1),
        "layer.b3": torch.zeros(2, 1),
    }
    migrated = migrate_state_dict(sd)
    assert "layer.b3" not in migrated
    assert "layer.b0" in migrated


def test_stacked_linear_weight_is_split_per_grade():
    sd = {
        "layer.weight": torch.arange(8 * 2 * 3, dtype=torch.float32).view(8, 2, 3),
        "layer.grade_mask": torch.ones(8),
    }
    migrated = migrate_state_dict(sd)
    assert set(migrated) == {"layer.w0", "layer.w1", "layer.w2", "layer.w3"}
    assert all(migrated[f"layer.w{g}"].shape == (2, 3) for g in range(4))


def test_wide_edge_embedding_columns_are_trimmed():
    sd = {
        "net.edge_embed.rbf.offsets": torch.zeros(8),
        "net.edge_embed.scalar_net.0.weight": torch.zeros(4, 13),
        "net.edge_embed.vector_net.0.weight": torch.zeros(4, 13),
    }
    migrated = migrate_state_dict(sd)
    assert migrated["net.edge_embed.scalar_net.0.weight"].shape == (4, 8)
    assert migrated["net.edge_embed.vector_net.0.weight"].shape == (4, 8)


def test_migration_is_idempotent():
    sd = {"layer.b3": torch.zeros(2, 1), "layer.w0": torch.zeros(2, 2)}
    once = migrate_state_dict(sd)
    assert migrate_state_dict(once).keys() == once.keys()


def test_a_current_state_dict_is_unchanged():
    model = make_model()
    sd = model.state_dict()
    migrated = migrate_state_dict(sd)
    assert migrated.keys() == sd.keys()


def test_a_state_dict_with_extra_bias_entries_loads_after_migration():
    model = make_model()
    sd = dict(model.state_dict())
    for name in [k for k in sd if k.endswith(".b0")]:
        sd[name.replace(".b0", ".b3")] = torch.zeros_like(sd[name])
    model.load_state_dict(migrate_state_dict(sd), strict=True)


def test_checkpoint_file_round_trip(tmp_path):
    path = tmp_path / "old.ckpt"
    torch.save(
        {"state_dict": {"layer.b3": torch.zeros(2, 1), "layer.w0": torch.zeros(2, 2)}, "epoch": 3},
        path,
    )
    out = tmp_path / "new.ckpt"
    ckpt = migrate_checkpoint_file(str(path), str(out))
    assert "layer.b3" not in ckpt["state_dict"]
    assert ckpt["epoch"] == 3
    assert out.exists()
