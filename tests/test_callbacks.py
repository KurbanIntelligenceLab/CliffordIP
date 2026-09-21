"""Training callbacks."""

import torch
from conftest import make_model

from cliffordip.lightning.callbacks import DeNSCallback, EMACallback, ExponentialMovingAverage


def test_ema_tracks_a_running_average():
    model = make_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update()
    name = next(iter(before))
    assert torch.allclose(ema.shadow[name], 0.5 * before[name] + 0.5 * (before[name] + 1.0))


def test_apply_shadow_and_restore_round_trip():
    model = make_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    live = {k: v.detach().clone() for k, v in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.apply_shadow()
    name = next(iter(ema.shadow))
    assert torch.allclose(dict(model.named_parameters())[name], live[name])
    ema.restore()
    assert torch.allclose(dict(model.named_parameters())[name], live[name] + 2.0)


def test_ema_state_survives_a_save_and_load():
    model = make_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.update()
    state = ema.state_dict()

    restored = ExponentialMovingAverage(make_model(seed=9), decay=0.9)
    restored.load_state_dict(state)
    for name, tensor in state.items():
        assert torch.allclose(restored.shadow[name], tensor)


def test_ema_callback_checkpoints_its_averages():
    class Module:
        model = make_model()

    callback = EMACallback(decay=0.99)
    callback.on_fit_start(None, Module())
    saved = callback.state_dict()
    assert saved["decay"] == 0.99
    assert saved["shadow"]

    reloaded = EMACallback()
    reloaded.load_state_dict(saved)
    reloaded.on_fit_start(None, Module())
    assert reloaded._decay == 0.99
    assert reloaded._ema.shadow.keys() == saved["shadow"].keys()


def test_dens_weight_warms_up_then_holds():
    class Trainer:
        current_epoch = 0

    class Module:
        dens_weight = 0.0

    callback = DeNSCallback(max_weight=0.2, warmup_epochs=4)
    trainer, module = Trainer(), Module()
    seen = []
    for epoch in range(6):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, module)
        seen.append(module.dens_weight)
    assert seen == [0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.2]


def test_dens_weight_is_constant_without_warmup():
    class Trainer:
        current_epoch = 0

    class Module:
        dens_weight = 0.0

    module = Module()
    DeNSCallback(max_weight=0.3, warmup_epochs=0).on_train_epoch_start(Trainer(), module)
    assert module.dens_weight == 0.3
