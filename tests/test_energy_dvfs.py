"""DVFS ([B] C4) & hiệu chỉnh ràng buộc năng lượng ([B] C3)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig

F = 2e9


@pytest.fixture
def env():
    return VehicularEdgeEnvironment(EnvironmentConfig())


def test_dvfs_equal_frequency_when_upload_not_binding(env):
    f_en, f_l = env._dvfs_frequencies(C_enc=2e8, C_local=2e8, t_remote=0.0, tau=0.5, F_loc=F)
    assert f_en == pytest.approx(8e8, rel=1e-5) and f_l == pytest.approx(8e8, rel=1e-5)


def test_dvfs_encryption_speeds_up_for_upload_deadline(env):
    f_en, f_l = env._dvfs_frequencies(C_enc=4e8, C_local=1e8, t_remote=0.6, tau=1.0, F_loc=F)
    assert f_en == pytest.approx(1e9, rel=1e-4)                  # 4e8 / (1 − 0.6)
    assert f_l == pytest.approx(max(1e8 / 0.6, 0.1 * F), rel=1e-4)
    assert 4e8 / f_en + 0.6 <= 1.0 + 1e-9


def test_dvfs_clips_to_max_frequency_and_disabled_uses_max():
    env = VehicularEdgeEnvironment(EnvironmentConfig())
    assert env._dvfs_frequencies(1e10, 0.0, 0.0, 1.0, F) == (F, F)
    off = VehicularEdgeEnvironment(EnvironmentConfig(dvfs_enabled=False))
    assert off._dvfs_frequencies(1e6, 1e6, 0.0, 1.0, F) == (F, F)


def test_dvfs_all_local_saves_energy_and_meets_deadline():
    res = {}
    for dvfs in (False, True):
        env = VehicularEdgeEnvironment(EnvironmentConfig(dvfs_enabled=dvfs))
        env.reset(seed=3)
        user = env.users[0]
        task = env.task_queue[user.id][0]
        tau, _ = env.effective_deadline(user, task)
        res[dvfs] = env._execute_task(user, task, 1.0, 0.0, 0.0, 0, tau, 32.0), tau
    (m_off, tau), (m_on, _) = res[False], res[True]
    assert m_on["energy_user"] < m_off["energy_user"]
    assert not m_on["viol_T"] and m_on["delay"] <= tau


def test_calibrated_model_is_non_degenerate():
    """Hiệu chỉnh (scripts/calibrate_energy.py): bài toán khả thi, offload là tối ưu ở phần đáng kể
    state offload được, và N_i tối ưu thay đổi theo state (không suy biến)."""
    from scripts.calibrate_energy import evaluate_config
    df = evaluate_config(n_states=120)
    assert df.feasible_any.mean() >= 0.85
    assert df.best_offload.dropna().astype(float).mean() >= 0.2
    assert df.best_N_i.dropna().nunique() >= 2
