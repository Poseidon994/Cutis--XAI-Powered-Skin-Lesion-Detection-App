"""Do the three explanation methods point where the model actually looks?

A toy network whose score depends only on how dark the top-left 48x48 pixels are (darker = higher score) is wrapped as a one-network ensemble.
A good explainer must put (almost) all of its positive evidence in that corner and get the sign right.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.artifacts import _load_module, VENDORED_LIB
from app.ensemble import Net
from app import explain as xai

L = _load_module(VENDORED_LIB)
S = 128


class Stage(nn.Module):
    def forward(self, x):
        f = 1.0 - nn.functional.avg_pool2d(x.mean(1, keepdim=True), 16)  # (B, 1, 8, 8): darker -> larger activation
        m = torch.zeros_like(f)
        m[:, :, :3, :3] = 1
        return torch.cat([f * m, 0.5 * f * m], 1)                          # (B, 2, 8, 8)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(Stage())

    def forward(self, x):
        return self.stages(x).flatten(1).sum(1, keepdim=True)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Linear(1, 5)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 10.0           # darker corner -> higher malignant logit
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(self.backbone(x))


class FakeEns:
    def __init__(self):
        self.dev = torch.device('cpu')
        mod = Toy().eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        self.nets = [Net('toy', 0, mod, L.Img(self.dev, S, (0, 0, 0), (1, 1, 1)), False, S, {})]
        self.stats = {'toy': {'f0_h0': [0.0, 1.0]}}

    def explain_nets(self):
        return self.nets


class Cfg:
    cam_method = 'gradcam'
    lime_samples = 400
    shap_samples = 64


def image():
    u8 = np.zeros((S, S, 3), np.uint8)
    u8[:] = (205, 160, 140)
    u8[8:44, 8:44] = (70, 40, 30)                    # dark lesion inside the sensitive corner
    u8[84:120, 84:120] = (70, 40, 30)                # an equally dark decoy where the model does not look
    return u8


def corner_share(m):
    m = np.clip(m, 0, None)
    return float(m[:48, :48].sum() / m.sum())


@pytest.fixture(scope='module')
def ens():
    return FakeEns()


@pytest.mark.parametrize('method', ['gradcam', 'lime', 'shap'])
def test_attribution_goes_to_the_corner(ens, method):
    out = xai.run(method, ens, image(), Cfg())
    share = corner_share(out['map'])
    assert share > 0.8, f'{method}: only {share:.2f} of the positive evidence is in the sensitive corner'
    assert out['summary']['direction'] in ('upper left', 'upper', 'left', 'centre')


def test_score_cam_also_localises(ens):
    class C(Cfg):
        cam_method = 'scorecam'
    out = xai.run('gradcam', ens, image(), C())
    assert out['cam_method'] == 'scorecam' and corner_share(out['map']) > 0.7


def test_sign_is_right(ens):
    """Making the corner lighter should lower the score, so LIME / SHAP must report it as evidence for higher risk when dark."""
    out = xai.run('shap', ens, image(), Cfg())
    assert out['summary']['attribution_sum'] > 0
    assert abs(out['summary']['completeness_gap']) < 0.15 * abs(out['summary']['output_z'] - out['summary']['reference_z'])


def test_direct_expected_gradients_matches_shap_library(ens):
    from app.explain.common import Scorer, skin_reference
    from app.explain import shap_xai
    sc = Scorer(ens)
    u8 = image()
    x = sc.to_tensor(u8)
    bg = shap_xai._backgrounds(sc, skin_reference(u8), S)
    direct = shap_xai._expected_gradients(sc, x, bg, 64, 8)[0].numpy().sum(0)
    lib = xai.run('shap', ens, u8, Cfg())['map']
    assert direct.sum() == pytest.approx(lib.sum(), rel=0.35)
    assert float(np.clip(direct, 0, None)[:48, :48].sum() / np.clip(direct, 0, None).sum()) > 0.8
