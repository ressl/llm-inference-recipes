"""Execute the actual wrapper constructor with tiny towers and no LLM weights.

Run inside the runtime image. Real PyTorch module registration, meta-device
construction and vLLM weight loading verify that later PP stages do not allocate
or attempt to execute unused image encoders.
"""
import ast
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch import nn
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader, StageMissingLayer
from vllm.config import MultiModalConfig
from vllm.model_executor.offloader import set_offloader, NoopOffloader

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/vl_model.py'
)
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == 'DeepseekV41ForCausalLM')
cls.bases = [ast.Name(id='Base', ctx=ast.Load())]
cls.decorator_list = []
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
            and n.name == '__init__']

class Base(nn.Module, SupportsMultiModal):
    pass

class Tower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, 8))
    def forward(self, x):
        return x @ self.weight

class Backbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4))
        self.make_empty_intermediate_tensors = lambda *args: None

namespace = dict(Base=Base, nn=nn, torch=torch,
    DeepseekV4ViT=Tower, DeepseekV4Aligner=Tower,
    DeepseekV41LLMForCausalLM=Backbone,
    is_vit_use_data_parallel=lambda heads: False,
    maybe_prefix=lambda prefix, name: name,
    WeightsMapper=lambda: None,
    _make_deepseek_v4_vl_weights_mapper=lambda *args: None,
    _linear_scale_param_name=lambda *args: 'weight_scale')
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
             str(path), 'exec'), namespace)
Model = namespace['DeepseekV41ForCausalLM']

from unittest.mock import patch
set_offloader(NoopOffloader())
for first in (True, False):
    for image_limit in (0, 4):
        mm = MultiModalConfig(limit_per_prompt={'image': image_limit})
        mc = NS(hf_config=NS(vision_n_heads=16, hidden_size=4),
                dtype=torch.bfloat16, multimodal_config=mm,
                get_multimodal_config=lambda: mm)
        vc = NS(model_config=mc)
        with patch('vllm.distributed.get_pp_group', return_value=NS(is_first_rank=first)):
            model = Model(vllm_config=vc)
        expected_active = first and image_limit > 0
        assert isinstance(model.vision, StageMissingLayer) != expected_active, (
            'Unused vision tower allocated on later PP stage', first, image_limit)
        assert isinstance(model.aligner, StageMissingLayer) != expected_active
        assert model.language_model.weight.device.type == 'cpu'
        weights = [('vision.weight', torch.ones(8, 8)),
                   ('aligner.weight', torch.full((8, 8), 2.)),
                   ('language_model.weight', torch.full((4,), 3.))]
        loaded = AutoWeightsLoader(model).load_weights(weights)
        assert ('vision.weight' in loaded) == expected_active
        assert ('aligner.weight' in loaded) == expected_active
        assert torch.all(model.language_model.weight == 3)
        if expected_active:
            assert model.vision.weight.dtype == torch.bfloat16
            assert torch.all(model.vision.weight == 1)
        else:
            try:
                model.vision(torch.ones(1, 8))
            except RuntimeError as error:
                assert 'should not be called' in str(error)
            else:
                raise AssertionError('Missing tower was executable')
print('First PP stage retains vision; later stages skip tower allocation/loading; text-only stays supported')
