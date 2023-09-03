import torch
from vllm.config import LoRAConfig
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import ColumnParallelLinear
from vllm.model_executor.weight_utils import load_tensor_parallel_weights



def init_method_const(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)

    return init_

class LoraLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
    ):
        super().__init__()
        self.linear_in = ColumnParallelLinear(
            in_features, dim, bias=False, gather_output=True, init_method=self._get_init_fn("xavier")
        )
        self.linear_out = ColumnParallelLinear(
            dim, out_features, bias=False, gather_output=False, init_method=self._get_init_fn("zero")
        )

    def _get_init_fn(self, init_method: str):
        if init_method == 'xavier':
            init_fn = torch.nn.init.xavier_normal_
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, or xavier")
        return init_fn

    def forward(self, x):
        x, _ = self.linear_in(x)  
        x, _ = self.linear_out(x)
        return x, None
    
class LoRAModel(torch.nn.Module):
    def __init__(
            self, 
            config:LoRAConfig
    ):
        self.config = config

        self.layers = torch.nn.ModuleList([
            LoraLayer(config) for _ in range (config.num_layers)
        ])

    def forward(self,
                hidden_states: torch.Tensor,
                layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        output, _ = layer(hidden_states)
        return output
    
    def map_name(self, name):
        return name.replace('model.language_model.encoder.', 'model.').replace("adapter_layer.lora_kqv_adapter.", "lora_layer.")
    
    _column_parallel_weights = ["lora_layer.linear_in.weight"]
    _row_parallel_weights = []

    def load_lora_weights(self, lora_path: str):
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for nemo_name, loaded_weight in torch.load(lora_path, map_location="cpu").items():
            name = self.map_name(nemo_name)
            param = state_dict[name]
            if "lora_layer.linear_out" in name:
                # NVGPT's fused QKV has the shape of
                # [num_heads * 3 * head_size, hidden_size], while the
                # required shape is [3 * num_heads * head_size, hidden_size].
                # Thus, we need weight conversion.
                # In the case of lora_layer.linear_out we have [num_heads * 3 * head_size, adapter_dim] shaped matrix
                shard_size = param.shape[0]
                start = shard_size * tensor_model_parallel_rank
                end = shard_size * (tensor_model_parallel_rank + 1)
                loaded_weight = loaded_weight[start:end]

                num_heads_per_rank = self.config.num_attention_heads // get_tensor_model_parallel_world_size()
                head_size = self.config.hidden_size // self.config.num_attention_heads
                rank_size = 32

                #loaded_weight = loaded_weight.view(-1, 3, head_size, hidden_size)
                loaded_weight = loaded_weight.view(num_heads_per_rank, 3, head_size, rank_size)
                loaded_weight = loaded_weight.transpose(0, 1)
                loaded_weight = loaded_weight.reshape(3 * head_size * num_heads_per_rank, rank_size)                  

            #loaded_weight = loaded_weight.to(torch.bfloat16)
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)    