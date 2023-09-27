import io
import uuid
from enum import Enum
from typing import Optional
import torch

from vllm.config import ModelConfig
from vllm.model_executor.weight_utils import (load_tensor_parallel_weights)


class CustomizationType(Enum):
    NVGPT_LORA = "nvgpt_lora"


class Customization:
    id: uuid.UUID
    type: CustomizationType
    checkpoint: bytes
    dtype: torch.dtype
    rank: int
    state_dict: dict

    def __init__(
            self,
            id: uuid.UUID,
            type: CustomizationType,
            checkpoint: bytes,
            dtype: torch.dtype,
            rank: int,
            world_size: int,
            model_config: ModelConfig
    ) -> None:
        self.id = id
        self.type = type
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.rank = rank
        self.world_size = world_size
        self.model_config = model_config

        self.process_checkpoint(self.model_config)

    def get_state_dict(self):
        return self.state_dict

    def process_checkpoint(self, model_config: ModelConfig):
        # TODO: mkhadkevich only load a chunk of customization for the current rank
        self.state_dict = torch.load(io.BytesIO(self.checkpoint), map_location='cpu')
        self.process_nvgpt_lora_checkpoint(model_config)
        for key in self.state_dict.keys():
            self.state_dict[key] = self.state_dict[key].to(self.dtype).cuda()

    @staticmethod
    def get_column_parallel_weights():
        return ["ptuning_embeddings.weight", "embedding.weight", "lm_head.weight", "dense_h_to_4h.weight",
                "lora_layer.linear_in.weight"]

    @staticmethod
    def get_row_parallel_weights():
        return ["dense.weight", "dense_4h_to_h.weight"]

    def process_nvgpt_lora_checkpoint(self, model_config: ModelConfig):
        for nemo_name, loaded_weight in self.state_dict.items():
            cloned = loaded_weight.clone()
            cloned2 = loaded_weight.clone()
            if "adapter_layer.lora_kqv_adapter.linear_out" in nemo_name:
                # NVGPT's fused QKV has the shape of
                # [num_heads * 3 * head_size, hidden_size], while the
                # required shape is [3 * num_heads * head_size, hidden_size].
                # Thus, we need weight conversion.
                # In the case of lora_layer.linear_out we have [num_heads * 3 * head_size, adapter_dim] shaped matrix
                shard_size = loaded_weight.shape[0]
                start = shard_size * self.rank
                end = shard_size * (self.rank + 1)
                loaded_weight = loaded_weight[start:end]

                num_heads_per_rank = model_config.num_attention_heads // self.world_size
                head_size = model_config.hidden_size // model_config.num_attention_heads
                rank_size = loaded_weight.shape[1] // self.world_size

                loaded_weight = loaded_weight.view(num_heads_per_rank, 3, head_size, rank_size)
                loaded_weight = loaded_weight.transpose(0, 1)
                loaded_weight = loaded_weight.reshape(3 * head_size * num_heads_per_rank, rank_size)

            load_tensor_parallel_weights(
                cloned2,
                loaded_weight,
                nemo_name,
                self.get_column_parallel_weights(),
                self.get_row_parallel_weights(),
                self.rank,
            )
            self.state_dict[nemo_name] = cloned2


class CustomizationCache:
    def __init__(self):
        self._cache = dict()

    def put(self, customization: Customization):
        self._cache[customization.id] = customization

    def get(self, customization_id: uuid.UUID) -> Optional[Customization]:
        return self._cache.get(customization_id)

    def delete(self, customization_id: uuid.UUID):
        # TODO remove from GPU RAM
        self._cache.pop(customization_id)
