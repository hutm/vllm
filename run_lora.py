#!/usr/bin/env
import asyncio
import json
import uuid

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.model_executor.adapters.customization_cache import CustomizationType

lora_base_path = '/hub/lora'
lora_id_1 = uuid.UUID('de05b696-711c-4d7c-983d-e1fb9c1a618a')
lora_id_2 = uuid.UUID('5857e9da-c7cb-4541-a419-fb364e6151a2')
loras = [
    (f'{lora_base_path}/2b_lora_weights.ckpt', lora_id_1,),
    (f'{lora_base_path}/lora_state_dict.ckpt', lora_id_2,),
]

def init_customizations(llm_engine: AsyncLLMEngine, checkpoint_id, blob):
    llm_engine.engine.add_customization_to_cache(
        customization_id=checkpoint_id,
        customization_type=CustomizationType.NVGPT_LORA,
        checkpoint=blob
    )

def read_loras(llm_engine: AsyncLLMEngine):
    for file, customization_id in loras:
        with open(file, mode="rb") as f:
            blob = f.read()
            init_customizations(llm_engine, customization_id, blob)

async def generate():
    from vllm import LLM, SamplingParams
    import os
    #prompt = open("input.hellaswag.txt").read().strip()
    f = "Context: After Washington had returned to Williamsburg, Dinwiddie ordered him to lead a larger force to assist Trent in his work. While en route, Washington learned of Trent's retreat. Since Tanaghrisson had promised support to the British, Washington continued toward Fort Duquesne and met with the Mingo leader. Learning of a French scouting party in the area, Washington, with Tanaghrisson and his party, surprised the Canadians on May 28 in what became known as the Battle of Jumonville Glen. They killed many of the Canadians, including their commanding officer, Joseph Coulon de Jumonville, whose head was reportedly split open by Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that Tanaghrisson was acting to gain the support of the British and regain authority over his own people. They had been inclined to support the French, with whom they had long trading relationships. One of Tanaghrisson's men told Contrecoeur that Jumonville had been killed by British musket fire. Question: Upon learning of a French scounting party in the area, what did Washington do? Answer:"
    f2 = "Context: Washington continued toward Fort Duquesne and met with the Mingo leader. Question: Upon learning of a French scounting party in the area, what did Washington do? Answer:"
    prompts=[f, f]
    tasks=["TASK2", None]
    sampling_params = SamplingParams(temperature=0.0, top_k=-1)

    tp = 1
    engine_args = AsyncEngineArgs(
        '/hub/nvgpt/nvgpt-2b-001/',
        tokenizer_mode='slow',
        dtype='bfloat16',
        worker_use_ray=tp > 1,
        tensor_parallel_size=tp,
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    read_loras(llm_engine)

    results = llm_engine.generate(prompt=f, sampling_params=sampling_params, request_id=uuid.uuid4().__str__(),
                                  customization_id=None)
    async for output in results:
        if output.finished:
            for completion in output.outputs:
                print(completion.text)

if __name__ == "__main__":
    asyncio.run(generate())
