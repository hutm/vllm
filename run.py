#!/usr/bin/env

def generate():
    from vllm import LLM, SamplingParams
    import os
    prompt = open("input.txt").read()
    prompts=[prompt]

    sampling_params = SamplingParams(temperature=0.0, top_k=-1)

    llm = LLM(model='/hub/nvgpt-2b-steerlm/', tensor_parallel_size=1)
    #llm = LLM(model='bigscience/bloomz-3b', tensor_parallel_size=4)
    os.system('clear')
    print('#'*100)
    print('#'*100)
    print('#'*100)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')

if __name__ == "__main__":
    generate()
