#!/usr/bin/env
import json

def generate():
    from vllm import LLM, SamplingParams
    import os
    vtokens = ''.join(["<VTok>"] * 10)
    f = "Context: After Washington had returned to Williamsburg, Dinwiddie ordered him to lead a larger force to assist Trent in his work. While en route, Washington learned of Trent's retreat. Since Tanaghrisson had promised support to the British, Washington continued toward Fort Duquesne and met with the Mingo leader. Learning of a French scouting party in the area, Washington, with Tanaghrisson and his party, surprised the Canadians on May 28 in what became known as the Battle of Jumonville Glen. They killed many of the Canadians, including their commanding officer, Joseph Coulon de Jumonville, whose head was reportedly split open by Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that Tanaghrisson was acting to gain the support of the British and regain authority over his own people. They had been inclined to support the French, with whom they had long trading relationships. One of Tanaghrisson's men told Contrecoeur that Jumonville had been killed by British musket fire. Question: Upon learning of a French scounting party in the area, what did Washington do? Answer:"
    f2 = "Context: In the early 1970s, ABC completed its transition to color; the decade as a whole would mark a turning point for ABC, as it began to pass CBS and NBC in the ratings to become the first place network. It also began to use behavioral and demographic data to better determine what types of sponsors to sell advertising slots to and provide programming that would appeal towards certain audiences. ABC's gains in audience share were greatly helped by the fact that several smaller markets had grown large enough to allow full-time affiliations from all three networks. Question: The 1970s allowed which network to move in to first place in the ratings? Answer:"
    f = vtokens + f
    f2 = vtokens + f2
    prompts=[f2, f]

    sampling_params = SamplingParams(temperature=0.0, top_k=-1)

    llm = LLM(model='/hub/nvgpt-2b-001/', ptuning_model_path="/hub/2b_ptuning_overfit_squad/ptuning_state_dict.ckpt", tensor_parallel_size=1)

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
