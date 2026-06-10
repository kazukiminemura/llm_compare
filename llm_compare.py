from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
import openvino_genai as ov_genai


def main():
    model_id = "pfnet/plamo-3-nict-8b-base"
    ov_model_path = "plamo-3-nict-8b-base"
    max_new_tokens = 32
    prompt = "table is made of"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
    )
    model.generation_config.max_length = None

    encoded_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded_prompt = encoded_inputs.input_ids
    hf_encoded_output = model.generate(
        inputs=encoded_prompt,
        attention_mask=encoded_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])
    print(f"hf_output: {hf_output}")

    # pipe = ov_genai.LLMPipeline(ov_model_path, "CPU")
    # ov_encoded_output = pipe.generate(
    #     ov.Tensor(encoded_prompt.numpy()),
    #     max_new_tokens=max_new_tokens,
    #     do_sample=False,
    # )
    # ov_output = tokenizer.decode(ov_encoded_output.tokens[0])
    # print(f"ov_output: {ov_output}")

    # assert hf_output == ov_output


if __name__ == "__main__":
    main()
