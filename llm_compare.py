from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
import openvino_genai as ov_genai


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ov_model_path = "TinyLlama-1.1B-Chat-v1.0"
    max_new_tokens = 32
    prompt = "table is made of"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.generation_config.max_length = None

    encoded_prompt = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    hf_encoded_output = model.generate(
        encoded_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])
    print(f"hf_output: {hf_output}")

    pipe = ov_genai.LLMPipeline(ov_model_path, "CPU")
    ov_encoded_output = pipe.generate(
        ov.Tensor(encoded_prompt.numpy()),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    ov_output = tokenizer.decode(ov_encoded_output.tokens[0])
    print(f"ov_output: {ov_output}")

    assert hf_output == ov_output


if __name__ == "__main__":
    main()
