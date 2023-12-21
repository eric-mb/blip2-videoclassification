import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class Llava:
    def __init__(
        self,
        model_id="llava-hf/llava-1.5-7b-hf",
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        use_flash_attention_2=True,
    ):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_4bit=load_in_4bit,
            use_flash_attention_2=use_flash_attention_2,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_response(
        self,
        prompt,
        image,
        max_new_tokens=200,
        do_sample=False,
    ):
        prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(prompt, image, return_tensors="pt").to(0, torch.float16)
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )

        generated_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return generated_text.split("ASSISTANT:")[-1].strip()
