import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


class InstructBLIP:
    def __init__(
        self,
        model="Salesforce/instructblip-flan-t5-xl",
        processor="Salesforce/instructblip-flan-t5-xl",
        do_sample=False,
        num_beams=1,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
        precision=4,
    ):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision == 4:
            self._dtype = torch.bfloat16
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                model,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=self._dtype,
            )

        elif precision == 8:
            self._dtype = torch.bfloat16
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                model,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=self._dtype,
            )
        else:  # 32bit
            self._model = InstructBlipForConditionalGeneration.from_pretrained(model)
            self._model.to(self._device)

        self._processor = InstructBlipProcessor.from_pretrained(processor)

        self._params = {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "max_length": max_length,
            "min_length": min_length,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "temperature": temperature,
        }

    def get_response(self, image, prompt):
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device, dtype=self._dtype
        )
        outputs = self._model.generate(**inputs, **self._params)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
