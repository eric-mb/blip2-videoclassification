import argparse
import logging
import os
import pickle
import sys
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from video_decoder import VideoDecoder


class BLIP2:
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
    ):
        self._model = InstructBlipForConditionalGeneration.from_pretrained(model)
        self._processor = InstructBlipProcessor.from_pretrained(processor)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

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

    def get_response(self, image, query):
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self._device)
        outputs = self._model.generate(**inputs, **self._params)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--video", type=str, required=True, help="input video to process")
    parser.add_argument("-q", "--query", type=str, required=True, help="BLIP query")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file")

    # optional model parameters
    parser.add_argument("-m", "--model", type=str, default="Salesforce/instructblip-flan-t5-xl")
    parser.add_argument("-p", "--processor", type=str, default="Salesforce/instructblip-flan-t5-xl")

    # optional input parameters
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=1920, help="max dimension of the video frames"
    )
    parser.add_argument("--debug", action="store_true", help="debug output")
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load model
    model = BLIP2(model=args.model, processor=args.processor)

    # decode video
    vd = VideoDecoder(path=args.video, max_dimension=args.max_dimension, fps=args.fps)

    times = []
    responses = []
    for frame in vd:
        response = model.get_response(image=frame["frame"], query=args.query)
        responses.append(response)
        times.append(frame["time"])
        logging.debug(f"{frame['time']} s: {response}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"responses": responses, "times": times}, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
