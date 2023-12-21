import argparse
import logging
import os
import pickle
import sys
from video_decoder import VideoDecoder

from instructblip import InstructBLIP
from llava import Llava


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-v", "--video", type=str, required=True, help="input video to process"
    )
    parser.add_argument("-q", "--query", type=str, required=True, help="query")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file")

    # mllm
    parser.add_argument(
        "-m",
        "--model",
        type="str",
        choices=["instructblip", "llava-1.5"],
        required=True,
        help="mllm model to use",
    )

    # optional intructblip parameters
    parser.add_argument(
        "-ibm",
        "--instructblip_model",
        type=str,
        default="Salesforce/instructblip-flan-t5-xl",
    )
    parser.add_argument(
        "-ibp",
        "--instructblip_processor",
        type=str,
        default="Salesforce/instructblip-flan-t5-xl",
    )

    # optional input parameters
    parser.add_argument(
        "--fps", type=int, required=False, default=2, help="fps to process video"
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1920,
        help="max dimension of the video frames",
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

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # load model
    model = None
    if args.model == "instructblip":
        model = InstructBLIP(
            model=args.instructblip_model, processor=args.instructblip_processor
        )

    if args.model == "llava-1.5":
        model = Llava()

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
