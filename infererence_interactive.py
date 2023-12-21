# %%
import imageio
from instructblip import InstructBLIP
from llava import Llava
from video_decoder import VideoDecoder
import matplotlib.pyplot as plt

# %% Variables
mllm = "Llava-1.5"  # choices: ["InstructBLIP", "Llava-1.5"]
prompt = "Is the photo taken in a news studio? Please answer with yes or no!"

# %% Load Model
if mllm == "InstructBLIP":
    print("Loading InstructBLIP")
    model = InstructBLIP(
        model="Salesforce/instructblip-flan-t5-xl",
        processor="Salesforce/instructblip-flan-t5-xl",
    )

if mllm == "Llava-1.5":
    print("Loading LLava-1.5")
    model = Llava(use_flash_attention_2=False)

# %% Decode video and select prompt
vd = VideoDecoder(
    path="../fakenarratives/temp/compacttv_2022_01_10_kX5DF1JDnqg.mp4",
    max_dimension=1920,
    fps=1,
)
vd_iter = iter(vd)


# %% Get resonsonses for video frames
frame = next(vd_iter)["frame"]
response = model.get_response(image=frame, prompt=prompt)

plt.imshow(frame)
print(f"Prompt: {prompt}")
print(f"Response: {response}")

# %% Get resonsonses for an image
frame = imageio.imread("temp/screenshots_fakenarratives/compacttv_protest_anchor.png")[
    :, :, 0:3
]
response = model.get_response(image=frame, prompt=prompt)

plt.imshow(frame)
print(f"Prompt: {prompt}")
print(f"Response: {response}")

# %%
