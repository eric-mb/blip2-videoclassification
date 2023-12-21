# %%
import imageio
from inference import BLIP2
from video_decoder import VideoDecoder
import matplotlib.pyplot as plt

# %% Load model
model = BLIP2(model="Salesforce/instructblip-flan-t5-xl", processor="Salesforce/instructblip-flan-t5-xl")

# %% Decode video and select prompt
vd = VideoDecoder(path="../fakenarratives/temp/compacttv_2022_01_10_kX5DF1JDnqg.mp4", max_dimension=1920, fps=1)
vd_iter = iter(vd)
query = "Is the photo taken in a news studio?"

# %% Get resonsonses for video frames
frame = next(vd_iter)["frame"]
response = model.get_response(image=frame, query=query)

plt.imshow(frame)
print(f"Query: {query}")
print(f"Response: {response}")

# %% Get resonsonses for an image

frame = imageio.imread("temp/screenshots_fakenarratives/compacttv_protest_anchor.png")[:, :, 0:3]
response = model.get_response(image=frame, query=query)

plt.imshow(frame)
print(f"Query: {query}")
print(f"Response: {response}")

# %%
