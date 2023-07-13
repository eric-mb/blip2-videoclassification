# %%
from inference import BLIP2
from video_decoder import VideoDecoder
import matplotlib.pyplot as plt

# %% Load model
model = BLIP2(model="Salesforce/instructblip-flan-t5-xl", processor="Salesforce/instructblip-flan-t5-xl")

# %% Decode video and select prompt
vd = VideoDecoder(path="../fakenarratives/temp/compacttv_2022_01_10_kX5DF1JDnqg.mp4", max_dimension=1920, fps=1)
vd_iter = iter(vd)
query = "Is the photo taken in a news studio?"

# %%
frame = next(vd_iter)
response = model.get_response(image=frame["frame"], query=query)

plt.imshow(frame["frame"])
print(f"Query: {query}")
print(f"Response: {response}")

# %%
