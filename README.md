# Concept Classification in Videos using BLIP-2

## Installation

To install all dependencies, please exectute the following commands:

~~~sh
conda create -n blip2_videoclassification_py3-11 python=3.11
conda activate blip2_videoclassification_py311
pip install -r requirements.txt
~~~

## Inference Script

To analyse a specific video for a given query please run:

~~~sh
python inference.py --video <PATH/TO/VIDEO.mp4> --query <YOUR QUERY> --output <PATH/TO/OUTPUT.pkl
~~~

## Interactive Mode

BLIP-2 takes a lot of time to initialize. To test out different prompt, we suggest to use a Jupyter notebook or VS Code in interactive mode. For the latter, please execute the following steps.

1. Open VS Code
2. Select Conda environment ```blip2_videoclassification_py311```
3. Run the cells in ```inference_interactive.py``` step by step.

In case you are connected to a GPU cluster, execute these additional steps.

4. Run the first cell with imports in ```inference_interactive.py```
5. Connect to the GPU cluster and activate the conda environment in this project folder
6. Open JupyterLab: ```jupyter-lab --ip localhost --port <PORT> --no-browser```
7. Copy the URL to the Jupyter server
8. Select the Jupyter service and corresponding ipykernel in the interactive shell of VS Code