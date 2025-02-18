# Massive-activations-Vlms

![image](https://github.com/zhangbaijin/Massive-activations-VLMs/blob/main/massive.png)


## Setup
```bash
conda create -n fastv python=3.10
conda activate fastv
cd src
bash setup.sh
```



## Visualization: Inefficient Attention over Visual Tokens 

we provide a script (./src/FastV/inference/visualization.sh) to reproduce the visualization result of each LLaVA model layer for a given image and prompt.

```bash
bash ./src/FastV/inference/visualization.sh
```
or
```bash
python ./src/FastV/inference/plot_inefficient_attention_massive.py \
    --model-path "PATH-to-HF-LLaVA1.5-Checkpoints" \
    --image-path "./src/LLaVA/images/llava_logo.png" \
    --prompt "Describe the image in details."\
    --output-path "./output_example"\
```
it will obtain a json file contain massive activation weights.

## Visualization
```
python plt_massive.py  
```
