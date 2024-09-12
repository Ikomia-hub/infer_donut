<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_donut</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_donut">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_donut">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_donut/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_donut.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Donut :doughnut:, Document understanding transformer, is a new method of document understanding 
that utilizes an OCR-free end-to-end Transformer model. 
Donut does not require off-the-shelf OCR engines/APIs, yet it shows state-of-the-art performances 
on various visual document understanding tasks, such as visual document classification or 
information extraction (a.k.a. document parsing).

![ocr illustration](https://github.com/clovaai/donut/raw/master/misc/overview.png)



## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_donut", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-hub/infer_donut/blob/main/images/example.jpg?raw=true")

# Display results
extracted_data = algo.get_output(1)
print(extracted_data.data)
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'naver-clova-ix/donut-base-finetuned-docvqa': Name of the Donut pre-trained model for VGA. Other models available:
    - naver-clova-ix/donut-base-finetuned-rvlcdip
    - naver-clova-ix/donut-base-finetuned-cord-v1
    - naver-clova-ix/donut-base-finetuned-cord-v2
- **prompt** (str): question about document understanding for example.
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.
- **custom_model_folder**: custom model folder (optional)
- **task_name**: in case of custom model, you should specify the corresponding task

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_donut", auto_connect=True)

algo.set_parameters({
    "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
    "prompt": "What is the date of the document",
    "cuda": "True"
})

wf.run_on(url="https://github.com/Ikomia-hub/infer_donut/blob/main/images/example.jpg?raw=true")

# Display results
extracted_data = algo.get_output(1)
print(extracted_data.data)
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_donut", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-hub/infer_donut/blob/main/images/example.jpg?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
