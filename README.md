# InspireDebate

## 📰 News
- 🎉  (2025-05-15) InspireDebate is accepted to ACL 2025 Main Conference!

## 👋 InspireDebate Framework

We propose an innovative intelligent debating system that addresses the limitations of existing LLM-based debating systems through two core components:


1. **InspireScore**: Multi-dimensional Evaluation System

   - Subjective Assessment Dimensions:
     - Emotional Appeal
     - Argument Clarity
     - Argument Arrangement
     - Topic Relevance
   - Objective Assessment Dimensions:
     - Fact Authenticity
     - Logical Validity
2. **InspireDebate**: Optimized Debating Framework

   - Phased Optimization Approach
   - Chain-of-Thought (CoT) Reasoning Enhancement (SFT)
   - Multi-dimensional Direct Preference Optimization (DPO)
   - Real-time Knowledge Grounding via Web-based Retrieval Augmented Generation (Web-RAG)

<div align="center">
    <img width="95%" alt="MAD" src="README/img/framework.png" />
</div>


## 📂 Project Structure

```
InspireDebate/
├── debating.py              # Main debating prcessing implementation
├── inspirescore.py         # inspirescore system implementation
...
```

## 🏃‍♂️‍➡️ Quick Start

###  💻 Requirements

- Python 3.9+
- PyTorch 2.0.0+
- CUDA 11.6+ (for GPU version)

### 🔑 Installation

1. Clone the repository

```bash
git clone https://github.com/fywang12/InspireDebate.git
cd InspireDebate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

### 🔧 Usage

#### Starting your own Debate Processing

```shell
python debating.py
```

#### InspireScore for Debate Dialogue Evaluation System

```python
python inspirescore.py
```

#### How to Perform SFT and DPO with Open-source Models

##### 1. Install llama-factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```
##### 2. Supervised Fine-tuning (SFT)

First, format your debate dataset as follows:

```json
[
    {
        "instruction": "",
        "input": "",
        "output": "",
        "system": ""
    }
]

```
You can access our SFT datasets on [🤗 Hugging Face](https://huggingface.co/datasets/fywang12/InspireDebate)

To start SFT training with open-source models such as LLaMA-3, simply run:

```bash
llamafactory-cli train examples/train_lora/InspireDebate_sft.yaml
```

##### 3. Direct Preference Optimization (DPO)

DPO preference pairs are selected by comparing InspireScore ratings, with the higher-scoring response as “chosen” and the lower-scoring one as “rejected”.

Json formatted:
```json
[
   {
       "chosen": "Preferred response",
       "rejected": "Less preferred response",
       "prompt": ""
   }
]
```
To perform DPO training with models like LLaMA-3, use:
```bash
llamafactory-cli train examples/train_lora/InspireDebate_dpo.yaml
```

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:

```bibtex
@inproceedings{
    wang2025inspiredebate,
    title={InspireDebate: Multi-Dimensional Subjective-Objective Evaluation-Guided Reasoning and Optimization for Debating},
    author={Fuyu Wang and Jiangtong li and Kun Zhu and Changjun Jiang},
    booktitle={ACL 2025 (Main)},
    year={2025}
}
```

## 📄 Acknowledgement

Portions of our codebase and prompt design are adapted from open-source repositories. 
Special thanks to [MAD](https://github.com/Skytliang/Multi-Agents-Debate) and [llamafactory](https://github.com/hiyouga/LLaMA-Factory) for their invaluable contributions.

##  License
This project is licensed under the GNU 3.0 License - see the [LICENSE](LICENSE) file for details
