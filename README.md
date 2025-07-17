# DeepSeek-Medical-Model

Fine-tuning the [DeepSeek-R1](https://huggingface.co/dee/DeepSeek-R1-Distill-Llama-8B) model for better performance on **medical question answering and clinical reasoning** tasks.
This repo shows how to take a general-purpose LLM and adapt it using LoRA and a focused dataset to handle complex medical queries with more confidence and context.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Setup](#setup)
4. [Fine-Tuning the Model](#fine-tuning-the-model)
5. [Performance Boost: Before & After Fine-Tuning](#performance-boost-before--after-fine-tuning)
6. [Inference](#inference)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The **DeepSeek-Medical-Model** is built to answer medical questions with greater accuracy and confidence. By fine-tuning the DeepSeek-R1 model on a clinical dataset, we train it to understand complex medical contexts and provide more useful, structured answers.

After applying LoRA (Low-Rank Adaptation), the model becomes more efficient to train and much better at handling domain-specific tasks.

---

## Installation

To get started, install the required libraries:

```bash
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

Or just open the Colab notebook and run it directly.

---

## Setup

### Hugging Face Token

To access the model from Hugging Face, create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then log in:

```python
from huggingface_hub import login
hf_token = "your_token_here"
login(hf_token)
```

### GPU Check (for Colab)

Ensure your runtime supports CUDA for faster performance:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

---

## Fine-Tuning the Model

We use the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset which includes expert-written clinical questions, step-by-step reasoning (chain-of-thought), and answers.

We fine-tune using LoRA to update only small, efficient adapter layers.

Steps include:

* Preprocessing dataset with structured prompts
* Applying LoRA using `FastLanguageModel.get_peft_model()`
* Training using `SFTTrainer` and Hugging Face’s `TrainingArguments`

```python
finetune_dataset = medical_dataset.map(preprocess_input_data, batched=True)
trainer.train()
```

---

## Performance Boost: Before & After Fine-Tuning

### Before Fine-Tuning

Without medical fine-tuning, the model produces vague or general answers:

**Example:**

```text
Question:
A 61-year-old woman with stress incontinence... what would cystometry reveal?

Answer:
The cystometry test may show increased residual volume, with detrusor contractions potentially being weak.
```

This answer lacks detail and clinical reasoning.

---

### After Fine-Tuning

Post fine-tuning, the model is more confident and precise:

```text
Answer:
Cystometry is expected to show normal residual volume and normal detrusor activity. These findings are consistent with stress urinary incontinence, where the issue lies with urethral support, not detrusor overactivity or retention.
```

This is a much stronger, more helpful medical response.

---

## Inference

You can use the fine-tuned model like this:

```python
question = "A 59-year-old man presents with fever, chills, and vegetation on the aortic valve..."
inputs = tokenizer([prompt_template.format(question, "")], return_tensors="pt").to("cuda")

outputs = model_lora.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200
)

print(tokenizer.decode(outputs[0]))
```

---

## Testing

Test with real or challenging medical cases to evaluate performance. After fine-tuning, answers are:

* More specific
* Medically grounded
* Less likely to hallucinate

---

## Contributing

Pull requests are welcome. Contributions are encouraged if you'd like to:

* Improve prompt formatting
* Integrate a new dataset
* Add evaluation metrics
* Optimize training

---

## Troubleshooting

* **Hugging Face login not working:** Check your token’s permissions (should have `read` access).
* **No GPU in Colab:** Go to Runtime > Change runtime type > Select GPU.
* **WandB errors:** If not using Weights & Biases, you can skip those parts or comment out `wandb` calls.

---

## Credits

* **Model**: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/dee/DeepSeek-R1-Distill-Llama-8B)
* **Dataset**: [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
* **Training Framework**: [Unsloth](https://github.com/unslothai/unsloth)

---

## Final Notes

If you’re experimenting with LLMs in healthcare or medical QA, this project is a great example of how much domain-specific fine-tuning can improve model reliability. The pipeline is fast, lightweight, and focused on quality over quantity.
