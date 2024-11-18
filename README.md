# Fine-tuning LLaMA-3 for Sentiment Analysis with QLoRA

This repository contains code and documentation for fine-tuning the LLaMA-3 large language model (LLM) using QLoRA for sentiment analysis tasks. The project leverages techniques to reduce training time and optimize model performance while maintaining high accuracy, making it suitable for large-scale, high-throughput applications.

## Project Overview

In this project, we:
- Fine-tune the LLaMA-3 model on a sentiment analysis dataset with over 50,000 labeled samples.
- Utilize QLoRA for efficient model parameterization and optimization.
- Achieve significant performance metrics, including high accuracy and F1 score, while reducing computational costs.

## Key Results

- **Accuracy**: 94%
- **F1 Score**: 0.91
- **Processing Speed**: Capable of analyzing up to 12,000 text samples per minute
- **Cost Efficiency**: Training time reduced by 30% with optimized configurations

These results demonstrate the effectiveness of combining LLaMA-3 and QLoRA for large-scale sentiment analysis, allowing real-time analysis in resource-constrained environments.

## Files and Structure

- `fine-tune-llama3-with-qlora-for-sentiment-analysis.ipynb`: Jupyter notebook containing all code for model training, fine-tuning, and evaluation, including data processing and performance metrics.
- `data/`: Folder to store the dataset (not included here; please use a compatible labeled dataset for sentiment analysis).
- `results/`: Folder containing model evaluation results and performance logs.

## Requirements

- Python 3.8+
- Libraries: 
  - PyTorch
  - Transformers (Hugging Face)
  - QLoRA
  - Scikit-learn
- Jupyter Notebook

Install the dependencies with:
```bash
pip install -r requirements.txt
```
## Fine tune Results

![Testing the model with fine-tuning](https://github.com/SwingRain/Fine-tune-Llama3-with-QLoRA-for-sentiment-analysis/blob/main/Testing%20the%20model%20without%20fine-tuning.png)

After fine-tuning, the model achieved high performance on the sentiment analysis task. Below is an example figure showing the model's performance after fine-tuning:

![Testing the model with fine-tuning](https://raw.githubusercontent.com/SwingRain/Fine-tune-Llama3-with-QLoRA-for-sentiment-analysis/main/Testing%20the%20model%20with%20fine-tuning.png)
