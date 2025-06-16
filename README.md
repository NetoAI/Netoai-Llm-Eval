# LLM Response Testing and Benchmarking Framework

A modular Python framework for evaluating Large Language Models (LLMs) using another LLM as a judge. This framework generates responses to questions and evaluates them against ground truth answers, producing numerical scores.

## Features

* Modular architecture with separate inference and evaluation pipelines
* Customizable models, datasets, prompts, and output parameters
* Integration with Hugging Face `transformers` and `datasets`
* Flexible prompting with string or chat-style templates
* Judge-based numerical scoring using a second LLM
* JSON-based output for reproducibility and analysis

## Requirements

* Python 3.8+
* CUDA-compatible GPU (recommended)
* Hugging Face account and API token (for hosted models)

### Install Dependencies

```bash
   pip install torch transformers datasets langchain accelerate huggingface-hub
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## Setup

1. **Install dependencies**:

   ```bash
   pip install torch transformers datasets langchain accelerate huggingface-hub
   ```

2. **Set your Hugging Face token**:
   Replace `"hf_..."` in the notebook with your token

3. **Update the configuration dictionary** inside `llm_benchmarking.ipynb` to match your model and dataset

## Dataset Format

You can use any dataset from Hugging Face or a custom one as long as it has the following columns:

* `"question"`: The input query for the LLM
* `"answer"`: The ground truth answer for evaluation

During inference, results are stored in the following format:

```python
results.append({
    "question": item[question_column],
    "ground_truth": ground_truth,
    "llm_response": llm_response,
})
logging.info(f"Generated response for sample {i+1}.")
```

This format is also the required input for the judge model.

## Configuration

Configure the framework by modifying the `config` dictionary:

```python
config = {
    "model_to_be_tested": "Qwen/Qwen2-0.5B-Instruct",
    "dataset_hf_name": "squad",
    "dataset_split": "validation",
    "question_column_name": "question",
    "answer_column_name": "answers",
    "prompt_template": [
        {
            "role": "system",
            "content": "You are an expert assistant who provides concise and accurate answers."
        },
        {
            "role": "user",
            "content": "Please answer the following question: <question>"
        }
    ],
    "num_samples": 3,
    "max_new_tokens": 256,
    "output_file": "inference_results.json",
    "judge_model": "meta-llama/Llama-3-8b"
}
```

### Notes

* The `prompt_template` **must** contain a `<question>` placeholder.
* The dataset must have at least the `"question"` and `"answer"` columns.
* Judge models use `"question"`, `"ground_truth"`, and `"llm_response"` fields from the generated JSON.

## Usage

Run the framework through the notebook:

```bash
jupyter notebook llm_benchmarking.ipynb
```

## Framework Structure

### Inference Pipeline

* Loads LLM and tokenizer
* Reads the dataset and prepares prompts using `<question>` placeholder
* Generates responses using the target model
* Saves results to `inference_results.json`

### Evaluation Pipeline

* Loads judge model and tokenizer
* Reads `inference_results.json`
* Sends responses, ground truths, and questions to judge model
* Produces numeric scores (0.0–10.0)
* Outputs evaluation results to `final_scores.json`

## Output Files

* `inference_results.json`: Contains questions, ground truth answers, and generated responses
* `final_scores.json`: Contains all data plus numerical scores from the judge model

## Prompt Template Formats

**String Template Example**:

```python
"prompt_template": "Answer this question: <question>"
```

**Chat Format Example**:

```python
"prompt_template": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question: <question>"}
]
```

Make sure the `<question>` token exists in the user prompt.

## Troubleshooting

**Out of Memory**

* Reduce `num_samples`
* Decrease `max_new_tokens`
* Use smaller model

**Dataset/Column Errors**

* Check dataset column names: must include `question` and `answer`
* Validate Hugging Face dataset loading logic

**Model Loading Errors**

* Double-check model name spelling
* Confirm Hugging Face token is valid

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. Please open a pull request or create an issue.

---

