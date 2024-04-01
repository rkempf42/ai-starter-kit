<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>


YoDa
======================

<!-- TOC -->

- [Overview](#overview)
- [Workflow](#workflow)
    - [Data generation](#data-generation)
    - [Data preparation](#data-preparation)
    - [Training and finetuning](#training-and-finetuning)
    - [Evaluation](#evaluation)
- [Getting started](#getting-started)
    - [Deploy your models in SambaStudio](#deploy-your-models-in-sambastudio)
    - [Get your SambaStudio API key](#get-your-sambastudio-api-key)
    - [Set the starter kit environment](#set-up-the-starter-kit-environment)
- [Starterkit usage](#starter-kit-usage)
    - [Data Generation](#data-generation)
        - [To Generate pretraining data](#to-generate-pretraining-data)
        - [To generate finetuning data](#to-generate-finetuning-data)
        - [Both pretraining and fine-tuning data generation](#to-generate-both-pretraining-and-fine-tuning-data)
    - [Data Preprocessing](#data-preprocessing)
    - [Launch pretraining/finetuning and host endpoints on SambaStudio](#launch-pretrainingfinetuning-and-host-endpoints-on-sambastudio)
    - [Evaluation](#evaluation)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Overview

YoDa is an acronym for **Your Data, Your Model**. This project aims to train a Large Language Model (LLM) using a customer's private data. The goal is to compete with general solutions on tasks that are related to the customer's data.

# Workflow 

## Data generation

This phase involves the generation of synthetic data relevant to the customer's domain. The main data generation methods may vary depending on the task requirements and include:

* *Pretraining Generation*: Generates a JSONL file containing sections of the provided data. Enables the model to do completion over queries.

* *Finetuning Generation*: Utilizing a powerful LLM `Llama 2 70B` and a pipeline composed of prompting and postprocessing techniques, this step processes each document to create a series of synthetic questions and answers based on the content. The generated data is stored in JSONL files. Finetuning generation teaches the model to follow instructions and answer questions (and not just do completion).

## Data preparation

Data preparation involves preprocessing and formatting the generated data to make it suitable for training. This step transforms the data into the required format and structure necessary for training the large language model on a SambaNova system.

## Training and finetuning

In this stage, the large language model is fine-tuned in SambaStudio using your data. Finetuning includes updating the model's parameters to adapt it to the specific characteristics and patterns present in the prepared dataset.

## Evaluation

The evaluation phase creates a set of responses to assess the performance of the fine-tuned language model on relevant queries. 

Evaluation involves using the set of evaluation queries for:

- Obtaining responses from a baseline model.
- Obtaining responses from your custom model.
- Obtaining responses from your custom model and giving them in the exact context used in question generation of the evaluation queries.
- Obtaining responses from your custom model and employing a simple RAG pipeline for response generation.

Evaluation facilitates performing further analysis of your model's effectiveness in solving the domain-specific tasks.

# Getting started

These instructions illustrate how to generate training data, preprocess the data, train the model, launch the online inference service, and evaluate the results.

## Deploy your models in SambaStudio

1. First deploy a powerful LLM (e.g. Llama 2 70B chat) to an endpoint for inference in SambaStudio either through the GUI or CLI. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

2. Then deploy your baseline model (e.g. Llama 2 7B) to an endpoint for inference in SambaStudio either through the GUI or CLI.

## Get your SambaStudio API key

In this starter kit you can optionally use the SambaNova SDK `SKSDK` to run training and inference jobs in SambaStudio. You only need to set the API Authorization Key. The Authorization Key will be used to access to the API Resources on SambaStudio. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).

## Set up the starter kit environment

1. Clone the repo.
    ```bash
    git clone https://github.com/sambanova/ai-starter-kit.git
    ```


2. Update API information for the SambaNova LLM and your environment [sambastudio key](#get-your-sambastudio-api-key). 
    
These are represented as configurable variables in the environment variables file in the root repo directory **```sn-ai-starter-kit/.env```**. For example, assume you want to specify three things in the environment file:
   
* a Llama70B chat endpoint with the URL
    `https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123`
* a Lama7B baseline model with the URL 
    `https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/987654ef-fedc-9876-1234-01fedbac9876`
* A SambaStudio key `1234567890abcdef987654321fedcba0123456789abcdef`.

You enter those items in the environment file (with no spaces) as follows:
```yaml
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"

    YODA_BASE_URL="https://api-stage.sambanova.net"
    YODA_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    BASELINE_ENDPOINT_ID="987654ef-fedc-9876-1234-01fedbac9876"
    BASELINE_API_KEY="12fedcba-9876-1234-abcd76543"

    SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
```

3. Install requirements.
   Use a `virtualenv` or `conda` environment for installation, and update with `pip install`, for example:
```bash
    cd ai-starter-kit/yoda
    python3 -m venv yoda_env
    source/yoda_env/bin/activate
    pip install -r requirements.txt
```
5. Download your dataset and update the `src_folder` variable in your [sn expert config file](./sn_expert_conf.yaml), with the path of the folder and sub-folders in `src_subfolders`. Follow the same step for including your own data.

6. Optionally download and install SambaNova SNSDK. Follow the instructions in this [guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) for installing SambaNova SNSDK and SNAPI. You can omit the *Create a virtual environment* step because you are using the ```yoda_env``` environment you just created.

7. Download the [SamabaNova data preparation repository](https://github.com/sambanova/generative_data_prep)
   ```bash
    deactivate
    cd ../..
    git clone https://github.com/sambanova/generative_data_prep
    cd generative_data_prep
    python3 -m venv generative_data_prep_env
    source/generative_data_prep_env/bin/activate
   ```
8. Follow the [installation guide](https://github.com/sambanova/generative_data_prep?tab=readme-ov-file#installation)

# Starter kit usage

## Before you begin

1. Request access to the Meta Llama2 tokenizer and have a [local copy](https://llama.meta.com/llama-downloads/) or [Hugging Face model granted access](https://huggingface.co/meta-llama/Llama-2-70b).
2. Put the path of the tokenizer or name of the Hugging Face model in the config file.
3. Ensure that you have a SambaStudio endpoint to the LLAMA 70B Chat model and add the configurations to your env file, which is used for synthetic data generation.

## Data generation 

For domain-adaptive pre-training and data generation run one of the following scripts:

1. Replace the value of the `--config` parameter with your actual config file path. An example config is shown in `./sn_expert_conf.yaml` (this is set as the default parameter in the data generation scripts below).
2. In your config file, set the `dest_folder`, `tokenizer` and `n_eval_samples` parameters.
3. Activate your YoDa starter kit environment.

```bash
deactivate
cd ../..
cd ai-starter-kit/yoda
source/yoda_env/bin/activate
```

### To generate pretraining data
```bash
python -m src/gen_data.py
    --config ./sn_expert_conf.yaml
    --purpose pretrain 
```

### To generate finetuning data

```bash
python src/gen_data.py
    --config ./sn_expert_conf.yaml
    --purpose finetune 
```

### To generate both pretraining and fine tuning data
```bash
python -m src.gen_data
    --config ./sn_expert_conf.yaml
    --purpose both 
```

## Data Preprocessing

For pretraining and finetuning on SambaStudio, the data must be hdf5 files that you can upload as dataset to SambaStudio.
To preprocess the data:
1. Open `scripts/preprocess.sh`.
2. Replace the variables `ROOT_GEN_DATA_PREP_DIR` with the path to your [generative data preparation](https://github.com/sambanova/generative_data_prep)
directory
3. Set the absolute path of the output JSONL from [pretraining/finetuning](#data-generation-1)
   * in the `INPUT_FILE` parameter of `scripts/preprocess.sh`.
   * in the `OUTPUT_DIR` where you want your hdf5 files to be dumped before you upload them to SambaStudio Datasets.
4. Activate the generative_data_prep_env:
```bash
   deactivate
   source ../../generative_data_prep_env/bin/activate
```
5. Run the script

```bash
   sh scripts/preprocess.sh
```

## Launch pretraining/finetuning and host endpoints on SambaStudio

On SambaStudio, you can create and host your model checkpoints. 
This can be done on the [**SambaStudio GUI**](https://docs.sambanova.ai/sambastudio/latest/dashboard.html) following the next steps:

1. Upload your generated dataset from [gen_data_prep](#data-preparation) step.

2. Create a [project](https://docs.sambanova.ai/sambastudio/latest/projects.html).

3. Run a [training job](https://docs.sambanova.ai/sambastudio/latest/training.html).

4. [Create an endpoint](https://docs.sambanova.ai/sambastudio/latest/endpoints.html) for your trained model.

5. Add the endpoint details to the `.env` file. Now your `.env` file should look like this:
```yaml
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"

    YODA_BASE_URL="https://api-stage.sambanova.net"
    YODA_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    BASELINE_ENDPOINT_ID="987654ef-fedc-9876-1234-01fedbac9876"
    BASELINE_API_KEY="12fedcba-9876-1234-abcd76543"

    #finetuned model endpoint details
    FINETUNED_ENDPOINT_ID="your endpoint ID"
    FINETUNED_API_KEY="your endpoint API key"

    SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
```

This training process can also be done with **snapapi** and **snapsdk**. 
<!-- NOT PUBLIC
If you are 
interested in how this done via **SNSDK**, please have a look at the WIP [notebook](./notebooks/SambaStudio_job_spinup.ipynb) using the yoda env.
-->

## Evaluation

For evaluation, you send questions from the held-out synthetic question-answer pairs that you procured
when you were generating the fine tuning data to the fine-tuned model. You can benchmark the approach against responses you get from also using RAG as well as from a golden context.

1. Reactivate the YoDa env.

```bash
deactivate 
source yoda_env/bin/activate
```
2. To assess the trained model, execute the following script (replace  `--config` parameter with your actual config file path):
```bash
python src/evaluate.py 
    --config sn_expert_conf.yaml
```

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- scikit-learn  (version 1.4.1.post1)
- jsonlines  (version 4.0.0)
- transformers (version4.33)
- wordcloud  (version 1.9.3)
- sacrebleu  (version 2.4.0)
- datasets  (version 2.18.0)
- sqlitedict  (version 2.1.0)
- accelerate  (version 0.27.2)
- omegaconf  (version 2.3.0)
- evaluate  (version 0.4.1)
- pycountry  (version 23.12.11)
- rouge_score  (version 0.1.2)
- parallelformers  (version 1.2.7)
- peft  (version 0.9.0)
- plotly (version 5.18.0)
- langchain (version 0.1.2)
- pydantic (version1.10.13)
- python-dotenv (version 1.0.0)
- sseclient (version 0.0.27)
