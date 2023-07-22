# Model Interaction Code 

### Create virtual environment

- setup the virtual environment
    ```bash
    conda create -n msqa python=3.10
    conda activate msqa
    git clone https://github.com/ModelInteraction/MSQA.git
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    ```

### Process pretrain Azure documentation

- direct to `pretrain_azure_doc/` and run the below commandline to download the azure documentation for pretrain
    ```bash
    chmod +x clone_repos.sh
    ./clone_repos.sh
    ```
- extract and rename markdown files and save to `pretrain_azure_doc/data/`
    ```bash
    python save_azure.py
    ```
- split the markwdown files into json file limited with max token length for pretrain, save json file in to `pretrain_azure_doc/azure_json_output/`
    ```bash
    python process_azure.py
    ```

### Process MSQA data
Note we only show sample MSQA data and full data will be available when the paper is accepted.
- direct to `msqa_process/`
- post process the msqa data collected from [Microsoft Q&A forum](https://learn.microsoft.com/en-us/answers/)
    ```bash
    python post_process.py
    ```
- split and save to train and test json, they should be saved to `msqa_process/data/MSQA_train.json` and `msqa_process/data/MSQA_test.json`, respectively.
    ```bash
    python split.py
    ```

### Pretrain and finetune

- direct to `train/`
- pretrain with Azure documentation following the commandline with DeepSpeed
    ```bash
    deepspeed train.py \
    --model_name_or_path {YOUR_MODEL_PATH} \
    --data_path {AZURE_JSON_PATH} \
    --output_dir {PRETRAIN_MODEL_SAVE_PATH} \
    --num_train_epochs 8 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
    ```

    where {AZURE_JSON_PATH} is the path where you save processed azure documentation json `pretrain_azure_doc/azure_json_output/`

- finetune with MSQA train data previously saved in `msqa_process/data/MSQA_train.json`
    ```bash
    deepspeed train.py \
    --model_name_or_path {PRETRAIN_MODEL_SAVE_PATH} \
    --data_path {MSQA_TRAIN_JSON_PATH} \
    --output_dir {FINETUNE_MODEL_SAVE_PATH} \
    --num_train_epochs 5 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
    ```

### Inference with finetuned model
- generate domain knowledge with our finetuned model with the commandline
    ```bash
    python inference.py \
    --base_model= {FINETUNE_MODEL_SAVE_PATH} \
    --infer_ids_path= {QUESTION_ID_TO_INFERENCE} \
    --save_path= {RESULT_SAVE_PATH} \
    --batch_size=1 \
    --max_new_tokens=512 \
    --num_beams=4
    ```

### Result generation and evaluation
- Once the domain-specific model output its response to the question, we perform LLM generation taking either our domain knowledge or the chunks from retrieval-based methods.
- You should save your OAI key in the `keybook.py` and the endpoint function of LLM is in `llm_components.py`.
- Standard metrics, including BLEU, ROUGE-1/2/L, METEOR, BERT-Score, SIM, are defined in `eval_metrics.py`.
- Our proposed metrics
    - CAR is defined in `is_no_answer` in `eval_metrics.py`.
    - KHR is defined in `KHR.py` and keywords need to be extracted beforehand with `keyword_extract.py`.
    - LLM-based metrics is defined in `llm_eval.py`.
- `result_generation.py` contains all prompts to generate baseline results given either domain knowledge from our model or chunks from retrieval-based methods.
- `score_conflict.py` and `conflict_stat_plot.py` is to get the conflict analysis from LLM-based metric and visualization, respectively.

### Human evaluation UI
We also include the UI for human evaluators
- Direct to `ui/`
- Setup the python virtual environment
    ```bash
    conda create -n humaneval python=3.10
    conda activate humaneval
    pip install -r requirements.txt
    ```
- put the data to be evaluated in `ui/human_eval_data/`
- prepare the data
    ```bash
    python preprocess_human_eval_data.py
    ```
- Run the UI
    ```bash
    streamlit run qa_preference.py
    ```

### Human evaluation analysis
Direct to `human_annotataion/`
- put the `.csv` files of each human evaluator to `human_annotation/data/`
- process the human evaluation
    ```base
    python annotation_process.py
    ```
- output statistics and plot results in `annotation_stats.py`
