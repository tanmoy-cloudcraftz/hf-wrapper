

def hf_create_repo(user:str, model_name:str):
    '''
    this will create a model under the huggingface user currently logged in
    '''

    from huggingface_hub import create_repo

    try:
        repo = f"{user}/{model_name}"
        create_repo(repo, private=False, exist_ok=True)

        return repo

    except Exception as ex:
        print(f'error while creating repo. {ex}')


def hf_get_tokenizer(tokenizer_id:str, tokenizer_type:str="auto"):
    '''
    '''
    from transformers import AutoTokenizer

    if tokenizer_type == "auto":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        tokenizer = None
        print(f'tokenizer_type={tokenizer_type} is not supported yet, so returning token as None')

    return tokenizer


def hf_prepare_data(datastet_id:str, tokenizer, sample_size:int=-1, data_source:str='HF'):
    '''
    pre
    '''
    from datasets import load_dataset




    if data_source == 'HF':
        data = load_dataset(datastet_id)
        if sample_size != -1:
            data = data['train'].select(range(sample_size))

        data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

    elif data_source == "GD":
        # TODO: implement data loading from google drive
        print('data loading from google drive is not supported yet')
    else:
        print(f'data_source={data_source} is not supported yet')

    return data

def hf_llm_train(model, tokenizer, data, save_model:bool=True, save_model_name:str=""):
    import transformers

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            output_dir=save_model_name,
            push_to_hub=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=5,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if save_model:
        # push the model to HF hub
        # trainer.push_to_hub()
        trainer.save_model(save_model_name)
        tokenizer.save_pretrained(save_model_name)
        # model.push_to_hub("test_model")

    return model