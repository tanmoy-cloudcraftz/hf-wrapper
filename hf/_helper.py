# colab specific configurations
env_type = 'colab'
env_name = 'colab_llm_env'
env_path = os.path.join('/content/drive/MyDrive/', env_name)
env_status_file = 'env.ok'


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


# TODO: option to remove columns when necessary
# TODO: streaming data option
def hf_prepare_data(datastet_id:str, tokenizer, sample_size:int=-1, data_source:str='HF', target_text_column:str='text'):
    '''
    pre
    '''
    from datasets import load_dataset


    if data_source == 'HF':
        data = load_dataset(datastet_id)
        if sample_size != -1:
            data = data['train'].select(range(sample_size))

        data = data.map(lambda samples: tokenizer(samples[target_text_column]), batched=True)

        # TODO: print token count
    elif data_source == "GD":
        # TODO: implement data loading from google drive
        print('data loading from google drive is not supported yet')
    else:
        print(f'data_source={data_source} is not supported yet')

    return data


# TODO: keyword arguement
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
        trainer.save_pretrained(save_model_name)
        tokenizer.save_pretrained(save_model_name)
        # model.push_to_hub("test_model")

    return model


def is_env_created_for_colab():
    import os

    if not os.path.exists(os.path.join(env_path, env_status_file)):
        print('virtual environment is not set properly, setting up')
        !pip install virtualenv
        # TODO: the name of the environment should not be fixed
        !virtualenv /content/drive/MyDrive/colab_llm_env
    else:
        print('virtual environment is set properly')    


def create_env_for_colab(env_type:str, env_name:str, activate_env:bool):
    '''
    creates an environment for transformer,peft specific implementation.
    uses the gdrive for logged in user
    '''
    import os, sys

    if env_type == 'colab':
        env_path = os.path.join("/content/drive/MyDrive/", env_name)
        env_site_package = os.path.join(env_path,"lib/python3.10/site-packages/")

        if not os.path.exists(os.path.join(env_path, env_status_file)):
            print('environment not found, creating...')

            # !virtualenv /content/drive/MyDrive/colab_llm_env

            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q -U bitsandbytes
            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q -U git+https://github.com/huggingface/transformers.git
            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q -U git+https://github.com/huggingface/peft.git
            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q -U git+https://github.com/huggingface/accelerate.git
            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q datasets
            !source /content/drive/MyDrive/colab_llm_env/bin/activate; pip install -q -U git+https://github.com/tanmoy-cloudcraftz/hf-wrapper.git

            env_state_file = os.path.join(env_path, env_status_file)
            with open(env_state_file, 'w') as stat_file:
                pass

        else:
            print('environment exists')

        if activate_env:
            print('activating environment')
            sys.path.append(env_site_package)
    else:
        print(f'envtype={env_type} is not supported yet')

