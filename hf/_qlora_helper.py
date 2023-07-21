

def hf_prepare_model_for_qlora(model_id:str, model_type:str, lora_target_modules:list):
    '''
    returns the model with required config for QLora
    '''
    import torch
    from transformers import AutoModelForCausalLM
    from transformers  import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, get_peft_model

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if model_type == 'AutoModelForCausalLM':
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

        # peft based preparation
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # lora based preparation
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    else:
        print(f'model_type={model_type} is not defined, setting model as None')
        model = None


    return model

def hf_load_peft_model_tokenizer(peft_model_id:str):
    '''
    this method loads qlora specific model and corresponding tokenizer
    '''
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    # load peft specific model
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = model.to("cuda:0")
    model.eval()

    return model, tokenizer