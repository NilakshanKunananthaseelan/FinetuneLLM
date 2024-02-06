from accelerate import FullyShardedDataParallelPlugin,Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig,FullStateDictConfig 

from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from transformers import TrainingArguments,Trainer,DataCollatorForLanguageModeling
from datetime import datetime
import torch
from datasets import load_dataset,load_from_disk
from functools import partial
from peft import (prepare_model_for_kbit_training, get_peft_model, get_peft_model_state_dict, LoraConfig, PeftModel)


def get_accelerator():

    fdsp_plugin  = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True,
                                                ),
    )

    accelerator = Accelerator(fsdp_plugin=fdsp_plugin)
    return accelerator

def get_dataset(name='Open-Orca/SlimOrca-Dedup'):
  

    dataset = load_dataset(name).shuffle(seed=42)
    train_eval = dataset['train'].train_test_split(test_size=0.2, seed=42,shuffle=False)
    test_eval = train_eval['test'].train_test_split(test_size=0.01, seed=42,shuffle=False)

    train_dataset = train_eval['train']
    test_dataset = test_eval['train']
    eval_dataset = test_eval['test']

    print(f'train_dataset: {len(train_dataset)}')
    print(f'test_dataset: {len(test_dataset)}')
    print(f'eval_dataset: {len(eval_dataset)}')
    return train_dataset,test_dataset,eval_dataset


def formatting_func(example,add_generation=False):
    template = ''
    
    for message in example['conversations']:
        if add_generation and message['from'] == 'gpt':
             continue
        #Remove 'Answer:' from the start of the message
        if message['from'] == 'human':
            message['from'] = 'user'
        if message['from'] == 'gpt':
            message['from'] = 'assistant'
        template += '<|im_start|>' + message['from'] + '\n' + message['value'] + '<|im_end|>' + '\n'
        
    
    if add_generation:
            template+='<|im_start|>assistant\n'
   

    return template


def get_base_model_tokenizer(model_name='mistralai/Mistral-7B-v0.1'):
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16

    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=bnb_config,
                                                device_map='auto'
                                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                padding_side='left',
                                                add_eos_token=True,
                                                )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    return model,tokenizer

def generate_and_tokenize_truncate_pad_prompt(prompt,tokenizer,add_generation=False,max_length=512):

    result = tokenizer(formatting_func(prompt,add_generation=add_generation), max_length=max_length, truncation=True, padding='max_length')

    result['labels'] = result['input_ids'].copy()

    return result

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

def setup_peft_model(model):
    lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",],
            bias='none',
            lora_dropout=0.05,
            task_type='CAUSAL_LM'
    )
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)  
    print("Base Model:")
    print_trainable_parameters(model)

    model = get_peft_model(model, lora_config)
    print("PEFT Model:")
    print_trainable_parameters(model)

    if torch.cuda.device_count() > 1:
        print('Devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        model.is_parallelizable = True
        model.model_parallel = True

    return model


def get_training_args(dataset_name='SlimOrca',base_name='Mistral-7B',project='PEFT'):
    run_name = f'{project}-{base_name}-{dataset_name}'
    output_dir = f'./outputs/{run_name}'
    print(f'output_dir: {output_dir}')
    print(f'run_name: {run_name}')

    args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=10,
        save_total_limit=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2.5e-5,
        logging_steps=100,
        logging_dir='./logs',
        bf16=True,
        optim='paged_adamw_8bit',
        save_strategy='steps',
        save_steps=100,
        evaluation_strategy='steps',
        eval_steps=100,
        do_eval=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        report_to='wandb',
        load_best_model_at_end=False)
    
    return args
    


if __name__ == '__main__':

    model_name = "mistralai/Mistral-7B-v0.1"
    dataset_name='SlimOrca'
    max_length = 512

    accelerator = get_accelerator()
    model,tokenizer = get_base_model_tokenizer(model_name=model_name)
    train_dataset,test_dataset,eval_dataset = get_dataset(name='Open-Orca/SlimOrca-Dedup')


    #load from disk
    try:
        tokenized_train_dataset = load_from_disk(f'../datasets/tokenized_datasets/{dataset_name}_train')
        tokenized_test_dataset = load_from_disk(f'../datasets/tokenized_datasets/{dataset_name}_test')
        tokenized_eval_dataset = load_from_disk(f'../datasets/tokenized_datasets/{dataset_name}_eval')
        print('Loaded tokenized datasets from disk')
        #sanity check
        sample = tokenized_train_dataset[5]
        print(tokenizer.decode(sample['input_ids']))
    except:
        tokenized_train_dataset = train_dataset.map(partial(generate_and_tokenize_truncate_pad_prompt,tokenizer=tokenizer,max_length=max_length))
        tokenized_test_dataset = test_dataset.map(partial(generate_and_tokenize_truncate_pad_prompt,tokenizer=tokenizer,max_length=max_length))
        tokenized_eval_dataset = eval_dataset.map(partial(generate_and_tokenize_truncate_pad_prompt,tokenizer=tokenizer,max_length=max_length))

        #save the tokenized datasets
        tokenized_train_dataset.save_to_disk(f'../datasets/tokenized_datasets/{dataset_name}_train')
        tokenized_test_dataset.save_to_disk(f'../datasets/tokenized_datasets/{dataset_name}_test')
        tokenized_eval_dataset.save_to_disk(f'../datasets/tokenized_datasets/{dataset_name}_eval')

        print('Tokenized datasets created and saved to disk')

    



    peft_model = setup_peft_model(model)
    acc_model = accelerator.prepare_model(peft_model)

    training_args = get_training_args(dataset_name='SlimOrca',base_name='Mistral-7B',project='PEFT')

    trainer = Trainer(
                model=acc_model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_eval_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    )

    acc_model.config.use_cache = False

    trainer.train()





