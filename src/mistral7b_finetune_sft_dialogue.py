
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from trl import SFTTrainer
import torch

from datasets import load_dataset



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
   

    return {'text':template}

def get_dataset(name='Open-Orca/SlimOrca-Dedup'):
  

    dataset = load_dataset(name).shuffle(seed=42)

    subset= dataset['train'].train_test_split(test_size=0.965, seed=42)
    train_eval = subset['train'].train_test_split(test_size=0.2, seed=42)
    test_eval = train_eval['test'].train_test_split(test_size=0.02, seed=42)

    train_dataset = train_eval['train'].map(formatting_func,remove_columns=['conversations'])
    test_dataset = test_eval['train'].map(formatting_func,remove_columns=['conversations'])
    eval_dataset = test_eval['test'].map(formatting_func,remove_columns=['conversations'])

    print(f'train_dataset: {len(train_dataset)}')
    print(f'test_dataset: {len(test_dataset)}')
    print(f'eval_dataset: {len(eval_dataset)}')

    #Sanity check   
    print(train_dataset[0])
    print(test_dataset[0])
    print(eval_dataset[0])
    return train_dataset,test_dataset,eval_dataset





def get_base_model_tokenizer(model_name='mistralai/Mistral-7B-v0.1'):
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16

    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                load_in_4bit=True,
                                                quantization_config=bnb_config,
                                                torch_dtype=torch.bfloat16,
                                                
                                                device_map='auto',
                                                trust_remote_code=True,
                                                )

    model.config.use_cache=False 
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    print(tokenizer.add_bos_token, tokenizer.add_eos_token)


    
    return model,tokenizer


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
            lora_alpha=16,
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


    model = prepare_model_for_kbit_training(model)  
    print("Base Model:")
    print_trainable_parameters(model)

    model = get_peft_model(model, lora_config)
    print("PEFT Model:")
    print_trainable_parameters(model)

    return model,lora_config

from datetime import datetime

def get_training_args(dataset_name='SlimOrca',base_name='Mistral-7B',project='PEFT'):
    run_name =  f'{base_name}-{dataset_name}-{project}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    os.makedirs(f'../outputs/{base_name}-{dataset_name}-{project}',exist_ok=True)
    output_dir = f'../outputs/{base_name}-{dataset_name}-{project}'
    print(f'output_dir: {output_dir}')
    print(f'run_name: {run_name}')

    args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=10,
        save_total_limit=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=1,
        # max_steps=15000,
        learning_rate=2.5e-5,
        logging_steps=100,
        logging_dir=f'./logs/{base_name}-{dataset_name}-{project}',
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",

        optim='paged_adamw_8bit',
        save_strategy='steps',
        save_steps=100,
        evaluation_strategy='steps',
        eval_steps=100,
        do_eval=True,
        run_name=run_name,
        report_to='wandb',
        load_best_model_at_end=False,
        ddp_find_unused_parameters= False if torch.cuda.device_count() > 1 else None,)
    
    return args
    


if __name__ == '__main__':

    model_name = "mistralai/Mistral-7B-v0.1"
    dataset_name='SlimOrca'



    model,tokenizer = get_base_model_tokenizer(model_name=model_name)
    train_dataset,test_dataset,eval_dataset = get_dataset(name='Open-Orca/SlimOrca-Dedup')


    peft_model,peft_config = setup_peft_model(model)
    

    training_args = get_training_args(dataset_name='SlimOrca',base_name='Mistral-7B',project='PEFT_SFT')

    trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length= 1024,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing= False,
)
    wandb.login(key='6548ec4351addfce815f0c4faf3a4a37912efb07')
    run = wandb.init(project='Mistral7B-SlimOrca-PEFT-SFT', job_type="training", anonymous="allow",name=training_args.run_name)

    trainer.train()
    run.finish()
    trainer.save_model(f'../outputs/{training_args.run_name}')





