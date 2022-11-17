import os
from gc import callbacks
from lib2to3.pytree import convert 
import this
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer , TFGPT2LMHeadModel, AutoConfig
from huggingface_hub import notebook_login
from transformers import create_optimizer
import tensorflow as tf
from transformers import pipeline
import pandas as pd


dataset1 = load_dataset('csv', data_files=['amazon_flip_review.csv'])


#print(dataset1)
#print(dataset2)


class text_gen:
    def text_gen_flip(self,ds1):
        pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        sample1 = ds1["train"].shuffle(seed = 2).select(range(10))
        sample1_final = sample1.remove_columns('Sn')
        print(sample1_final)
        def length(example):  
            return{"Length":len(example["review"].split())}
        
        sample1_final = sample1_final.map(length)
        print(sample1_final[1])
        sample1_final.sort("Length")[:9]
        print(sample1_final.sort("Length")[:3])


        def remove_repeated(example):
            example["review"] = example["review"].replace('The','')
            example["review"] = example["review"].replace('a','')
            return {"review": example["review"].replace('It','')}
        
        sample1_final = sample1_final.map(remove_repeated)
        sample1_final[:9]
        print(sample1_final[:9])

        sample1_final = sample1_final.train_test_split(train_size = 0.9, seed = 9)
        sample1_final["validation"] = sample1_final.pop("test")
        print(sample1_final)
        for key in sample1_final["train"][0]:
            print(f"{key.upper()}.{sample1_final['train'][0][key]}")
        
        sample1_final.push_to_hub("Priyash/natural_language")
        
        context_length = 50

        

        def get_training_corpus():
            batch_size = 5
            return(
                sample1_final["train"][i: i+batch_size]["review"]
                for i in range(0,len(sample1_final["train"]),batch_size)

            )
        training_corpus = get_training_corpus()
        for reviews in training_corpus:
            print(len(reviews))
        
        vocab_size =  1000
        tokenizer=pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)
        print(tokenizer.eos_token_id)
        print(tokenizer.vocab_size)

        
      

        def tokenize(element):
            outputs = tokenizer(
                element["review"],
                max_length = context_length,
                truncation=True,
                return_overflowing_tokens = True,
                return_length = True,
            )
            input_batch =[]
            for length,input_ids in zip(outputs["length"],outputs["input_ids"]):
                if length == context_length:
                    input_batch.append(input_ids)
            return{"input_ids": input_batch}
        
            

        tokenized_datasets = sample1_final.map(
            tokenize, batched=True, remove_columns = sample1_final["train"].column_names
        )

        print(tokenized_datasets)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer,mlm = False, return_tensors = "tf")

        out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
        for key in out:
                print(f"{key} shape:{out[key].shape}")

        
        tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
            columns = ["input_ids","attention_mask","labels"],
            collate_fn = data_collator,
            shuffle = True,
            batch_size=32,
        )
        tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
            columns=["input_ids","attention_mask","labels"],
            collate_fn= data_collator,
            shuffle = False,
            batch_size = 32,
        )
        
        len(tf_train_dataset)
        len(tf_eval_dataset) 

        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size = len(tokenizer),
            n_ctx = context_length,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )
        model = TFGPT2LMHeadModel(config)
        model(model.dummy_inputs)
        model.summary()

        

        num_train_steps = len(tf_train_dataset)
        optimizer, schedule = create_optimizer(
            init_lr = 5e-5,
            num_warmup_steps= 1_000,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
        )
        model.compile(optimizer=optimizer)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model.fit(tf_train_dataset,validation_data=tf_eval_dataset,epochs=1)
        
        dataset_samp = load_dataset('csv',data_files=['amazon_fold_review.csv'])
        print(dataset_samp)
        review_model = TFGPT2LMHeadModel.from_pretrained('gpt2')
        review_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        pipe = pipeline(
            "text-generation", model = review_model, tokenizer = review_tokenizer, device = 0
        )
        #data = dataset_samp['review'].tolist()
        prompts = ['Camera quality ', 'Performance Of Samsung z flip' , 'Performance of Samsung Z fold','Affordability Of Samsung']
        

        
        output1 = pipe(prompts,num_return_sequences=1)[0][0]["generated_text"]
        print("For prompts",prompts[0],"Text is: ")
        print(output1)

obj1 = text_gen() 
obj1.text_gen_flip(dataset1)
