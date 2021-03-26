# Finetune GPT2-XL (1.5 Billion Parameters) on a single 16 GB VRAM V100 Google Cloud instance.

- Easy to run, modify and integrate: Uses the default language modeling script of Huggingface Transformers: run_clm.py with just 2 lines of code added
- Explains how to setup a preemptible V100 16GB VRAM GPU server with 78 GB CPU RAM on Google Compute Engine. At the time of writing, this configuration only costs about $1.28 / hour in GCE.
- Uses the Zero Optimizer (stage 2) and Zero-Offload from the Deepspeed library (with the Huggingface integration), which offloads data to RAM and therefore reduces necessary GPU VRAM (this is why you need a server with lots of RAM). It also uses gradient checkpointing  which further decreases GPU memory usage by trading it off with compute. Also uses FP16.

## 1. (Optional) Setup VM with V100 in Google Compute Engine

### Requirements (for people who have never used GCE):

1. Install the Google Cloud SDK: [Click Here](https://cloud.google.com/sdk/docs/install)
2. Register a Google Cloud Account, create a project and set up billing.
3. Request a quota limit increase for "GPU All Regions" to 1.
4. Log in and initialize the cloud sdk with `gcloud auth login` and `gcloud init` .

### Create VM

- Replace PROJECTID in the command below with the project id from your GCE project.
- You can remove the preemptible flag, if you want to be sure that google doesn't randomly shut down your instance, but then you will pay about 3x more.
- We need a GPU server with at least 70 GB RAM, otherwise the run will crash, whenever the script wants to pickle a model. This setup below gives us as much RAM as possible with 12 cpus in GCE. You also can't use more than 12 CPUs with a single V100 GPU in GCE.

Run this to create the instance:

```markdown
gcloud compute instances create gpuserver \
   --project PROJECTID \
   --zone us-central1-a \
   --custom-cpu 12 \
   --custom-memory 78 \
   --maintenance-policy TERMINATE \
   --image-family pytorch-latest-gpu \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --metadata "install-nvidia-driver=True" \
   --accelerator="type=nvidia-tesla-v100,count=1" \
   --preemptible
```

After 5 minutes or so (the server needs to install nvidia drivers first), you can connect to your instance (replace ACCOUNTNAME with your sdk accountname): 

```markdown
gcloud compute ssh ACCOUNTNAME@gpuserver
```

Don't forget to shut down the server once your done, otherwise you will keep getting billed for it.

The next time you can restart the server from the web ui [here](https://console.cloud.google.com/compute/instance).

## 2. Download script and install libraries

Run this to download the script and to install all libraries:

```markdown
git clone https://github.com/Xirider/finetune-gpt2xl.git
chmod -R 777 finetune-gpt2xl/
cd finetune-gpt2xl
pip install -r requirements.txt 

```

- This installs transformers from source, as the current release doesn't work well with deepspeed.

(Optional) If you want to use [Wandb.ai](http://wandb.ai) for experiment tracking, you have to login:

```markdown
wandb login
```

## 3. Finetune GPT2-xl

- replace the example train.txt and validation.txt files in the folder with your own training data and then run `python texttocsv.py`. This converts your .txt files into one column csv files with a "text" header and puts all the text into a single line. We need to use .csv files instead of .txt files, because Huggingface's dataloader removes line breaks when loading text from a .txt file, which does not happen with the .csv files.
- If you want to feed the model separate examples instead of one continuous block of text, you need to pack each of your examples into an separate line in the csv train and validation files.
- Be careful with the encoding of your text. If you don't clean your text files or if just copy text from the web into a text editor, the dataloader from the datasets library might not load them.

Run this:

```markdown
deepspeed --num_gpus=1 run_clm.py \
--deepspeed ds_config.json \
--model_name_or_path gpt2-xl \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--save_steps 200 \
--eval_steps 200 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8

```

- This command runs the the standard run_clm.py file from Huggingface's examples with deepspeed, just with 2 lines added to enable gradient checkpointing to use less memory.
- Training on the Shakespeare example should take about 17 minutes. With gradient accumulation 4 and batch size 4, one gradient step takes about 9 seconds. This means the model training speed should be almost 2 examples / second. You can go up to batch size of 12 before running out of memory, but that doesn't provide any speedups.

## 4. Generate text with your finetuned model

You can test your finetuned model with this script from Huggingface Transfomers (is included in the folder):

```markdown
python run_generation.py --model_type=gpt2 --model_name_or_path=finetuned --length 200
```

Or you can use it now in your own code like this:

```python
# credit to Niels Rogge - https://github.com/huggingface/transformers/issues/10704

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
print("model loaded")

# this is a single input batch
texts = ["From off a hill whose concave womb", "Another try", "A third test"]

encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding, max_length=100)
generated_texts = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True)

print(generated_texts)
```

- model inference runs on even small gpus or on cpus without any more additional changes

## (Optional) Configuration

You can change the learning rate, weight decay and warmup in the deepspeed ds_config.json file. It uses the default settings, except for a reduced allgather_bucket_size and reduced reduce_bucket_size, to save even more gpu memory. You can check all the explanations here:

[https://huggingface.co/transformers/master/main_classes/trainer.html#deepspeed](https://huggingface.co/transformers/master/main_classes/trainer.html#deepspeed)

The rest of the training arguments can be provided as a flag and are all listed here:

[https://huggingface.co/transformers/master/main_classes/trainer.html#trainingarguments](https://huggingface.co/transformers/master/main_classes/trainer.html#trainingarguments)