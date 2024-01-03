"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import sys
import gzip
import csv
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from datasets import load_dataset, Dataset, DatasetDict
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class MyTrainingArguments:
    model_name_or_path: Optional[str] = field(default=None)
    train_batch_size: Optional[int] = field(default=1)
    num_epochs: Optional[int] = field(default=5)
    model_save_path: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    checkpoint_save_steps: Optional[int] = field(default=1000),
    checkpoint_save_total_limit: Optional[int] = field(default=5)
    evaluation_steps: Optional[int] = field(default=0)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
#### /print debug information to stdout

if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
        # import pdb; pdb.set_trace()
        
    
    # Read the dataset and convert the dataset to a DataLoader ready for training
    assert "json" in args.train_file.split(".")[-1], "Train file must be in json/jsonl format."
    train_dataset = load_dataset("json", data_files=args.train_file, split="train").to_list()
    """
        MultipleNegativesRankingLoss expects as input a batch consisting of sentence pairs 
        (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
    """
    train_samples = []
    for example in train_dataset:
        query = example["query"][1]
        pos = example["pos"][1]
        neg = example["neg"][1]
        inp_example = InputExample(texts=[query, pos, neg])
        train_samples.append(inp_example)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    
    # Load a pre-trained sentence transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model_name_or_path, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    logging.info(f"Train on device: {model.device}")
    # train_loss = losses.CosineSimilarityLoss(model=model)
    
    
    # Development set: Measure correlation between cosine score and gold labels
    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    evaluator = None
    
    
    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=warmup_steps,
        epochs=args.num_epochs,
        evaluation_steps=args.evaluation_steps,
        output_path=args.model_save_path,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_steps
    )
