#!/usr/bin/env python3
import click

import numpy as np

from datasets import load_dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm

from llm_experiments.utils.logutils import create_logger

NUM_PROC = 4


def write_tokenized_dataset(tokenized: DatasetDict, output: Path):
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = Path(output, f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = max(len(dset) // 512, 1)

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

def load_wikitext(
    output: Path,
    encoding: str,
    did: int = 2,
    eot_token: str="eot_token"
):
    output.mkdir(exist_ok=True, parents=True)
    _logger = create_logger("prep-wikitext")

    assert did in {2, 103}, "invalid dataset id"
    if encoding.startswith("gpt"):
        import tiktoken
        tokenizer = tiktoken.get_encoding(encoding)
        enc_function = tokenizer.encode_ordinary
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoding)
        enc_function = lambda x: tokenizer.encode(x, add_special_tokens=False)

    eot_token = getattr(tokenizer, eot_token)
    def process_(example):
        ids = enc_function(example["text"])
        ids.append(eot_token)
        out = {"ids": ids, "len": len(ids)}
        return out

    dataset = load_dataset('wikitext', name=f"wikitext-{did}-raw-v1", split='train', num_proc=NUM_PROC)
    # Remove empty strings and paragraph headings
    dataset = dataset.filter(lambda x: x["text"] != "" and not x["text"].startswith(" = "))

    # Perform the splits
    split_dataset = dataset.train_test_split(test_size=0.01, seed=1337, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")
    _logger.info(f"training set size = {len(split_dataset['train'])}")
    _logger.info(f"val set size = {len(split_dataset['val'])}")

    tokenized = split_dataset.map(
        process_,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=NUM_PROC
    )

    write_tokenized_dataset(tokenized, output)


@click.command()
@click.argument("output", type=Path)
@click.argument("encoding", type=str)
@click.option("-id", "--dataset-id", default=2, type=int)
def main(output: Path, encoding: str, dataset_id: int=2):
    load_wikitext(output, encoding, did=dataset_id)


if __name__ == "__main__":
    main()
