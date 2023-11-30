"""
Compute language probability per token.

Based on https://huggingface.co/docs/transformers/perplexity

Why? Want to create language model scores for each token in
a text to see if there's a difference in distributional patterns
between AI generated text and human text.

What? https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview
"""
import click
from pathlib import Path

from tqdm import tqdm
from pandas import read_csv
from torch.cuda import is_available as cuda_is_available
from torch import no_grad, zeros, gather, log_softmax, Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerplexityCalculator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, context_size: int = None):
        self.device = 'cuda' if cuda_is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tok = tokenizer
        if context_size:
            self.context_size = context_size
        elif hasattr(self.model.config, "n_positions"):
            self.context_size = self.model.config.n_positions
        else:
            raise ValueError("cannot determine context size from config, please specify")

    def compute_token_probabilities(self, text: str, stride: int = 256) -> Tensor:
        """
        Compute the logits for each token, using a sliding window for the
        case where the text is longer than the context window. For each word,
        we use the prediction that was made with the maximum context window.
        """
        # Note: tokenization of sequences longer than allowed length may
        # result in warning
        encodings = self.tok(text, return_tensors="pt")
        seq_len = encodings.input_ids.shape[1]
        logits = zeros(seq_len)

        logit_idx = 1 # no logit for first token
        for window_start in range(0, seq_len, stride):
            window_end = min(seq_len, window_start + self.context_size)
            input_ids = encodings.input_ids[:, window_start:window_end].to(self.device)
            with no_grad():
                outputs = self.model(input_ids)

            # Logits are those for next token, so we drop the last
            _logp = log_softmax(outputs.logits, dim=-1)[:, :-1, :]
            _indices = input_ids[:, 1:].unsqueeze(-1)
            _logp0 = gather(_logp, index=_indices, dim=-1).squeeze()

            # While striding, we always want to take the scores for the
            # last n tokens, such that any earlier tokens we might have
            # predicted maintain the scores computed with the maximal context
            n_new_tok = window_end - logit_idx
            logits[logit_idx:window_end] = _logp0[-n_new_tok:]
            logit_idx = window_end

            if logit_idx == seq_len:
                break

        return logits[1:]

@click.command()
@click.option("--model", type=str)
@click.option("--texts", type=Path)
@click.option("--output", type=Path)
def main(model: str, texts: Path, output: Path):
    _model = AutoModelForCausalLM.from_pretrained(model)
    _tokenizer = AutoTokenizer.from_pretrained(model)
    calculator = PerplexityCalculator(_model, _tokenizer)

    if texts.name.endswith(".csv"):
        from numpy import savez
        results = dict()
        df = read_csv(texts)
        for _idx, _row in tqdm(df.iterrows(), total=df.shape[0]):
            results[str(_idx)] = calculator.compute_token_probabilities(
                _row.text
            ).numpy()
        savez(output, **results)
    else:
        from numpy import save
        with open(texts, "r") as fin:
            line = fin.read()
        p = calculator.compute_token_probabilities(line).numpy()
        save(output, p)


if __name__ == "__main__":
    main()
