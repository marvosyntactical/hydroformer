import os, math, time, urllib.request, zipfile
import torch, torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import requests, textwrap, tarfile, io
from tqdm import tqdm
import argparse
import neptune
import random




def parse_args():
    parser = argparse.ArgumentParser(description="Swarm Experiments on MNIST")
    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")
    parser.add_argument("--baseline", action="store_true", help="Do vanilla Sinkformer?")
    parser.add_argument("--dataset", type=str, choices=[
            "tinystories",
            "wiki2",
        ], default="tinystories", help="Dataset name"
    )
    parser.add_argument("--enc", type=str, choices=["char", "bpe"], default="char", help="Tokenizer type (char, bpe)")
    parser.add_argument("--device", type=str, choices=["cuda", "gpu", "cpu"], default="gpu")

    return parser.parse_args()


def init_neptune(args):

    with open("../.neptune_tok", "r") as f:
        tok = f.read()

    run = neptune.init_run(
        project="halcyon/hydro",
        api_token=tok,
    )

    # run["parameters/bla"] = args.bla # TODO
    run["parameters/baseline"] = args.baseline

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)
    else:
        run = {}

    if args.baseline:
        from model import GPT, GPTConfig
    else:
        from hydro_model import GPT, GPTConfig

    if args.dataset == "wiki2":

        print("Loading WikiText2 via HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(dataset['train']['text'])[:10_000_000]  # 10MB slice
        val_text   = "\n".join(dataset['validation']['text'])

        # Byte-level tokenizer
        chars = sorted(set(train_text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        def encode(s): return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
        def decode(t): return ''.join(itos[int(i)] for i in t)

        train_ids = encode(train_text)
        val_ids   = encode(val_text)

        BLOCK = 128
        class CharDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data) - BLOCK
            def __getitem__(self, idx):
                x = self.data[idx:idx+BLOCK]
                y = self.data[idx+1:idx+BLOCK+1]
                return x, y


        trn_data = CharDataset(train_ids)
        val_data = CharDataset(val_ids)

        voc_size = len(stoi)


    elif args.dataset == "tinystories":

        print("Loading TinyStories …")
        ds = load_dataset("roneneldan/TinyStories")  # 17‑tokenised already

        # Concatenate train split into raw string
        raw_text = "\n".join(ds["train"]["text"])
        # Take ~25 MB slice for quick runs (25_000_000 chars)
        raw_text = raw_text[:25_000_000]

        if args.enc == "char":
            # Very simple byte‑level vocab
            chars = sorted(set(raw_text))
            stoi  = {ch:i for i,ch in enumerate(chars)}
            itos  = {i:ch for ch,i in stoi.items()}
            def encode(s): return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
            def decode(t): return "".join(itos[int(i)] for i in t)

            ids = encode(raw_text)

            voc_size = len(stoi)

        elif args.enc == "bpe":

            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")   # any GPT2‑style BPE
            ids = torch.tensor(tok.encode(raw_text), dtype=torch.long)
            voc_size = tok.vocab_size

        # 90/10 split
        split = int(0.9 * len(ids))
        train_ids = ids[:split]
        val_ids   = ids[split:]

        BLOCK = 256  # Stories use longer context

        class TinyDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data) - BLOCK
            def __getitem__(self, idx):
                x = self.data[idx:idx+BLOCK]
                y = self.data[idx+1:idx+BLOCK+1]
                return x, y

        trn_data = TinyDataset(train_ids)
        val_data = TinyDataset(val_ids)


# text = open(path_txt).read()
# tok  = ByteTokenizer(text)
# data = torch.tensor(tok.encode(text), dtype=torch.long)



# ------------------------------------------------------------------
# 2. Split & Dataset
# ------------------------------------------------------------------
# n = int(0.9*len(data))


    BATCH = 64
    train_loader = DataLoader(trn_data, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH, shuffle=False, drop_last=True)

    device = 'cuda' if torch.cuda.is_available() and not args.device == "cpu" else "cpu"
    print("Using Device:\t", device)
    print(f"Vocab size: {voc_size}")

# ------------------------------------------------------------------
# 3. Model instantiation
# ------------------------------------------------------------------

    cfg = GPTConfig(
        block_size = BLOCK,
        vocab_size = voc_size,
        n_layer    = 3,
        n_head     = 8,
        n_embd     = 256, # 512,
        dropout    = 0.1,
        bias       = False
    )
    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-1)

# LR schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=3*len(train_loader))

# ------------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------------
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(3):                        # 3 epochs ~ quick demo
        model.train()
        # loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        loop = train_loader
        for step,(x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()

            ppl = math.exp(min(10, loss.item()))
            # loop.set_postfix({
            #     "loss": f"{loss.item():.3f}",
            #     "ppl": f"{ppl:6.1f}",
            #     "grad": f"{grad_norm:.2f}",
            #     "lr": f"{sched.get_last_lr()[0]:.2e}"
            # })
            print("Loss:\t",loss.item())

            if args.neptune:
                run["train/loss"].append(loss.item())

            if step % 1000 == 999:
                print("Validating ...")
                model.eval(); val_loss=0; n=0
                with torch.no_grad():
                    for vx,vy in val_loader:
                        vx,vy = vx.to(device), vy.to(device)
                        _, l = model(vx,vy); val_loss+=l.item()*len(vx); n+=len(vx)
                val_loss /= n
                ppl = math.exp(min(10,val_loss))
                print(f"ep {epoch} it {step}  train {loss.item():6.3f}   val {val_loss:6.3f}  ppl {ppl:6.1f}  "
                      f"elapsed {time.time()-t0:5.1f}s")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"{ROOT}/hydro_lite_best.pt")
                    print("  > saved checkpoint")
                model.train()

    print("Training done. Best val loss:", best_val)

if __name__ == "__main__":

    args = parse_args()
    main(args)
