import torch
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
# Tokenize natural language and code
nl_tokens = tokenizer.tokenize("return maximum value")
code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]

from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
outputs = fill_mask("if (x is not None) <mask> (x>1)")
print(outputs)

