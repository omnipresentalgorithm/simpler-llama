"""
Sample from the trained model with PyTorch
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# dtype = "float32"
dtype = "bfloat16"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
print(gptconf)
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    print(k, state_dict[k].shape)
    #layer_params = state_dict[k]
    #for a in layer_params :
    #  for b in a :
    #    print(type(b), end='')
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

total_params = 0
smallest = 0
largest = 0

def print_vector (t) :
  global total_params, smallest, largest
  for e in t :
    if abs(e.item()) > 0.009999 :
      total_params += 1
      # normalized = 56.0 + 92.742 * e.item()
      normalized = e.item()
      if normalized < smallest :
        smallest = normalized
      if normalized > largest :
        largest = normalized
      print(" %.2f" % e.item(), end='')
      # print(" %x" % int(normalized), end='')
      # print("%s" % chr(int(normalized)), end='')
    else :
      print(" 0", end='')

'''
for k, v in list(state_dict.items()):
    if k == 'output.weight' or k == 'tok_embeddings.weight':
      continue

    layer_params = state_dict[k]
    nrows = layer_params.shape[0]
    if len(layer_params.shape) > 1 :
      ncolumns = layer_params.shape[1]
      print(f'\n\n--- { k } parameters form a { nrows }x{ ncolumns } matrix:', end='')
      i = 0
      for column in layer_params :
        print(f'\n  -- column { i }:\n', end='')
        i += 1
        print_vector(column)
    else :
      print(f'\n\n--- { k } parameters form a { nrows }-dimensional vector:')
      print_vector(layer_params)
'''
# print('\n\nTotal number of parameters:', total_params, 'Smallest value:', smallest, 'Largest', largest)
# exit()

model.load_state_dict(state_dict, strict=False)


model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            sys.stdout.write(enc.decode(x[0].tolist()))
            sys.stdout.flush()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decoder=enc)
            # print(enc.decode(y[0].tolist()))
            # print('---------------')
            sys.stdout.write('\n------------------\n')
            sys.stdout.flush()
