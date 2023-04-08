from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os, sys
from typing import List, Tuple


def get_tokenized_text_info(tokenized_text:dict, tokenizer:AutoTokenizer, text:str)->None:
  '''
  input:
    tokenized_text: output from tokenizer
    text: List[str] of words which have been tokenized
  output:
    information of tokenizer output ( key, values ) and how each word is tokenized
    here we can see if a word is split into multiple tokens ( marked by prefix ## )
  '''
  print("\n\n{} is tokenized as {}".format(text, tokenized_text))
  for key, value in tokenized_text.items():
    print("{} = {}".format(key, value))
  for each_id in tokenized_text["input_ids"][0]:
    print(tokenizer.decode(each_id))

def get_tokenized_input(tokenizer:AutoTokenizer, text:List[str])->dict:
    '''
    input:
      tokenizer: AutoTokenizer of relevant model 
      text: List of words to tokenize
    output:
      AutoTokenizer output - typically input_ids, token_type_ids, attention_mask
    '''
    tokenized_input = tokenizer(text, is_split_into_words=True,return_tensors='pt',padding=True)
    get_tokenized_text_info(tokenized_input, tokenizer, text)
    return tokenized_input


def get_token_embed(**kwargs)->np.ndarray:
  '''
  input:
      model : AutoModelForMaskedLM or AutoModelForSequenceClassification 
      tokenized_input : Output from AutoTokenizer or preprocessor for corresponding model
      batch_index: Index of sample who's embedding we want
      token_index: Index of token whose embedding we want to access
      hidden_state_index: Value from 0 to 12 representing the hidden layer whose embedding we want to access
  output:
      numpy array of the required hidden state of the specified token 
  '''
  model = kwargs["model"]
  tokenized_input = kwargs["tokenized_input"]
  hidden_state_index = kwargs["hidden_state_index"]
  batch_index = kwargs["batch_index"]
  token_index = kwargs["token_index"]
  model.eval()
  with torch.no_grad():
    output = model(**tokenized_input)
  all_hidden_states = output["hidden_states"]
  return all_hidden_states[hidden_state_index][batch_index][token_index].detach().numpy()

def get_min_max_values(embeddings:List[torch.Tensor])->Tuple[np.float64, np.float64]:
  '''
  input: embeddings List of numpy.ndarray
  output: (min, max) value for this list of tensors
  '''
  consolidated = np.asarray(embeddings)
  max = np.max(consolidated)
  min = np.min(consolidated)
  return min ,max

def get_sum_last_N_layers(**kwargs)->np.ndarray:
  '''
  input:
    model : AutoModelForMaskedLM or AutoModelForSequenceClassification 
    tokenized_input : Output from AutoTokenizer or preprocessor for corresponding model
    last_N: last N layers to take
    token_index: Calculate embedding for this token
  output:
    np.ndarray of the sum of the last N hidden layers
  '''
  model = kwargs["model"]
  tokenized_input = kwargs["tokenized_input"]
  last_N = kwargs["last_N"]
  token_index = kwargs["token_index"]
  model.eval()
  with torch.no_grad():
    output = model(**tokenized_input)
  consolidated = torch.stack(output["hidden_states"])#13,1,6,768
  selected_token = consolidated[:,:,token_index,:]
  last_N_layers = selected_token[-last_N:,:,:]

  summed_embed = torch.sum(last_N_layers, axis=0)
  return summed_embed[0].detach().numpy()

def get_sentence_embed(**kwargs):
  '''
  input:
    model : AutoModelForMaskedLM or AutoModelForSequenceClassification 
    tokenized_input : Output from AutoTokenizer or preprocessor for corresponding model
  output:
    second last hidden layer mean across 768 dimensions to get 1 embedding vector per sequence
  '''
  model = kwargs["model"]
  tokenized_input = kwargs["tokenized_input"]
  model.eval()
  with torch.no_grad():
    output = model(**tokenized_input)
  all_hidden_states = torch.stack(output["hidden_states"])
  #Take out the second last layer's embedding, using 0 as batch=1 for us
  consolidated = all_hidden_states[-2,0,:,:]
  #Ignoring the first and last token for [CLS] and [SEP]
  consolidated = consolidated[1:-1,:]
  sentence_vector = torch.mean(consolidated, axis=0)
  return sentence_vector.detach().numpy()
