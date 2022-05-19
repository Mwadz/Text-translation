# -*- coding: utf-8 -*-
"""Final App Translation

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iG1ZyrjnPhWXZIEiReOfvH9sOeZxPlR4

# Text Translation Using Neural Networks

##  Defining the Question

### i) Specifying the Question

Use neural networks to translate English text to a local Kenyan language(Luo).

### ii) Defining the metrics of success

Building a model that can accurately translate English text to Luo with an accuracy score of at least 85%

### iii) Understanding the context

There are several translation websites that mostly translate between international languages such as English to Swahili. In Kenya, there a professional bodies that offer translation and interpretation services. Hiring these services can be quite expensive especially when trying to communicate an important information such as constitution interpretation to a pre-dominantly native speaking community. Having a web application can greatly reduce this burden of having to outsource translation services everytime they are needed.

### iv) Recording the Experimental Design

1. Loading the datasets.

2. Cleaning the datasets.

3. Preprocessing.

4. Creating a TensorFlow model.

5. Test Processing.

6. Training the model.

7. Translating.

8. Visualizing the process.

### v) Relevance of the data

The data used in this project is for performing Text Translation using Neural Networks. The dataset link : https://drive.google.com/drive/folders/1qJgQvNd99E_U6oitRIToOdXPbjqEHqnG?usp=sharing

##  Installations
"""

pip install "tensorflow-text==2.8.*"

"""##  Importing the libraries"""

from __future__ import absolute_import, division, print_function
# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time
print(tf.__version__)#to check the tensorflow version

"""##  Shapechecker

Function to prevent loading of data of wrong shape
"""

class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    if isinstance(names, str):
      names = (names,)

    shape = tf.shape(tensor)
    rank = tf.rank(tensor)

    if rank != len(names):
      raise ValueError(f'Rank mismatch:\n'
                       f'    found {rank}: {shape.numpy()}\n'
                       f'    expected {len(names)}: {names}\n')

    for i, name in enumerate(names):
      if isinstance(name, int):
        old_dim = name
      else:
        old_dim = self.shapes.get(name, None)
      new_dim = shape[i]

      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

"""##  Loading the datasets"""

# Loading the datasets
english = pd.read_csv('/content/english.txt', sep='delimiter', engine = 'python', header=None)
kiuk = pd.read_csv('/content/Kikuyu.txt', sep='delimiter', engine = 'python', header=None)
kale = pd.read_csv('/content/kale.txt', sep='delimiter',  engine = 'python', header=None)
luo = pd.read_csv('/content/luo.txt', sep='delimiter', engine = 'python', header=None)

"""##  Previewing the datasets"""

# print the shape of the various datasets
files = [english, kiuk, luo, kale]
dataset_names = ['English', 'Kikuyu', 'Luo', 'Kalenjin']
for file in files:
  #for index in range(len(dataset_names)):
    rows, columns = file.shape
    print(f'The dataset has {rows} rows and {columns} columns')

"""##  Pre_processing"""

# (Unicode is the universal character encoding used to process, store and facilitate the interchange of text data in any language 
# while ASCII is used for the representation of text such as symbols, letters, digits, etc)

"""Preprocessing steps includes

- Converting the unicode file to ascii
- Creating a space between a word and the punctuation following it
eg: “he is a boy.” => “he is a boy .” Reference
- Replacing everything with space except (a-z, A-Z, “.”, “?”, “!”, “,”)
- Adding a start and an end token to the sentence so that the model know when to start and stop predicting.
- Removing the accents
- Cleaning the sentences
- Return word pairs in the format: [ENGLISH, LUO]
- Creating a word -> index mapping (e.g,. 'Further' -> 5) and vice-versa. (e.g., 5 -> 'Further' ) for each language.
"""

# Creating an index column for Kalenjin file
kale['index_col'] = kale.index

# Creating an index column for Kalenjin file
english['index_col'] = kale.index

# Joining the English and Kalenjin file with the Index column
df_kale = pd.merge(english, kale, on = 'index_col')

# Renaming the Kalenjin Columns
df_kale.head()
df_kale.columns = ['feature', 'index', 'target']

# Dropping the Index column in the Kalenjjin file
df_kale.columns
df_kale = df_kale.drop(columns = ['index'])

# Displaying the first rows on the Kalenjin file
df_kale.head()

# Removing the numbers at the beginning of the feature column
df_kale['feature'] = df_kale['feature'].str.replace('\d+', '')

df_kale.head()

df_kale['feature'] = df_kale['feature'].str.replace('\d+', '')

df_kale['target'] = df_kale['target'].str.replace('\d+', '')

df_kale

# from google.colab import files
# files.download("df_kale.csv")

inp = df_kale['target'].to_list()

targ = df_kale['feature'].to_list()

"""##  Creating tf_dataset

Creating a tf.data.Dataset of strings that shuffles and batches them efficiently:
"""

# Tells TensorFlow to create a buffer of at most buffer_size elements, and a background thread to fill that buffer in the background
BUFFER_SIZE = len(inp)

# Number of samples to be feed into the neural network
BATCH_SIZE = 3

# Creating the dataset and shuffling it 
dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset

for example_input_batch, example_target_batch in dataset.take(1):
  print(example_input_batch[:5])
  print()
  print(example_target_batch[:5])
  break

"""##  Text processing

### i) Standardization

Since the model is dealing with multilingual text with a limited vocabulary standardization of the text is crucial. Steps;
1.  Unicode normalization to split accented characters
2.  replace compatibility characters with their ASCII equivalents.
"""



import tensorflow_text as tf_text

# example of a text normalized and uni encoded
sample_text = tf.constant('Kiacheng’in eng’ muguleldanyu tugul')

print(sample_text.numpy())
print(tf_text.normalize_utf8(sample_text, 'NFKD').numpy())

# Unicode normalization 
def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

# Priniting an example of the original text
print(sample_text.numpy().decode())

# printing the text afterunicode normalization
print(tf_lower_and_split_punct(sample_text).numpy().decode())

# Extracting and coverting input text to sequences of tokens
# max_vocab_size limit RAM usage during the initial scan of the training corpus to discover the vocabulary.
max_vocab_size = 25000 

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

# Reading one epoch of the training data with the adapt method 
input_text_processor.adapt(inp)

# Here are the first 10 words from the vocabulary:
input_text_processor.get_vocabulary()[:10]

# Using the Kalenjin TextVectorization layer to build the English layer with .adapt() method
output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

output_text_processor.adapt(targ)
output_text_processor.get_vocabulary()[:10]

# Using the layers created to convert a batch of strings into a batch of token IDs
example_tokens = input_text_processor(example_input_batch)
example_tokens[:3, :10]

# Applying the token IDs that are zero-padded that can be turned into a mask
plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')

# Defining constants for the model
# Embedding layer enables us to convert each word into a fixed length vector of defined size
embedding_dim = 512
units = 1024

"""##  The encoder

The first thing to do is build the encoder. The process is as follows:

1. Taking a list of token IDs. 

2. Using the embedding vector for each token.

3. Processessing the embeddings into a new sequence
"""

# Applying the  list of token IDs
class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    shape_checker = ShapeChecker()
    shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The GRU processes the embedding sequence.
    #    output shape: (batch, s, enc_units)
    #    state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    shape_checker(output, ('batch', 's', 'enc_units'))
    shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the new sequence and its state.
    return output, state

# Convert the input text to tokens.
example_tokens = input_text_processor(example_input_batch)

# Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(),
                  embedding_dim, units)
example_enc_output, example_enc_state = encoder(example_tokens)

print(f'Input batch, shape (batch): {example_input_batch.shape}')
print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')



"""##  The attention head

The decoder uses attention to selectively focus on parts of the input sequence. The attention takes a sequence of vectors as input for each example and returns an "attention" vector for each example.
"""

# The BahdanauAttention class handles the weight matrices in a pair of dense layers and calls the builtin implementation
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super().__init__()
    # For Eqn. (4), the  Bahdanau attention
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(units, use_bias=False)

    self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value, mask):
    shape_checker = ShapeChecker()
    shape_checker(query, ('batch', 't', 'query_units'))
    shape_checker(value, ('batch', 's', 'value_units'))
    shape_checker(mask, ('batch', 's'))

    # From Eqn. (4), `W1@ht`.
    w1_query = self.W1(query)
    shape_checker(w1_query, ('batch', 't', 'attn_units'))

    # From Eqn. (4), `W2@hs`.
    w2_key = self.W2(value)
    shape_checker(w2_key, ('batch', 's', 'attn_units'))

    query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
    value_mask = mask

    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value, w2_key],
        mask=[query_mask, value_mask],
        return_attention_scores = True,
    )
    shape_checker(context_vector, ('batch', 't', 'value_units'))
    shape_checker(attention_weights, ('batch', 't', 's'))

    return context_vector, attention_weights

"""### i) Attention head layer"""

# Creating a BahdanauAttention layer
attention_layer = BahdanauAttention(units)

# Excluding the padding
(example_tokens != 0).shape

# Later, the decoder will generate this attention query
example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])

# Attend to the encoded tokens

context_vector, attention_weights = attention_layer(
    query=example_attention_query,
    value=example_enc_output,
    mask=(example_tokens != 0))

print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

# attention weights across the sequences at t=0
# t is used for slicing, for selecting different parts of the data.
plt.subplot(1, 2, 1)
plt.pcolormesh(attention_weights[:, 0, :])
plt.title('Attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')

# Displaying the shape of the attention weights
attention_weights.shape

attention_slice = attention_weights[0, 0].numpy()
attention_slice = attention_slice[attention_slice != 0]

"""### ii) Toogle code"""

# Plotting attention weights
plt.suptitle('Attention weights for one sequence')

plt.figure(figsize=(12, 6))
a1 = plt.subplot(1, 2, 1)
plt.bar(range(len(attention_slice)), attention_slice)
# freeze the xlim
plt.xlim(plt.xlim())
plt.xlabel('Attention weights')

a2 = plt.subplot(1, 2, 2)
plt.bar(range(len(attention_slice)), attention_slice)
plt.xlabel('Attention weights, zoomed')

# zoom in
top = max(a1.get_ylim())
zoom = 0.85*top
a2.set_ylim([0.90*top, top])
a1.plot(a1.get_xlim(), [zoom, zoom], color='k')

"""### iii) The decoder

The decoder generates predictions for the next output token.
1. The decoder receives the complete encoder output.

2. It uses an RNN to keep track of what it has generated so far.

3. It uses its RNN output as the query to the attention over the encoder's output, producing the context vector.

4. It combines the RNN output and the context vector to generate the attention vector.

5. It generates logit predictions for the next token based on the attention vector.
"""

# Decoder class and its initializer creates all the necessary layers.

class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim

    # The embedding layer convets token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)

    # The RNN keeps track of what's been generated so far.
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # The RNN output will be the query for the attention layer.
    self.attention = BahdanauAttention(self.dec_units)

    #  Eqn. (3): converting `ct` to `at`
    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                    use_bias=False)

    # This fully connected layer produces the logits for each
    # output token.
    self.fc = tf.keras.layers.Dense(self.output_vocab_size)

# Importing libraries
import typing
from typing import Any, Tuple

# Applying the call method for this layer which  takes and returns multiple tensors.
# Organizing those into simple container classes.
class DecoderInput(typing.NamedTuple):
  new_tokens: Any
  enc_output: Any
  mask: Any

class DecoderOutput(typing.NamedTuple):
  logits: Any
  attention_weights: Any

# Implementing the call method
def call(self,
         inputs: DecoderInput,
         state=None) -> Tuple[DecoderOutput, tf.Tensor]:
  shape_checker = ShapeChecker()
  shape_checker(inputs.new_tokens, ('batch', 't'))
  shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
  shape_checker(inputs.mask, ('batch', 's'))

  if state is not None:
    shape_checker(state, ('batch', 'dec_units'))

  # Step 1. Lookup the embeddings
  vectors = self.embedding(inputs.new_tokens)
  shape_checker(vectors, ('batch', 't', 'embedding_dim'))

  # Step 2. Process one step with the RNN
  rnn_output, state = self.gru(vectors, initial_state=state)

  shape_checker(rnn_output, ('batch', 't', 'dec_units'))
  shape_checker(state, ('batch', 'dec_units'))

  # Step 3. Use the RNN output as the query for the attention over the
  # encoder output.
  context_vector, attention_weights = self.attention(
      query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
  shape_checker(context_vector, ('batch', 't', 'dec_units'))
  shape_checker(attention_weights, ('batch', 't', 's'))

  # Step 4. Eqn. (3): Join the context_vector and rnn_output
  #     [ct; ht] shape: (batch t, value_units + query_units)
  context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

  # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
  attention_vector = self.Wc(context_and_rnn_output)
  shape_checker(attention_vector, ('batch', 't', 'dec_units'))

  # Step 5. Generate logit predictions:
  logits = self.fc(attention_vector)
  shape_checker(logits, ('batch', 't', 'output_vocab_size'))

  return DecoderOutput(logits, attention_weights), state

Decoder.call = call

# Implementing  of the decoder 
decoder = Decoder(output_text_processor.vocabulary_size(),
                  embedding_dim, units)

# Convert the target sequence, and collect the "[START]" tokens
example_output_tokens = output_text_processor(example_target_batch)

start_index = output_text_processor.get_vocabulary().index('[START]')
first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])

# Run the decoder
dec_result, dec_state = decoder(
    inputs = DecoderInput(new_tokens=first_token,
                          enc_output=example_enc_output,
                          mask=(example_tokens != 0)),
    state = example_enc_state
)

print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')
print(f'state shape: (batch_size, dec_units) {dec_state.shape}')

# Sampling a token with the logits
sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)

# Decoding the token as the first word of the output
vocab = np.array(output_text_processor.get_vocabulary())
first_word = vocab[sampled_token.numpy()]
first_word[:5]

# Applying the same enc_output, mask and sampled tokens as new tokens.

dec_result, dec_state = decoder(
    DecoderInput(sampled_token,
                 example_enc_output,
                 mask=(example_tokens != 0)),
    state=dec_state)

# Generating a second set of logits using the decoder
sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)
first_word = vocab[sampled_token.numpy()]
first_word[:5]

"""##  Training

To train the model we'll follow the following steps:

1. A loss function and optimizer to perform the optimization.

2. A training step function defining how to update the model for each input/target batch.

3. A training loop to drive the training and save checkpoints.

### i) Define the loss function
"""

# Implementing the loss function and optimizer to perform the optimization.
class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):
    shape_checker = ShapeChecker()
    shape_checker(y_true, ('batch', 't'))
    shape_checker(y_pred, ('batch', 't', 'logits'))

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)
    shape_checker(loss, ('batch', 't'))

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    shape_checker(mask, ('batch', 't'))
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)

"""### ii) Implementing the training step"""

# Implementing a model class, the training process will be implemented as the train_step method 
class TrainTranslator(tf.keras.Model):
  def __init__(self, embedding_dim, units,
               input_text_processor,
               output_text_processor, 
               use_tf_function=True):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(input_text_processor.vocabulary_size(),
                      embedding_dim, units)
    decoder = Decoder(output_text_processor.vocabulary_size(),
                      embedding_dim, units)

    self.encoder = encoder
    self.decoder = decoder
    self.input_text_processor = input_text_processor
    self.output_text_processor = output_text_processor
    self.use_tf_function = use_tf_function
    self.shape_checker = ShapeChecker()

  def train_step(self, inputs):
    self.shape_checker = ShapeChecker()
    if self.use_tf_function:
      return self._tf_train_step(inputs)
    else:
      return self._train_step(inputs)

# Getting a batch of input_text, target_text from the tf.data.Dataset.
def _preprocess(self, input_text, target_text):
  self.shape_checker(input_text, ('batch',))
  self.shape_checker(target_text, ('batch',))

  # Convert the text to token IDs
  input_tokens = self.input_text_processor(input_text)
  target_tokens = self.output_text_processor(target_text)
  self.shape_checker(input_tokens, ('batch', 's'))
  self.shape_checker(target_tokens, ('batch', 't'))

  # Convert IDs to masks.
  input_mask = input_tokens != 0
  self.shape_checker(input_mask, ('batch', 's'))

  target_mask = target_tokens != 0
  self.shape_checker(target_mask, ('batch', 't'))

  return input_tokens, input_mask, target_tokens, target_mask

TrainTranslator._preprocess = _preprocess

# Applying the _train_step method
def _train_step(self, inputs):
  input_text, target_text = inputs  

  (input_tokens, input_mask,
   target_tokens, target_mask) = self._preprocess(input_text, target_text)

  max_target_length = tf.shape(target_tokens)[1]

  with tf.GradientTape() as tape:
    # Encode the input
    enc_output, enc_state = self.encoder(input_tokens)
    self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
    self.shape_checker(enc_state, ('batch', 'enc_units'))

    # Initialize the decoder's state to the encoder's final state.
    # This only works if the encoder and decoder have the same number of
    # units.
    dec_state = enc_state
    loss = tf.constant(0.0)

    for t in tf.range(max_target_length-1):
      # Pass in two tokens from the target sequence:
      # 1. The current input to the decoder.
      # 2. The target for the decoder's next prediction.
      new_tokens = target_tokens[:, t:t+2]
      step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                             enc_output, dec_state)
      loss = loss + step_loss

    # Average the loss over all non padding tokens.
    average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

  # Apply an optimization step
  variables = self.trainable_variables 
  gradients = tape.gradient(average_loss, variables)
  self.optimizer.apply_gradients(zip(gradients, variables))

  # Return a dict mapping metric names to current value
  return {'batch_loss': average_loss}

TrainTranslator._train_step = _train_step

def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
  input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

  # Run the decoder one step.
  decoder_input = DecoderInput(new_tokens=input_token,
                               enc_output=enc_output,
                               mask=input_mask)

  dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
  self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
  self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
  self.shape_checker(dec_state, ('batch', 'dec_units'))

  # `self.loss` returns the total for non-padded tokens
  y = target_token
  y_pred = dec_result.logits
  step_loss = self.loss(y, y_pred)

  return step_loss, dec_state

TrainTranslator._loop_step = _loop_step

"""### iii) Test the training step"""

# Building a TrainTranslator and configuring it for training using the Model.compile method
translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
    use_tf_function=False)

# Configure the loss and optimizer
translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

# Testing the train_step model
np.log(output_text_processor.vocabulary_size())

# Applying the tf.function-wrapped _tf_train_step, to maximize performance while training
@tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                               tf.TensorSpec(dtype=tf.string, shape=[None])]])
def _tf_train_step(self, inputs):
  return self._train_step(inputs)

TrainTranslator._tf_train_step = _tf_train_step

translator.use_tf_function = True

# Tracing the function
translator.train_step([example_input_batch, example_target_batch])

# Commented out IPython magic to ensure Python compatibility.
# # Printing out the Batch loss of our model
# %%time
# for n in range(10):
#   print(translator.train_step([example_input_batch, example_target_batch]))
# print()

# Plotting our batch losses
losses = []
for n in range(100):
  print('.', end='')
  logs = translator.train_step([example_input_batch, example_target_batch])
  losses.append(logs['batch_loss'].numpy())

print()
plt.plot(losses)

# Building another model to train
train_translator = TrainTranslator(
    embedding_dim, units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor)

# Configure the loss and optimizer
train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

"""### iv) Train the model"""

# Training a couple of epochs by applying the callbacks.Callback method
# to collect the history of batch losses
class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key):
    self.key = key
    self.logs = []

  def on_train_batch_end(self, n, logs):
    self.logs.append(logs[self.key])

batch_loss = BatchLogs('batch_loss')

# Displaying the batch loss using 15 epochs 
train_translator.fit(dataset, epochs=15,
                     callbacks=[batch_loss])

# Plotting the epochs
plt.plot(batch_loss.logs)
plt.ylim([0, 3])
plt.xlabel('Batch #')
plt.ylabel('CE/token')

"""##  Translate"""

# Executing the full text => texttranslation
# This is by inverting the text => token IDsmapping provided by the output_text_processor
class Translator(tf.Module):

  def __init__(self, encoder, decoder, input_text_processor,
               output_text_processor):
    self.encoder = encoder
    self.decoder = decoder
    self.input_text_processor = input_text_processor
    self.output_text_processor = output_text_processor

    self.output_token_string_from_index = (
        tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),
            mask_token='',
            invert=True))

    # The output should never generate padding, unknown, or start.
    index_from_string = tf.keras.layers.StringLookup(
        vocabulary=output_text_processor.get_vocabulary(), mask_token='')
    token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

    token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
    token_mask[np.array(token_mask_ids)] = True
    self.token_mask = token_mask

    self.start_token = index_from_string(tf.constant('[START]'))
    self.end_token = index_from_string(tf.constant('[END]'))

translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

"""### i) Convert IDs to text"""

# Implementing the tokens_to_text which converts from token IDs to human readable text.
def tokens_to_text(self, result_tokens):
  shape_checker = ShapeChecker()
  shape_checker(result_tokens, ('batch', 't'))
  result_text_tokens = self.output_token_string_from_index(result_tokens)
  shape_checker(result_text_tokens, ('batch', 't'))

  result_text = tf.strings.reduce_join(result_text_tokens,
                                       axis=1, separator=' ')
  shape_checker(result_text, ('batch'))

  result_text = tf.strings.strip(result_text)
  shape_checker(result_text, ('batch',))
  return result_text

Translator.tokens_to_text = tokens_to_text

# Inputting some random token IDs and see what it generates (example)
example_output_tokens = tf.random.uniform(
    shape=[5, 2], minval=0, dtype=tf.int64,
    maxval=output_text_processor.vocabulary_size())
translator.tokens_to_text(example_output_tokens).numpy()

"""### ii) Sample from the decoder's predictions"""

# Taking the decoder's logit outputs and samples token IDs from the distribution
def sample(self, logits, temperature):
  shape_checker = ShapeChecker()
  # 't' is usually 1 here.
  shape_checker(logits, ('batch', 't', 'vocab'))
  shape_checker(self.token_mask, ('vocab',))

  token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
  shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

  # Set the logits for all masked tokens to -inf, so they are never chosen.
  logits = tf.where(self.token_mask, -np.inf, logits)

  if temperature == 0.0:
    new_tokens = tf.argmax(logits, axis=-1)
  else: 
    logits = tf.squeeze(logits, axis=1)
    new_tokens = tf.random.categorical(logits/temperature,
                                        num_samples=1)

  shape_checker(new_tokens, ('batch', 't'))

  return new_tokens

Translator.sample = sample

# Random inputs (example)
example_logits = tf.random.normal([5, 1, output_text_processor.vocabulary_size()])
example_output_tokens = translator.sample(example_logits, temperature=1.0)
example_output_tokens

"""### iii) Implement translation loop"""

# Taking the results into python lists before joining them  using tf.concat into tensors.
# This unfolds the graph out to max_length iterations.
def translate_unrolled(self,
                       input_text, *,
                       max_length=50,
                       return_attention=True,
                       temperature=1.0):
  batch_size = tf.shape(input_text)[0]
  input_tokens = self.input_text_processor(input_text)
  enc_output, enc_state = self.encoder(input_tokens)

  dec_state = enc_state
  new_tokens = tf.fill([batch_size, 1], self.start_token)

  result_tokens = []
  attention = []
  done = tf.zeros([batch_size, 1], dtype=tf.bool)

  for _ in range(max_length):
    dec_input = DecoderInput(new_tokens=new_tokens,
                             enc_output=enc_output,
                             mask=(input_tokens!=0))

    dec_result, dec_state = self.decoder(dec_input, state=dec_state)

    attention.append(dec_result.attention_weights)

    new_tokens = self.sample(dec_result.logits, temperature)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (new_tokens == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

    # Collect the generated tokens
    result_tokens.append(new_tokens)

    if tf.executing_eagerly() and tf.reduce_all(done):
      break

  # Convert the list of generates token ids to a list of strings.
  result_tokens = tf.concat(result_tokens, axis=-1)
  result_text = self.tokens_to_text(result_tokens)

  if return_attention:
    attention_stack = tf.concat(attention, axis=1)
    return {'text': result_text, 'attention': attention_stack}
  else:
    return {'text': result_text}

Translator.translate = translate_unrolled

# Commented out IPython magic to ensure Python compatibility.
# # Running a simple input to view the translation
# %%time
# input_text = tf.constant([
#     'Boiboen che igesunotgei eng’ oret.', # "Blessed are the undefiled in the way."
#     'Kilosu Jehovah', # "I have gone astray like a lost sheep"
# ])
# 
# 
# result = translator.translate(
#     input_text = input_text)
# 
# print(result['text'][0].numpy().decode())
# print(result['text'][1].numpy().decode())
# print()

"""##  Visualize the process"""

# Calculating the sum of the attention over the input which should return all ones.
a = result['attention'][0]

print(np.sum(a, axis=-1))

# The attention distribution for the first output step of the first example
# It is focused than it was in the untrained model
_ = plt.bar(range(len(a[0, :])), a[0, :])

# There is some rough alignment between the input and output words
plt.imshow(np.array(a), vmin=0.0)

"""### i) Labelled attention plots"""

# Visualizing the attention plots.
def plot_attention(attention, sentence, predicted_sentence):
  sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()
  predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)

  attention = attention[:len(predicted_sentence), :len(sentence)]

  ax.matshow(attention, cmap='viridis', vmin=0.0)

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  ax.set_xlabel('Input text')
  ax.set_ylabel('Output text')
  plt.suptitle('Attention weights')

i=0
plot_attention(result['attention'][i], input_text[i], result['text'][i])

"""## Export"""

@tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
def tf_translate(self, input_text):
  return self.translate(input_text)

Translator.tf_translate = tf_translate

"""##  Conclusion

a). Did we have the right data?

b). Do we need other data to answer our question?

c) Did we have the right question?

## Installations
"""

pip install ipykernel>=5.1.2
pip install pydeck
pip install streamlit==0.75.0
pip install pyngrok
pip install streamlit -q
pip install streamlit --upgrade
pip install streamlit-option-menu

"""## App"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile finalLHTranslation.py
# import streamlit as st
# import pandas as pd
# import plotly.figure_factory as ff
# from streamlit_option_menu import option_menu
# from textblob import TextBlob 
# import tensorflow as tf
#  
#  
# 
# # let's do the navigation bar first
# 
# selected = option_menu(
#       menu_title= None, options=['Home','Features', 'About'], icons =['house','book','boxes'],menu_icon='cast', default_index=0, orientation = 'horizontal'
#   )
# 
# 
# #theme
# CURRENT_THEME = "light"
# IS_DARK_THEME = False
# EXPANDER_TEXT = """
#     This is Streamlit's default *Light* theme. It should be enabled by default 🎈
#     If not, you can enable it in the app menu (☰ -> Settings -> Theme).
#     """
# 
# # setting containers
# header = st.container()
# translation = st.container()
# dataset = st.container()
# features = st.container()
# modelTraining = st.container()
# 
# with header:
#   col1, col2 = st.columns([1,6])
# 
#   with col1:
#     st.image(
#     "https://cdn0.iconfinder.com/data/icons/joker-circus-by-joker2011-d3g8h6s/256/lion.png", width=100,)
# 
#   with col2:
#     st.markdown("<h1 style='text-align: left; color: Orange;'> Lion Heart Translation</h1>", unsafe_allow_html=True)
# 
# 
# 
#   # st.image(
#   #   "https://cdn0.iconfinder.com/data/icons/joker-circus-by-joker2011-d3g8h6s/256/lion.png", width=100,)
#   # st.markdown("<h1 style='text-align: center; color: Purple;'> Lion Heart Translation</h1>", unsafe_allow_html=True)
#   # st.text("Lion Heart Translation is an App who's main porpose is to translate Kenyan local languages, that is Kalenjin, Luo and Kikuyu to English and vise versa. It's Using a tensorflow neural network model in order to do this.")
# with st.expander("Pick out a theme of your liking?"):
#   THEMES = [
#     "light",
#     "dark",
#     "green",
#     "blue",]
#   GITHUB_OWNER = "streamlit"
# 
# 
# 
# # modeling ie translation code
# if selected == 'Home':
#   with translation:
#     from textblob import TextBlob 
#     import spacy
#     from gensim.summarization import summarize
#     sp = spacy.load('en_core_web_sm')
#     from spacy import displacy
#  
#     # Add selectbox in streamlit
#     st.markdown("""<span style="word-wrap:break-word;">Lion Heart Translation is an App who's main porpose is to translate Kenyan local languages, that is Kalenjin, Luo and Kikuyu to English and vise versa. It's Using a tensorflow neural network model in order to do this.</span>""", unsafe_allow_html=True)
#     option = st.selectbox(
#      'Which Local language would you like to translate to?',
#         ('none', 'Kikuyu', 'Kalenjin', 'Luo'))
#     st.write('You selected:', option)
# 
#     def main():
#       text = st.text_area("Enter Text to translate here: ","lorem ipsum...",key = "<255>")
#       if st.button("Translate"):
#         input_text = tf.constant(text)
#         result = translator.translate(input_text = input_text)
#         show = result['text'][0].numpy().decode()
#         st.success(show)
# 
# if __name__=='__main__':
#   main()
# 
# 
# if selected == 'About':
#   with dataset:
#     st.header('About')
#     st.markdown("""<span style="word-wrap:break-word;">The data used to build this model was obtained from a chapter in the Bible in each of the four languages.</span>""", unsafe_allow_html=True)
#     
#     # st.text('The data used to build this model was obtained from a chapter in the Bible in each of the four languages.')
#     st.text("here's what the Kalenjin looks side by side with it's English translation")
#     kaleme = pd.read_csv('/content/kaleme.csv', sep='delimiter', engine = 'python', header=None)
#     st.write(kaleme.head(5))
# 
# if selected == 'Features':
#   with features:
#     st.header('Features')
#     st.markdown('* **Hyperparameter tuning:** Here the user can tweak the model settings in pursuit of higher accuracy')
#     st.markdown('* **Language Dropdown:** Here the user can tweak the model settings in pursuit of higher accuracy')
#     st.markdown('* **Translation textbox:** Here the user can tweak the model settings in pursuit of higher accuracy')

# ! streamlit run finalLHTranslation.py & npx localtunnel --port 8501