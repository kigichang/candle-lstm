import torch
import torch.nn as nn
import time

IN_DIM = 768
HIDDEN_DIM = 768
SEQ_LEN = 256
BATCH_SIZE = 1
LAYER = 1

rnn = nn.LSTM(input_size=IN_DIM, hidden_size=HIDDEN_DIM, num_layers=LAYER, batch_first=False)
input = torch.randn(SEQ_LEN, BATCH_SIZE, IN_DIM)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("lstm with batch_first = false taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_test.pt")


rnn = nn.LSTM(input_size=IN_DIM, hidden_size=HIDDEN_DIM, num_layers=LAYER, batch_first=True)
input = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("lstm with batch_first = true  taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_test_batch_first.pt")


# Bi-LSTM

rnn = nn.LSTM(input_size=IN_DIM, hidden_size=HIDDEN_DIM, num_layers=LAYER, batch_first=False, bidirectional=True)
input = torch.randn(SEQ_LEN, BATCH_SIZE, IN_DIM)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("bi-lstm with batch_first = false taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "bi_lstm_test.pt")


rnn = nn.LSTM(input_size=IN_DIM, hidden_size=HIDDEN_DIM, num_layers=LAYER, batch_first=True, bidirectional=True)
input = torch.randn(BATCH_SIZE, SEQ_LEN, IN_DIM)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("bi-lstm with batch_first = true  taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "bi_lstm_test_batch_first.pt")