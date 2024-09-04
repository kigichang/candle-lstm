# Candle LSTM

Re-implementing Candle LSTM inference to speed inference up, including bidirectional LSTM.

## CPU Only

My LSTM implementation sppeds up inference on CPU ONLY. It is not optimized for GPU.

## Test Data

Install Pytorch and run simple.py to generate test data.

1. lstm_test.pt: Pytorch LSTM with batch_first = False.
1. lstm_test_batch_first.pt: Pytorch LSTM with batch_first = True.
1. bi_lstm_test.pt: Pytorch Bidirectional LSTM with batch_first = False.
1. bi_lstm_test_batch_first.pt: Pytorch Bidirectional LSTM with batch_first = True.
