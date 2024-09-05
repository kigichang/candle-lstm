# Candle LSTM

Re-implementing Candle LSTM inference to speed inference up, including bidirectional LSTM.

## CPU And GPU Inference

I test inference on My Macbook Pro with M2 chip and on Metal GPU (see test cases). My LSTM implementation sppeds up inference on CPU ONLY. It is not optimized on Metal. It is ~5ms slower than Candle on Metal.

I have no Cuda platform, so I cannot verify it on Cuda.

## Test Data

Install Pytorch and run simple.py to generate test data.

1. lstm_test.pt: Pytorch LSTM with batch_first = False.
1. lstm_test_batch_first.pt: Pytorch LSTM with batch_first = True.
1. bi_lstm_test.pt: Pytorch Bidirectional LSTM with batch_first = False.
1. bi_lstm_test_batch_first.pt: Pytorch Bidirectional LSTM with batch_first = True.
