# Candle LSTM

Re-implementing Candle LSTM inference to speed inference up, including bidirectional LSTM.

This implementation is ONLY FOR CPU INFERENCE. DO NOT USE IT ON METAL OR CUDA.

## Metal and Cuda

I test inference on My Macbook Pro with M2 chip. It is ~5ms slower than Candle on Metal.

I test On RTX4090 with Cuda 12.5. It is ~6x slower than Candle on Cuda.

## Test Data

Install Pytorch and run simple.py to generate test data.

1. lstm_test.pt: Pytorch LSTM with batch_first = False.
1. lstm_test_batch_first.pt: Pytorch LSTM with batch_first = True.
1. bi_lstm_test.pt: Pytorch Bidirectional LSTM with batch_first = False.
1. bi_lstm_test_batch_first.pt: Pytorch Bidirectional LSTM with batch_first = True.
