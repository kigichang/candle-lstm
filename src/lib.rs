//
//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//               佛祖保佑         永無BUG
//
// FROM: https://gist.github.com/edokeh/7580064
//

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{init, Init, VarBuilder};

/// Get the input tensor for batch first, i.e. the second dimension is the sequence length.
fn get_tensor_for_batch_first(input: &Tensor, index: usize) -> Result<Tensor> {
    input.i((.., index, ..))?.contiguous()
}

// -----------------------------------------------------------------------------

/// Get the input tensor for sequence first, i.e. the first dimension is the sequence length.
fn get_tensor_for_seq_first(input: &Tensor, index: usize) -> Result<Tensor> {
    input.get(index)
}

// -----------------------------------------------------------------------------

/// Generate or get the LSTM weights and biases.
fn gen_lstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: VarBuilder,
    reverse: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
    let layer_idx = config.layer_idx;
    let reverse = if reverse { "_reverse" } else { "" };
    let w_ih = vb.get_with_hints(
        (4 * hidden_dim, in_dim),
        &format!("weight_ih_l{layer_idx}{reverse}"), // Only a single layer is supported.
        config.w_ih_init,
    )?;
    let w_hh = vb.get_with_hints(
        (4 * hidden_dim, hidden_dim),
        &format!("weight_hh_l{layer_idx}{reverse}"), // Only a single layer is supported.
        config.w_hh_init,
    )?;
    let b_ih = match config.b_ih_init {
        Some(init) => Some(vb.get_with_hints(
            4 * hidden_dim,
            &format!("bias_ih_l{layer_idx}{reverse}"),
            init,
        )?),
        None => None,
    };
    let b_hh = match config.b_hh_init {
        Some(init) => Some(vb.get_with_hints(
            4 * hidden_dim,
            &format!("bias_hh_l{layer_idx}{reverse}"),
            init,
        )?),
        None => None,
    };
    Ok((w_ih, w_hh, b_ih, b_hh))
}

// -----------------------------------------------------------------------------

fn lstm_cell(input: &Tensor, last_c: &Tensor) -> Result<(Tensor, Tensor)> {
    let chunks = input.chunk(4, 1)?;
    let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
    let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
    let cell_gate = chunks[2].tanh()?;
    let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;

    let next_c = ((forget_gate * last_c)? + (in_gate * cell_gate)?)?;
    let next_h = (out_gate * next_c.tanh()?)?;

    Ok((next_h, next_c))
}

// -----------------------------------------------------------------------------

fn compute_input_layer(input: &Tensor, w_ih: &Tensor, b_ih: Option<&Tensor>) -> Result<Tensor> {
    let (d1, d2, _features) = input.dims3()?;
    let (seq_len, seq_dim) = if d1 > d2 { (d2, 1) } else { (d1, 0) };
    let get = if d1 > d2 {
        get_tensor_for_batch_first
    } else {
        get_tensor_for_seq_first
    };

    let mut output: Vec<Tensor> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let x = get(input, i)?;
        let x = x.matmul(&w_ih)?;
        output.push(if let Some(b_ih) = b_ih {
            x.broadcast_add(b_ih)?
        } else {
            x
        });
    }

    Tensor::stack(&output, seq_dim)
}

// -----------------------------------------------------------------------------

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct LSTMConfig {
    pub w_ih_init: Init,
    pub w_hh_init: Init,
    pub b_ih_init: Option<Init>,
    pub b_hh_init: Option<Init>,
    pub layer_idx: usize,
    pub batch_first: bool,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(Init::Const(0.)),
            b_hh_init: Some(Init::Const(0.)),
            layer_idx: 0,
            batch_first: false,
        }
    }
}

impl LSTMConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            batch_first: false,
        }
    }
}

// -----------------------------------------------------------------------------

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }

    pub fn zero(
        batch_size: usize,
        hidden_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let z = Tensor::zeros((batch_size, hidden_dim), dtype, device)?.contiguous()?;
        Ok(Self { h: z.clone(), c: z })
    }
}

// -----------------------------------------------------------------------------

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms, unused)]
#[derive(Clone, Debug)]
pub struct LSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

// -----------------------------------------------------------------------------

pub fn lstm(in_dim: usize, hidden_dim: usize, config: LSTMConfig, vb: VarBuilder) -> Result<LSTM> {
    let (w_ih, w_hh, b_ih, b_hh) = gen_lstm(in_dim, hidden_dim, config, vb.clone(), false)?;
    Ok(LSTM {
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

// -----------------------------------------------------------------------------

impl LSTM {
    // pub fn zero_state(&self, batch_size: usize) -> Result<LSTMState> {
    //     let z =
    //         Tensor::zeros((batch_size, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;

    //     Ok(LSTMState { h: z.clone(), c: z })
    // }

    pub fn forward(
        &self,
        input: &Tensor,
        init_state: Option<&LSTMState>,
    ) -> Result<Vec<LSTMState>> {
        let (d1, d2, _features) = input.dims3()?;
        let (seq_len, batch_szie) = if self.config.batch_first {
            (d2, d1)
        } else {
            (d1, d2)
        };

        let get = if self.config.batch_first {
            get_tensor_for_batch_first
        } else {
            get_tensor_for_seq_first
        };

        let w_ih = self.w_ih.t()?;
        let w_hh = self.w_hh.t()?;

        let input = compute_input_layer(input, &w_ih, self.b_ih.as_ref())?;

        let mut states = Vec::with_capacity(seq_len + 1);
        states.push(if let Some(init_state) = init_state {
            init_state.clone()
        } else {
            LSTMState::zero(batch_szie, self.hidden_dim, self.dtype, &self.device)?
        });

        for i in 0..seq_len {
            let h = &states[i].h;
            let c = &states[i].c;

            let hh = h.matmul(&w_hh)?;
            let hh = if let Some(b_hh) = &self.b_hh {
                hh.broadcast_add(b_hh)?
            } else {
                hh
            };

            let (next_h, next_c) = lstm_cell(&get(&input, i)?.broadcast_add(&hh)?, c)?;

            states.push(LSTMState {
                c: next_c,
                h: next_h,
            });
        }

        Ok(states.into_iter().skip(1).collect())
    }

    pub fn states_to_tensor(&self, states: &[LSTMState]) -> Result<Tensor> {
        let output = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();

        if self.config.batch_first {
            Tensor::stack(&output, 1)
        } else {
            Tensor::stack(&output, 0)
        }
    }
}

pub struct BiLSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    w_ih_reverse: Tensor,
    w_hh_reverse: Tensor,
    b_ih_reverse: Option<Tensor>,
    b_hh_reverse: Option<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

pub fn bilstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: VarBuilder,
) -> Result<BiLSTM> {
    let (w_ih, w_hh, b_ih, b_hh) = gen_lstm(in_dim, hidden_dim, config, vb.clone(), false)?;
    let (w_ih_reverse, w_hh_reverse, b_ih_reverse, b_hh_reverse) =
        gen_lstm(in_dim, hidden_dim, config, vb.clone(), true)?;

    Ok(BiLSTM {
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        w_ih_reverse,
        w_hh_reverse,
        b_ih_reverse,
        b_hh_reverse,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

impl BiLSTM {
    pub fn forward(
        &self,
        input: &Tensor,
        init_state: Option<(&LSTMState, &LSTMState)>,
    ) -> Result<Vec<(LSTMState, LSTMState)>> {
        let (d1, d2, _features) = input.dims3()?;
        let (seq_len, batch_szie) = if self.config.batch_first {
            (d2, d1)
        } else {
            (d1, d2)
        };

        let get = if self.config.batch_first {
            get_tensor_for_batch_first
        } else {
            get_tensor_for_seq_first
        };

        let w_ih = self.w_ih.t()?;
        let w_hh = self.w_hh.t()?;

        let input_f = compute_input_layer(input, &w_ih, self.b_ih.as_ref())?;

        let mut forward_states = Vec::with_capacity(seq_len + 1);
        forward_states.push(if let Some((forward_state, _)) = init_state {
            forward_state.clone()
        } else {
            LSTMState::zero(batch_szie, self.hidden_dim, self.dtype, &self.device)?
        });

        for i in 0..seq_len {
            let h = &forward_states[i].h;
            let c = &forward_states[i].c;

            let hh = h.matmul(&w_hh)?;
            let hh = if let Some(b_hh) = &self.b_hh {
                hh.broadcast_add(b_hh)?
            } else {
                hh
            };

            let (next_h, next_c) = lstm_cell(&get(&input_f, i)?.broadcast_add(&hh)?, c)?;

            forward_states.push(LSTMState {
                c: next_c,
                h: next_h,
            });
        }

        let w_ih_reverse = self.w_ih_reverse.t()?;
        let w_hh_reverse = self.w_hh_reverse.t()?;

        let input_b = compute_input_layer(input, &w_ih_reverse, self.b_ih_reverse.as_ref())?;

        let mut backward_states = Vec::with_capacity(seq_len + 1);
        backward_states.push(if let Some((_, backward_state)) = init_state {
            backward_state.clone()
        } else {
            LSTMState::zero(batch_szie, self.hidden_dim, self.dtype, &self.device)?
        });

        let seq_len_minus_1 = seq_len - 1;
        for i in 0..=seq_len_minus_1 {
            let h = &backward_states[i].h;
            let c = &backward_states[i].c;

            let hh = h.matmul(&w_hh_reverse)?;
            let hh = if let Some(b_hh_reverse) = &self.b_hh_reverse {
                hh.broadcast_add(b_hh_reverse)?
            } else {
                hh
            };

            let (next_h, next_c) =
                lstm_cell(&get(&input_b, seq_len_minus_1 - i)?.broadcast_add(&hh)?, c)?;

            backward_states.push(LSTMState {
                c: next_c,
                h: next_h,
            });
        }

        let outputs = forward_states
            .into_iter()
            .skip(1)
            .zip(backward_states.into_iter().skip(1).rev())
            .collect::<Vec<_>>();

        Ok(outputs)
    }

    pub fn states_to_tensor(&self, states: &[(LSTMState, LSTMState)]) -> Result<Tensor> {
        let output = states
            .iter()
            .map(|(f, b)| Tensor::cat(&[&f.h, &b.h], 1).unwrap())
            .collect::<Vec<_>>();

        if self.config.batch_first {
            Tensor::stack(&output, 1)
        } else {
            Tensor::stack(&output, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Ok, Result};
    use candle_core::{Tensor, D};
    use candle_nn::RNN;

    const IN_DIM: usize = 768;
    const HIDDEN_DIM: usize = 768;
    const SEQ_LEN: usize = 256;
    const BATCH_SIZE: usize = 1;

    fn assert_tensor(a: &Tensor, b: &Tensor, dim: usize, v: f32) -> Result<()> {
        assert_eq!(a.dims(), b.dims());
        let mut t = (a - b)?.abs()?;

        for _i in 0..dim {
            t = t.max(D::Minus1)?;
        }

        let t = t.to_scalar::<f32>()?;
        assert!(t < v);
        Ok(())
    }

    #[test]
    fn load_lstm() -> Result<()> {
        let vb = VarBuilder::from_pth("lstm_test.pt", DType::F32, &Device::Cpu)?;
        lstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb)?;
        Ok(())
    }

    #[test]
    fn test_lstm() -> Result<()> {
        let vb = VarBuilder::from_pth("lstm_test.pt", DType::F32, &Device::Cpu)?;
        let lstm = lstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb.clone())?;

        let input = vb.get((SEQ_LEN, BATCH_SIZE, IN_DIM), "input")?;
        let output = vb.get((SEQ_LEN, BATCH_SIZE, HIDDEN_DIM), "output")?;

        let start = std::time::Instant::now();
        let states = lstm.forward(&input, None)?;
        let result = lstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("fast lstm elapsed: {:?}", elapsed.as_secs_f32());

        assert_tensor(&result, &output, 3, 1e-5)?;

        let candle_lstm = candle_nn::lstm(
            IN_DIM,
            HIDDEN_DIM,
            candle_nn::LSTMConfig::default(),
            vb.clone(),
        )?;

        let input = input.transpose(0, 1)?;
        let start = std::time::Instant::now();
        let states = candle_lstm.seq(&input)?;
        let result = candle_lstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("candle lstm elapsed: {:?}", elapsed.as_secs_f32());
        let result = result.transpose(0, 1)?;

        assert_tensor(&result, &output, 3, 1e-5)?;

        Ok(())
    }

    #[test]
    fn test_batch_first_lstm() -> Result<()> {
        let mut config = LSTMConfig::default();
        config.batch_first = true;

        let vb = VarBuilder::from_pth("lstm_test_batch_first.pt", DType::F32, &Device::Cpu)?;
        let lstm = lstm(IN_DIM, HIDDEN_DIM, config, vb.clone())?;

        let input = vb.get((BATCH_SIZE, SEQ_LEN, IN_DIM), "input")?;
        let output = vb.get((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), "output")?;

        let start = std::time::Instant::now();
        let states = lstm.forward(&input, None)?;
        let result = lstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("fast lstm elapsed: {:?}", elapsed.as_secs_f32());

        assert_tensor(&result, &output, 3, 1e-5)?;

        let candle_lstm = candle_nn::lstm(
            IN_DIM,
            HIDDEN_DIM,
            candle_nn::LSTMConfig::default(),
            vb.clone(),
        )?;

        let start = std::time::Instant::now();
        let states = candle_lstm.seq(&input)?;
        let result = candle_lstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("candle lstm elapsed: {:?}", elapsed.as_secs_f32());

        assert_tensor(&result, &output, 3, 1e-5)?;

        Ok(())
    }

    #[test]
    fn load_bilstm() -> Result<()> {
        let vb = VarBuilder::from_pth("bi_lstm_test.pt", DType::F32, &Device::Cpu)?;
        bilstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb)?;
        Ok(())
    }

    #[test]
    fn test_bilstm() -> Result<()> {
        let vb = VarBuilder::from_pth("bi_lstm_test.pt", DType::F32, &Device::Cpu)?;
        let bilstm = bilstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb.clone())?;

        let input = vb.get((SEQ_LEN, BATCH_SIZE, IN_DIM), "input")?;
        let output = vb.get((SEQ_LEN, BATCH_SIZE, HIDDEN_DIM * 2), "output")?;

        let f_init_state = LSTMState::zero(BATCH_SIZE, HIDDEN_DIM, input.dtype(), input.device())?;
        let b_init_state = f_init_state.clone();
        let init_state = Some((&f_init_state, &b_init_state));

        let start = std::time::Instant::now();
        let states = bilstm.forward(&input, init_state)?;
        let result = bilstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("fast bilstm elapsed: {:?}", elapsed.as_secs_f32());

        assert_tensor(&result, &output, 3, 1e-5)?;

        Ok(())
    }

    #[test]
    fn test_batch_first_bilstm() -> Result<()> {
        let mut config = LSTMConfig::default();
        config.batch_first = true;

        let vb = VarBuilder::from_pth("bi_lstm_test_batch_first.pt", DType::F32, &Device::Cpu)?;
        let bilstm = bilstm(IN_DIM, HIDDEN_DIM, config, vb.clone())?;

        let input = vb.get((BATCH_SIZE, SEQ_LEN, IN_DIM), "input")?;
        let output = vb.get((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM * 2), "output")?;

        let start = std::time::Instant::now();
        let states = bilstm.forward(&input, None)?;
        let result = bilstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("fast bilstm elapsed: {:?}", elapsed.as_secs_f32());

        assert_tensor(&result, &output, 3, 1e-5)?;

        Ok(())
    }
}
