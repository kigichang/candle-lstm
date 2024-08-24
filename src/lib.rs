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

fn get_tensor_for_batch_first(input: &Tensor, index: usize) -> Result<Tensor> {
    input.i((.., index, ..))?.contiguous()
}

fn get_tensor_for_seq_first(input: &Tensor, index: usize) -> Result<Tensor> {
    input.get(index)
}

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
    Ok((w_ih.t()?, w_hh.t()?, b_ih, b_hh))
}

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
}

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

impl LSTM {
    pub fn zero_state(&self, batch_size: usize) -> Result<LSTMState> {
        let z =
            Tensor::zeros((batch_size, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;

        Ok(LSTMState { h: z.clone(), c: z })
    }

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

        let input = self.compute_input_layer(input)?;
        let get = if self.config.batch_first {
            get_tensor_for_batch_first
        } else {
            get_tensor_for_seq_first
        };

        let mut states = Vec::with_capacity(seq_len + 1);
        states.push(if let Some(init_state) = init_state {
            init_state.clone()
        } else {
            self.zero_state(batch_szie)?
        });

        for i in 0..seq_len {
            let h = &states[i].h;
            let c = &states[i].c;

            let hh = h.matmul(&self.w_hh)?;
            let hh = if let Some(b_hh) = &self.b_hh {
                hh.broadcast_add(b_hh)?
            } else {
                hh
            };

            let chunks = &get(&input, i)?.broadcast_add(&hh)?.chunk(4, 1)?;
            let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
            let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
            let cell_gate = chunks[2].tanh()?;
            let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;

            let next_c = ((forget_gate * c)? + (in_gate * cell_gate)?)?;
            let next_h = (out_gate * next_c.tanh()?)?;
            states.push(LSTMState {
                c: next_c,
                h: next_h,
            });
        }

        Ok(states.into_iter().skip(1).collect())
    }

    fn compute_input_layer(&self, input: &Tensor) -> Result<Tensor> {
        let (d1, d2, _features) = input.dims3()?;
        if d1 > d2 {
            let mut output: Vec<Tensor> = Vec::with_capacity(d2);
            for i in 0..d2 {
                let x = input.i((.., i, ..))?;
                let x = x.matmul(&self.w_ih)?;
                output.push(if let Some(b_ih) = &self.b_ih {
                    x.broadcast_add(b_ih)?
                } else {
                    x
                });
            }

            Tensor::stack(&output, 1)
        } else {
            let mut output: Vec<Tensor> = Vec::with_capacity(d1);
            for i in 0..d2 {
                let x = input.i((.., i, ..))?;
                let x = x.matmul(&self.w_ih)?;
                output.push(if let Some(b_ih) = &self.b_ih {
                    x.broadcast_add(b_ih)?
                } else {
                    x
                });
            }
            Tensor::stack(&output, 0)
        }
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
    ) -> Result<Vec<LSTMState>> {
        todo!()
    }

    pub fn states_to_tensor(&self, states: &[(LSTMState, LSTMState)]) -> Result<Tensor> {
        todo!()
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

        let input = vb.get((SEQ_LEN, 1, IN_DIM), "input")?;
        let output = vb.get((SEQ_LEN, 1, HIDDEN_DIM), "output")?;

        let start = std::time::Instant::now();
        let states = lstm.forward(&input, None)?;
        let result = lstm.states_to_tensor(&states)?;
        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed.as_secs_f32());

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
        println!("Elapsed: {:?}", elapsed.as_secs_f32());
        let result = result.transpose(0, 1)?;

        assert_tensor(&result, &output, 3, 1e-5)?;

        Ok(())
    }

    #[test]
    fn load_bilstm() -> Result<()> {
        let vb = VarBuilder::from_pth("bi_lstm_test.pt", DType::F32, &Device::Cpu)?;
        bilstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb)?;
        Ok(())
    }
}
