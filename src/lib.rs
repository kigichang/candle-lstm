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

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{init, Init, VarBuilder};

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
    todo!()
}

impl LSTM {
    pub fn forward(
        &self,
        input: &Tensor,
        init_state: Option<&LSTMState>,
    ) -> Result<Vec<LSTMState>> {
        todo!()
    }

    pub fn states_to_tensor(&self, states: &[LSTMState]) -> Result<Tensor> {
        todo!()
    }
}

pub struct BiLSTM {
    forward_lstm: LSTM,
    backward_lstm: LSTM,
}

pub fn bilstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: VarBuilder,
) -> Result<BiLSTM> {
    todo!()
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
