use crate::Layer;
use crate::Mat;

pub trait Activation {
    fn activation(x: &Mat) -> Mat;
    fn activation_prime(x: &Mat) -> Mat;
}

pub struct ReLU {
    input: Option<Mat>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { input: None }
    }

    fn activation(x: &Mat) -> Mat {
        x.mapv(|x| x.max(0.))
    }

    fn activation_prime(x: &Mat) -> Mat {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &Mat) -> Mat {
        self.input = Some(x.clone());
        ReLU::activation(x)
    }

    fn backward(&mut self, x: &Mat, _learning_rate: f64) -> Mat {
        ReLU::activation_prime(
            &self
                .input
                .as_ref()
                .expect("Need to forward propagate first."),
        ) * x
    }
}
