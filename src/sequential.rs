use crate::loss::Loss;
use crate::Layer;
use crate::Mat;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    pub loss_fn: Box<dyn Loss>,
}

impl Sequential {
    pub fn new<L: Loss + 'static>(loss_fn: L) -> Self {
        Self {
            loss_fn: Box::new(loss_fn),
            layers: Vec::new(),
        }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Mat) -> Mat {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    fn backward(&mut self, error: &Mat, learning_rate: f64) -> Mat {
        let mut error = error.clone();
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, learning_rate);
        }
        error
    }
}
