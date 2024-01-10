use crate::Layer;
use crate::Mat;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Dense {
    weights: Mat,
    bias: Mat,
    input: Option<Mat>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Dense {
            weights: ndarray::Array::random(
                (input_size, output_size),
                Uniform::<f64>::new(-0.5, 0.5),
            ),
            bias: ndarray::Array::random((1, output_size), Uniform::<f64>::new(-0.5, 0.5)),
            input: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Mat) -> Mat {
        self.input = Some(input.clone());
        input.dot(&self.weights) + &self.bias
    }

    fn backward(&mut self, error: &Mat, learning_rate: f64) -> Mat {
        let weights_error = self
            .input
            .as_ref()
            .expect("Need to forward propagate first.")
            .t()
            .dot(error);

        // Apparently bias becomes shape (batch_size, input_size) when subtracted from itself via error
        // So need to first create another (input_size, 1) matrix of the correct shape to use as the subtractor
        // TODO: can this be simplified?
        let bias_error = error
            .t()
            .dot(&ndarray::Array2::ones(((error.shape())[0], 1)));

        self.weights = &self.weights - learning_rate * &weights_error;
        self.bias = &self.bias - learning_rate * &bias_error.t();

        error.dot(&self.weights.t())
    }
}
