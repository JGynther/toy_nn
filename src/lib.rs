pub mod activation;
pub mod dense;
pub mod loss;
pub mod sequential;

extern crate blas_src;

type Mat = ndarray::Array2<f64>;

pub trait Layer {
    fn forward(&mut self, input: &Mat) -> Mat;
    fn backward(&mut self, error: &Mat, learning_rate: f64) -> Mat;
}
