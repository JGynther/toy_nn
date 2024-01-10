use crate::Mat;

pub trait Loss {
    fn loss(&self, y_pred: &Mat, y_true: &Mat) -> f64;
    fn loss_prime(&self, y_pred: &Mat, y_true: &Mat) -> Mat;
}

pub struct MSE {}

impl Loss for MSE {
    fn loss(&self, y_pred: &Mat, y_true: &Mat) -> f64 {
        let diff = y_pred - y_true;
        diff.mapv(|x| x * x).mean().expect("Error calculating MSE")
    }

    fn loss_prime(&self, y_pred: &Mat, y_true: &Mat) -> Mat {
        2.0 * (y_pred - y_true) / y_pred.len() as f64
    }
}
