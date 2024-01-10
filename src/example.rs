use crate::{sequential::Sequential, Layer, Mat};
use ndarray::{Array, Array1};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use plotly::{Plot, Scatter};

pub fn rand_sine_2d(start: f64, stop: f64, n: usize) -> (Mat, Mat) {
    let x = Array::random((n, 1), Uniform::<f64>::new(start, stop));
    let y = x.mapv(|x| f64::sin(x));
    (x, y)
}

fn linspace_sine(start: f64, stop: f64, n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array::linspace(start, stop, n);
    let y = x.mapv(|x| f64::sin(x));
    (x, y)
}

pub fn plot_sine_example(start: f64, stop: f64, n: usize, model: &mut Sequential) {
    let (x, y) = linspace_sine(start, stop, n);

    let x_2d = x
        .clone()
        .into_shape((n, 1))
        .expect("Can not cast to shape.");

    let output = model.forward(&x_2d);
    let output_1d = output.into_shape(n).expect("Can not cast to shape.");

    let trace_pred = Scatter::from_array(x.clone(), output_1d).name("Prediction");
    let trace_true = Scatter::from_array(x, y).name("y=sin(x)");

    let mut plot = Plot::new();
    plot.add_trace(trace_pred);
    plot.add_trace(trace_true);
    plot.show()
}
