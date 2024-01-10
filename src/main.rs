use ndarray_rand::{rand_distr::Uniform, RandomExt};

use toy_nn::activation::ReLU;
use toy_nn::dense::Dense;
use toy_nn::loss::MSE;
use toy_nn::sequential::Sequential;
use toy_nn::Layer;

fn main() {
    let mut model = Sequential::new(MSE {});

    model.add(Dense::new(1, 128));
    model.add(ReLU::new());
    model.add(Dense::new(128, 128));
    model.add(ReLU::new());
    model.add(Dense::new(128, 128));
    model.add(ReLU::new());
    model.add(Dense::new(128, 1));

    let epochs = 10_000;
    let learning_rate = 0.0001;
    let batch_size = 16;

    for i in 0..epochs {
        let x = ndarray::Array::random(
            (batch_size, 1),
            Uniform::<f64>::new(-2. * std::f64::consts::PI, 2. * std::f64::consts::PI),
        );
        let y = x.mapv(|x| f64::sin(x));

        let output = model.forward(&x);
        let error = model.loss_fn.loss_prime(&output, &y);
        model.backward(&error, learning_rate);

        if i % 1000 == 99 {
            let loss = model.loss_fn.loss(&output, &y);
            println!("Epoch {}: Loss = {:?}", i + 1, loss);
        }
    }

    let mut plot = plotly::Plot::new();
    let x = ndarray::Array::linspace(-4. * std::f64::consts::PI, 4. * std::f64::consts::PI, 1000);
    let y = x.mapv(|x| f64::sin(x));

    let x_ = x.clone().into_shape((1000, 1)).expect("");
    let output = model.forward(&x_);

    let trace =
        plotly::Scatter::new(x.clone().into_raw_vec(), output.into_raw_vec()).name("Prediction");
    plot.add_trace(trace);

    let trace = plotly::Scatter::from_array(
        ndarray::Array::from_iter(x.iter().cloned()),
        ndarray::Array::from_iter(y.iter().cloned()),
    )
    .name("y=sin(x)");

    plot.add_trace(trace);
    plot.show()
}
