use toy_nn::activation::ReLU;
use toy_nn::dense::Dense;
use toy_nn::loss::MSE;
use toy_nn::sequential::Sequential;
use toy_nn::Layer;

// Example helper functions
use toy_nn::example::{plot_sine_example, rand_sine_2d};

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

    let start = 0.;
    let end = 2. * std::f64::consts::PI;

    for i in 0..epochs {
        let (x, y) = rand_sine_2d(start, end, batch_size);

        let output = model.forward(&x);
        let error = model.loss_fn.loss_prime(&output, &y);
        model.backward(&error, learning_rate);

        if i % 1000 == 99 {
            let loss = model.loss_fn.loss(&output, &y);
            println!("Epoch {}: Loss = {:?}", i + 1, loss);
        }
    }

    // Plot model predictions
    plot_sine_example(start, end, 1000, &mut model)
}
