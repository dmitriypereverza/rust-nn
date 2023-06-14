pub mod nn {
    pub mod activations;
    pub mod matrix;
    pub mod network;
}
mod image_nn;

use nn::activations::SIGMOID;
use nn::network::Network;

use crate::nn::activations::{Activation, IDENTITY, RELU};

fn main() {
    let inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let targets: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
    ];

    #[rustfmt::skip]
    let nn_architecture: Vec<(usize, Activation)> = vec![
        (2, SIGMOID), 
        (2, SIGMOID),
        (1, SIGMOID)
    ];
    let learning_rate = 0.001;
    let mut network = Network::new(nn_architecture.clone(), learning_rate.clone());

    let mut error: f64 = 0.0;
    for _ in 0..10000 {
        error = network.train_one_epoch(&vec![inputs.clone()], &vec![targets.clone()], learning_rate);
        println!("Err: {}", error);
    }

    let out = network.feed_forward(inputs.clone());

    for i in 0..inputs.len() {
        println!("{} {} = {}", inputs[i][0], inputs[i][1], out[i][0]);
    }

}
