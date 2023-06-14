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
    let inputs: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0]],
        vec![vec![5.0]],
        vec![vec![0.0]],
        vec![vec![1.0]],
    ];
    let targets: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![0.0]],
        vec![vec![10.0]],
        vec![vec![0.0]],
        vec![vec![2.0]],
    ];

    #[rustfmt::skip]
    let nn_architecture: Vec<(usize, Activation)> = vec![
        (1, IDENTITY), 
        (1, IDENTITY)
    ];
    let learning_rate = 0.01;
    let mut network = Network::new(nn_architecture.clone(), learning_rate.clone());

    let mut error: f64 = 0.0;
    for _ in 0..100 {
        error = network.train_one_epoch(&inputs.clone(), &targets.clone(), learning_rate);
        println!("Err: {}", error);
    }

    
    println!("\nResult:");
    let test = inputs.clone();
    for i in test.iter() {
        let out = network.feed_forward(i.clone());
        println!("{:?} = {:?}", i, out);
    }

}
