use std::process::exit;
use std::{
    fs::File,
    io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{activations::Activation, matrix::Matrix};

pub struct Network<'a> {
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub data: Vec<Matrix>,
    pub learning_rate: f64,
    layers: Vec<usize>,
    activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

fn from_minus_one_to_one(random_value: f64) -> f64 {
    random_value * 2.0 - 1.0
}
fn from_zero_to_one(random_value: f64) -> f64 {
    random_value
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, learning_rate: f64, activation: Activation) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(
                layers[i + 1],
                layers[i],
                &from_minus_one_to_one,
            ));
            biases.push(Matrix::random(layers[i + 1], 1, &from_minus_one_to_one));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
            activation,
        }
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: u16) {
        for i in 1..=epochs {
            let error = self.train_one_epoch(&inputs, &targets, self.learning_rate);
            if epochs < 100 || i % (epochs / 20) == 0 {
                println!("Loss: {:.7}, Epoch {} of {}", error, i, epochs);
            }
        }
    }

    pub fn train_one_epoch(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) -> f64 {
        let mut error: f64 = 0.0;
        for j in 0..inputs.len() {
            let outputs = self.feed_forward(&inputs[j].clone());

            let current_target = targets[j].clone();
            error += self.calculate_error(&outputs, &current_target);
            self.back_propagate(outputs, current_target, learning_rate);
        }
        error / inputs.len() as f64
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid inputs length");
        }

        let mut current = Matrix::from(vec![inputs.to_vec()]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_product(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }

        current.transpose().data[0].to_owned()
    }

    pub fn calculate_error(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> f64 {
        let parsed = Matrix::from(vec![outputs.to_vec()]).transpose();
        let errors = Matrix::from(vec![targets.to_vec()])
            .transpose()
            .subtract(&parsed);

        return errors.clone().pow().collect_sum() / errors.count() as f64;
    }

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>, learning_rate: f64) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalid targets length");
        }

        let parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = Matrix::from(vec![targets]).transpose().subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .scalar_multiplication(&errors)
                .map(&|x| x * learning_rate);

            self.weights[i] =
                self.weights[i].add(&gradients.dot_product(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().dot_product(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to touch save file");

        file.write_all(
			json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>()
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to open save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..self.layers.len() - 1 {
            weights.push(Matrix::from(save_data.weights[i].clone()));
            biases.push(Matrix::from(save_data.biases[i].clone()));
        }

        self.weights = weights;
        self.biases = biases;
    }
}
