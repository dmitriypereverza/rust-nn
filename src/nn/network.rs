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
    layers: Vec<(usize, Activation<'a>)>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

impl Network<'_> {
    pub fn new(layers: Vec<(usize, Activation)>, learning_rate: f64) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i].0, layers[i + 1].0, -1.0, 1.0));
            biases.push(Matrix::random(1, layers[i + 1].0, -1.0, 1.0));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        inputs: &Vec<Vec<Vec<f64>>>,
        targets: &Vec<Vec<Vec<f64>>>,
        epochs: usize,
    ) {
        for i in 1..=epochs {
            let error = self.train_one_epoch(&inputs, &targets, self.learning_rate);
            if epochs < 100 || i % (epochs / 20) == 0 {
                println!("Loss: {:.7}, Epoch {} of {}", error, i, epochs);
            }
        }
    }

    pub fn train_one_epoch(
        &mut self,
        inputs: &Vec<Vec<Vec<f64>>>,
        targets: &Vec<Vec<Vec<f64>>>,
        learning_rate: f64,
    ) -> f64 {
        let mut error: f64 = 0.0;
        for i in 0..inputs.len() {
            let next_input = inputs.get(i).unwrap();
            let next_target = targets.get(i).unwrap();
            let outputs = self.feed_forward(next_input.clone());

            let current_target = next_target.clone();
            error += self.calculate_error(&outputs, &current_target);
            self.back_propagate(outputs, current_target, learning_rate);
        }
        error / inputs.len() as f64
    }

    pub fn feed_forward(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut current = Matrix::from(inputs.clone());

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = current
                .dot_product(&self.weights[i])
                .add(&self.biases[i])
                .map(self.layers[i].1.function);

            self.data.push(current.clone());
        }

        current.data.to_owned()
    }

    pub fn calculate_error(&self, outputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
        let parsed = Matrix::from(outputs.clone());
        let errors = Matrix::from(targets.clone()).subtract(&parsed).square();

        return errors.clone().collect_sum() / errors.count() as f64;
    }

    pub fn back_propagate(
        &mut self,
        outputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        learning_rate: f64,
    ) {
        if targets[0].len() != self.layers[self.layers.len() - 1].0 {
            panic!("Invalid targets length");
        }

        /*
           +---+     +---+   +---+ +---+     +---+   +---+
           | x |-----| t |---| h | | x |-----| t |---| h |
           +---+ / \ +---+   +---+ +---+ / \ +---+   +---+
                /   \                   /   \
               /     \                 /     \
           +---+     +---+         +---+     +---+
           | w |     | b |         | w |     | b |
           +---+     +---+         +---+     +---+
        */

        let targets_matrix = Matrix::from(targets);
    
        let mut de_dt = Matrix::from(outputs).subtract(&targets_matrix);

        for i in (0..self.layers.len() - 1).rev() {
            let de_dw = self.data[i].transpose().dot_product(&de_dt);
            let de_db = de_dt.sum_by_axis(0);

            self.weights[i] =
                self.weights[i].subtract(&de_dw.map(&|x| x * learning_rate));
            self.biases[i] = self.biases[i].subtract(&de_db.map(&|x| x * learning_rate));

            let de_dh = (&de_dt).dot_product(&self.weights[i].transpose());
            de_dt = de_dh.map(self.layers[i].1.derivative);
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
