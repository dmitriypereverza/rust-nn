use macroquad::prelude::*;
use std::process::exit;

pub mod nn {
    pub mod activations;
    pub mod matrix;
    pub mod network;
}
mod image_nn;

use crate::nn::activations::{RELU, TANH};
use nn::activations::SIGMOID;
use nn::network::Network;

pub struct Vector2 {
    x: f32,
    y: f32,
}

pub fn vector2_distance(first: &Vector2, second: &Vector2) -> f32 {
    return f32::sqrt((first.x - second.x).powi(2) + (first.y - second.y).powi(2));
}

pub fn lerp(start: f64, end: f64, amt: f64) -> f64 {
    return (1.0 - amt) * start + amt * end;
}

#[macroquad::main("Neural network gym.")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1);
    match image_path {
        None => {
            eprintln!("[ERROR] Image path argument not found.");
            std::process::exit(1);
        }
        Some(_x) => {}
    }

    let mut inputs: Vec<Vec<f64>> = vec![];
    let mut targets: Vec<Vec<f64>> = vec![];
    let mut image_width: f32 = 0.0;
    let mut image_height: f32 = 0.0;
    image_nn::load_image(
        image_path.unwrap(),
        &mut inputs,
        &mut targets,
        &mut image_width,
        &mut image_height,
    );

    // println!("inputs = {:?}", inputs);
    // println!("targets = {:?}", targets);
    // println!("image_width = {:#?}", image_width);
    // println!("image_height = {:#?}", image_height);
    // exit(0);

    let mut learnig_rate_slider_dragging = false;
    let mut learnig_rate_slider_value = 0.4;

    let mut learning_rate = 0.1;
    let nn_architecture = vec![2, 40, 10, 1];
    let mut network = Network::new(nn_architecture.clone(), learning_rate.clone(), SIGMOID);
    let mut errors: Vec<f64> = vec![];
    loop {
        if is_key_pressed(KeyCode::Space) {
            network = Network::new(nn_architecture.clone(), learning_rate.clone(), SIGMOID);
            errors = vec![];
        }
        let mut current_error = 0.0;
        for _epoch in 0..7 {
            current_error = network.train_one_epoch(&inputs, &targets, learning_rate);
        }

        // println!("weights = {:?}", network.weights);
        // println!("biases = {:?}", network.biases);

        errors.push(current_error);

        clear_background(Color::new(0.18, 0.18, 0.18, 1.0));
        draw_text("IT WORKS!", 20.0, 20.0, 30.0, DARKGRAY);
        draw_text(
            &format!("Error: = {:.9}", current_error.to_string()),
            20.0,
            50.0,
            30.0,
            DARKGRAY,
        );
        draw_text(
            &format!("Learning rate: {}", &learning_rate),
            20.0,
            80.0,
            30.0,
            DARKGRAY,
        );

        // Error slider
        let slider_width = screen_width() / 2.0 - 20.0;
        let slider_position = Vector2 { x: 20.0, y: 120.0 };
        draw_line(
            slider_position.x,
            slider_position.y,
            slider_position.x + slider_width,
            slider_position.y,
            1.0,
            DARKGRAY,
        );

        let slider_circle_position = Vector2 {
            x: slider_position.x + slider_width * learnig_rate_slider_value as f32,
            y: slider_position.y,
        };
        let mouse_position = Vector2 {
            x: mouse_position().0,
            y: mouse_position().1,
        };

        learnig_rate_slider_dragging = is_mouse_button_down(MouseButton::Left)
            && vector2_distance(&slider_circle_position, &mouse_position) > 6.0;

        if (learnig_rate_slider_dragging) {
            learnig_rate_slider_value =
                ((mouse_position.x - slider_position.x) / slider_width).clamp(0.0001, 1.0) as f64;

            learning_rate = lerp(0.02, 0.2, learnig_rate_slider_value);
        }
        draw_circle(slider_circle_position.x, slider_circle_position.y, 6.0, RED);

        // ERROR PLOT
        let plot_padding_top = screen_height() / 4.0;
        let plot_height = screen_height() / 2.0;
        let plot_width = screen_width() / 2.0 - 30.0;
        let max_error = errors.clone().into_iter().reduce(f64::max).unwrap() as f32;
        let error_offset = plot_width / errors.len() as f32;
        draw_rectangle(0.0, plot_padding_top, plot_width, plot_height, DARKGRAY);
        for i in 0..errors.len() - 1 {
            let err = errors[i] as f32;
            let err_2 = errors[i + 1] as f32;
            draw_line(
                i as f32 * error_offset,
                plot_padding_top + plot_height - err / max_error * plot_height,
                (i + 1) as f32 * error_offset,
                plot_padding_top + plot_height - err_2 / max_error * plot_height,
                1.0,
                RED,
            );
        }

        let image_scale = 8.0;

        // IMAGE ORIGINAL VIEW
        let left_images_padding = screen_width() - image_width * image_scale - 60.0;
        let mut top_images_padding = 60.0;
        draw_text(
            "IMAGE ORIGINAL VIEW",
            left_images_padding,
            top_images_padding,
            30.0,
            DARKGRAY,
        );
        top_images_padding += 10.0;
        for sample_index in 0..inputs.len() {
            let input = inputs.get(sample_index);
            let position = input.unwrap();
            let target = targets.get(sample_index);
            let brightness = target.unwrap()[0] as f32;
            draw_rectangle(
                left_images_padding + position[0] as f32 * image_width * image_scale,
                top_images_padding + position[1] as f32 * image_height * image_scale,
                image_scale,
                image_scale,
                Color::new(brightness, brightness, brightness, 1.0),
            )
        }

        // IMAGE FROM NN VIEW
        top_images_padding += image_scale * image_height + 30.0;
        draw_text(
            "IMAGE FROM NN VIEW",
            left_images_padding,
            top_images_padding,
            30.0,
            DARKGRAY,
        );
        top_images_padding += 10.0;
        for sample_index in 0..inputs.len() {
            let input = inputs.get(sample_index);
            let position = input.unwrap();
            let target = network.feed_forward(position)[0] as f32;
            draw_rectangle(
                left_images_padding + position[0] as f32 * image_width * image_scale,
                top_images_padding + position[1] as f32 * image_height * image_scale,
                image_scale,
                image_scale,
                Color::new(target, target, target, 1.0),
            )
        }

        next_frame().await
    }
}
