#![feature(iterator_try_collect)]

use std::{error::Error, path::Path};

use img::Img;
use load_save::{Loadable, Savable};
use neural::NeuralNetwork;
use vector::Vector;

mod dot_product;
mod img;
mod load_save;
mod matrix;
mod neural;
mod vector;

pub fn load_imgs<P: AsRef<Path>>(
    csv_path: P,
    take_count: usize,
) -> Result<Vec<Img>, Box<dyn Error>> {
    let imgs_loader = Img::load_from_csv(csv_path)?;
    let imgs: Vec<Img> = imgs_loader.take(take_count).try_collect()?;
    Ok(imgs)
}

pub fn train<P: AsRef<Path>>(
    csv_path: P,
    max_count: usize,
    neural_network: &mut NeuralNetwork,
) -> Result<(), Box<dyn Error>> {
    println!("Loading images...");

    let train_batch = load_imgs(csv_path, max_count)?;

    println!("Loaded {} images", train_batch.len());

    neural_network.train_batch(train_batch)?;

    Ok(())
}

pub fn test<P: AsRef<Path>>(
    csv_path: P,
    take_count: usize,
    neural_network: &NeuralNetwork,
) -> Result<f64, Box<dyn Error>> {
    let test_batch = load_imgs(csv_path, take_count)?;
    let prediction = neural_network.test_prediction(test_batch, |output, reference| {
        let output_max = output.max_arg();
        let reference_max = reference.max_arg();
        if output_max == reference_max {
            1.0
        } else {
            0.0
        }
    })?;
    Ok(prediction)
}

fn main() -> Result<(), Box<dyn Error>> {

    let mut neural_network = NeuralNetwork::new(784, 300, 10, 0.01);

    train("data/mnist_train.csv", usize::MAX, &mut neural_network)?;

    neural_network.save("network_save/neural_network.json")?;

    let prediction = test("data/mnist_test.csv", usize::MAX, &neural_network)?;

    println!("{}", prediction);

    Ok(())
}
