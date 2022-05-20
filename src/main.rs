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
// mod neural_old;
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
) -> Result<usize, Box<dyn Error>> {
    println!("Loading images...");

    let train_batch = load_imgs(csv_path, max_count)?;

    let size = train_batch.len();

    println!("Loaded {} images", size);

    neural_network.train_batch(train_batch)?;

    Ok(size)
}

pub fn test<P: AsRef<Path>>(
    csv_path: P,
    take_count: usize,
    neural_network: &NeuralNetwork,
) -> Result<f64, Box<dyn Error>> {
    let test_batch = load_imgs(csv_path, take_count)?;
    let prediction = neural_network.predict_batch(test_batch, |output, reference| {
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
    let mut neural_network = NeuralNetwork::new(784, [500], 10, 0.2);

    let start = std::time::Instant::now();

    let batch_size = train("data/mnist_train.csv", usize::MAX, &mut neural_network)?;

    let elapsed = start.elapsed();

    println!("Training took {} seconds", elapsed.as_secs());

    println!(
        "Average training time per image: {} ms",
        elapsed.as_secs_f64() / (batch_size / 1000) as f64
    );

    neural_network.save("network_save/neural_network.json")?;

    let prediction = test("data/mnist_test.csv", usize::MAX, &neural_network)?;

    println!("{}", prediction);

    Ok(())
}
