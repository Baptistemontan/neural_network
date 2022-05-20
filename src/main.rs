#![feature(iterator_try_collect)]

use std::{error::Error, path::Path};

use img::Img;
use load_save::{Loadable, Savable};
use neural::NeuralNetwork;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
    println!("Loading training images...");

    let train_batch = load_imgs(csv_path, max_count)?;

    let size = train_batch.len();

    println!("Loaded {} training images", size);

    neural_network.train_batch(train_batch.iter())?;

    Ok(size)
}

pub fn test<P: AsRef<Path>>(
    csv_path: P,
    take_count: usize,
    neural_network: &NeuralNetwork,
) -> Result<(f64, usize), Box<dyn Error>> {

    println!("Loading test images...");

    let test_batch = load_imgs(csv_path, take_count)?;

    let size = test_batch.len();

    println!("Loaded {} test images", size);

    let mut i = 0;
    let score = neural_network.predict_batch(&test_batch, |output, reference| {
        if i % 100 == 0 {
            println!("{}", i);
        }
        i += 1;
        let output_max = output.max_arg();
        let reference_max = reference.max_arg();
        if output_max == reference_max {
            1.0
        } else {
            0.0
        }
    })?;
    Ok((score, size))
}

// fn main() -> Result<(), Box<dyn Error>> {
//     let mut neural_network = NeuralNetwork::new(784, [500], 10, 0.05);

//     let start = Instant::now();

//     let batch_size = train("data/mnist_train.csv", usize::MAX, &mut neural_network)?;

//     let elapsed = start.elapsed();

//     println!("Training took {} seconds", elapsed.as_secs_f64());

//     println!(
//         "Average training time per image: {} ms",
//         elapsed.as_millis() as f64 / batch_size as f64
//     );

//     neural_network.save("network_save/neural_network.json")?;

//     let start = Instant::now();

//     let (score, batch_size) = test("data/mnist_test.csv", usize::MAX, &neural_network)?;

//     let elapsed = start.elapsed();

//     println!("Testing took {} seconds", elapsed.as_secs_f64());

//     println!(
//         "Average testing time per image: {} ms",
//         elapsed.as_millis() as f64 / batch_size as f64
//     );

//     println!("Score: {}", score);

//     Ok(())
// }

fn main() -> Result<(), Box<dyn Error>> {

    let training_images = load_imgs("data/mnist_train.csv", 60000)?;

    let test_images = load_imgs("data/mnist_test.csv", usize::MAX)?;


    let mut result: Vec<(f64, f64)> = (1..=100).into_par_iter().map(|x| x as f64 / 100.0).map(|learning_rate| {

        println!("Learning rate: {}", learning_rate);
    
        let mut neural_network = NeuralNetwork::new(
            784, 
            [300], 
            10, 
            learning_rate
        );
    
        neural_network.train_batch(&training_images).unwrap();
    
        let score = neural_network.predict_batch(
            &test_images, 
            |output, reference| {
            if output.max_arg() == reference.max_arg() { 1.0 } else { 0.0 }
        }).unwrap();
    
        (learning_rate, score)
    }).collect();

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (learning_rate, score)) in result.into_iter().enumerate() {
        println!("{} : {} - {}", i + 1, learning_rate, score);
    }

    Ok(())
}
