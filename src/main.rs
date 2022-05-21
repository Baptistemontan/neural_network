#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]

use std::{error::Error, path::Path, sync::atomic::{AtomicUsize, Ordering}, time::Instant};

use img::Img;
use load_save::{Loadable, Savable};
use neural::NeuralNetwork;
use rayon::iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator};
use vector::Vector;

use activation::{ActivationFunction, SigmoidActivation, OutputActivationFunction};

use crate::activation::{ReLUActivation, SoftMax};

mod activation;
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

pub fn train<P: AsRef<Path>, A: ActivationFunction, O: OutputActivationFunction>(
    csv_path: P,
    max_count: usize,
    neural_network: &mut NeuralNetwork<A, O>,
) -> Result<usize, Box<dyn Error>> {
    println!("Loading training images...");

    let train_batch = load_imgs(csv_path, max_count)?;

    let size = train_batch.len();

    println!("Loaded {} training images", size);

    let train_batch_iter = train_batch.iter().enumerate().map(|(i, img)| {
        if i % 100 == 0 {
            println!("Training {}/{}", i, size);
        }
        img
    });

    neural_network.train_batch(train_batch_iter, 20)?;

    Ok(size)
}

pub fn test<P: AsRef<Path>, A: ActivationFunction, O: OutputActivationFunction>(
    csv_path: P,
    take_count: usize,
    neural_network: &NeuralNetwork<A, O>,
) -> Result<(f64, usize), Box<dyn Error>> {
    println!("Loading test images...");

    let test_batch = load_imgs(csv_path, take_count)?;

    let size = test_batch.len();

    println!("Loaded {} test images", size);

    let image_index = AtomicUsize::new(0);


    let test_batch_iter = test_batch.par_iter().map(|img| {
        let i = image_index.fetch_add(1, Ordering::SeqCst);
        if i % 100 == 0 {
            println!("Testing {}/{}", i, size);
        }
        img
    });


    let score = neural_network.predict_batch(test_batch_iter, |output, reference| {
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

fn main() -> Result<(), Box<dyn Error>> {

    let mut neural_network: NeuralNetwork<SigmoidActivation, SoftMax> = NeuralNetwork::new(784, [500], 10, 0.2);
    let start = Instant::now();

    let batch_size = train("data/mnist_train.csv", usize::MAX, &mut neural_network)?;

    let elapsed = start.elapsed();

    println!("Training took {} seconds", elapsed.as_secs_f64());

    println!(
        "Average training time per image: {} ms",
        elapsed.as_millis() as f64 / batch_size as f64
    );

    neural_network.save("network_save/neural_network.json")?;

    let start = Instant::now();

    let (score, batch_size) = test("data/mnist_test.csv", usize::MAX, &neural_network)?;

    let elapsed = start.elapsed();

    println!("Testing took {} seconds", elapsed.as_secs_f64());

    println!(
        "Average testing time per image: {} ms",
        elapsed.as_millis() as f64 / batch_size as f64
    );

    println!("Score: {}", score);

    Ok(())
}

// fn main() -> Result<(), Box<dyn Error>> {
//     let training_images = load_imgs("data/mnist_train.csv", usize::MAX)?;

//     let test_images = load_imgs("data/mnist_test.csv", usize::MAX)?;

//     let index = AtomicUsize::new(0);

//     let start = Instant::now();

//     let mut result: Vec<(f64, f64)> = (1..=100)
//         .into_par_iter()
//         .map(|x| x as f64 / 100.0)
//         .map(|learning_rate| {
//             let learning_rate_index = index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

//             println!("Learning rate N° {} : {}", learning_rate_index, learning_rate);

//             let mut neural_network: NeuralNetwork<SigmoidActivation, SoftMax> =
//                 NeuralNetwork::new(784, [300], 10, learning_rate);

//             let training_batch_iter = training_images.iter().enumerate().map(|(i, img)| {
//                 if i % 500 == 0 {
//                     println!("Learning rate {} Training {}/{}", learning_rate_index, i, training_images.len());
//                 }
//                 img
//             });

//             neural_network.train_batch(training_batch_iter).unwrap();

//             let test_batch_iter = test_images.iter().enumerate().map(|(i, img)| {
//                 if i % 500 == 0 {
//                     println!("Learning rate {} Testing {}/{}", learning_rate_index, i, test_images.len());
//                 }
//                 img
//             });

//             let score = neural_network
//                 .predict_batch(test_batch_iter, |output, reference| {
//                     if output.max_arg() == reference.max_arg() {
//                         1.0
//                     } else {
//                         0.0
//                     }
//                 })
//                 .unwrap();

//             (learning_rate, score)
//         })
//         .collect();

//     let elapsed = start.elapsed().as_secs_f64();
//     let minutes = (elapsed / 60.0).floor();
//     let seconds = elapsed % 60.0;

//     println!("Training took {} minutes and {} seconds", minutes, seconds);

//     let total_size = training_images.len() * 100;

//     println!(
//         "Average training time per image: {} ms",
//         elapsed * 1000.0 / total_size as f64
//     );

//     result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

//     for (i, (learning_rate, score)) in result.into_iter().enumerate() {
//         println!("{} : {} - {}", i + 1, learning_rate, score);
//     }

//     Ok(())
// }
