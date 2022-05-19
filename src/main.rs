use std::{error::Error, path::Path};

use img::Img;
use load_save::{Savable, Loadable};
use matrix::Matrix;
use neural::NeuralNetwork;

mod img;
mod load_save;
mod matrix;
mod neural;

pub fn load_imgs<P: AsRef<Path>>(csv_path: P, take_count: usize) -> Result<Vec<(Img, Matrix)>, Box<dyn Error>> {
    let imgs_loader = Img::load_from_csv(csv_path)?;
    let imgs = imgs_loader.take(take_count)
        .try_fold(vec![], |mut acc, img| {
        match img {
            Ok(img) => {
                acc.push(img);
                Ok(acc)
            },
            Err(e) => Err(e),
        }
    })?;

    let train_batch: Vec<(Img, Matrix)> = imgs.into_iter().map(|img| {
        let label = img.get_label();
        let mut output = Matrix::new(10, 1);
        output[(label, 0)] = 1.0;
        (img, output)
    }).collect();
    Ok(train_batch)
}

pub fn train<P: AsRef<Path>>(csv_path: P, max_count: usize, neural_network: &mut NeuralNetwork) -> Result<(), Box<dyn Error>> {
    
    let train_batch = load_imgs(csv_path, max_count)?;

    neural_network.train_batch(train_batch.iter().map(|(img, output)| {
        (img.get_pixels(), output)
    }))?;

    Ok(())
}

pub fn test<P: AsRef<Path>>(csv_path: P, take_count: usize, neural_network: &NeuralNetwork) -> Result<f64, Box<dyn Error>> {
    let test_batch = load_imgs(csv_path, take_count)?;
    let prediction = neural_network.test_prediction(test_batch.iter().map(|(img, output)| {
        (img.get_pixels(), output)
    }), |output, reference| {
        let (output_max, _) = output.max_arg();
        let (reference_max, _) = reference.max_arg();
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
