use std::ops::Sub;

use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::{
    dot_product::DotProduct,
    matrix::{Matrix, Result},
    vector::{ColumnVector, Vector},
};

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    learning_rate: f64,
    layers: Vec<(Matrix, ColumnVector)>,
}

pub fn unifom_distrib(n: f64) -> Uniform<f64> {
    let max = 1.0 / n.sqrt();
    let min = -max;
    Uniform::new_inclusive(min, max)
}

pub struct TestCase {
    input: ColumnVector,
    expected_output: ColumnVector,
}

impl TestCase {
    pub fn new(input: ColumnVector, expected_output: ColumnVector) -> Self {
        Self {
            input,
            expected_output,
        }
    }

    pub fn input(&self) -> &ColumnVector {
        &self.input
    }

    pub fn expected_output(&self) -> &ColumnVector {
        &self.expected_output
    }

    pub fn collapse(self) -> (ColumnVector, ColumnVector) {
        (self.input, self.expected_output)
    }
}

impl NeuralNetwork {
    pub fn new<I>(
        input_size: usize,
        hidden_sizes: I,
        output_size: usize,
        learning_rate: f64,
    ) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut rng = rand::thread_rng();
        let new_layer = |last_size: &mut usize, size: usize| -> Option<(Matrix, ColumnVector)> {
            let mut layer = Matrix::new(size, *last_size);
            layer.randomize(&unifom_distrib(size as f64), &mut rng);
            *last_size = size;
            let bias = ColumnVector::new(size);
            Some((layer, bias))
        };

        let layers: Vec<(Matrix, ColumnVector)> = hidden_sizes
            .into_iter()
            .chain(Some(output_size))
            .scan(input_size, new_layer)
            .collect();

        NeuralNetwork {
            learning_rate,
            layers,
        }
    }

    // fn back_propagate(
    //     outputs: &Matrix,
    //     errors: &Matrix,
    //     input: &Matrix,
    //     learning_rate: f64,
    //     target: &Matrix,
    // ) -> Result {
    //     errors
    //         .mul(&outputs.sigmoid_prime())?
    //         .dot(&input.transpose())?
    //         .mul(learning_rate)
    //         .add(target)
    // }

    pub fn train(&mut self, input: ColumnVector, expected_output: ColumnVector) -> Result<()> {
        let outputs = self.prediction_steps(input)?;

        let iter = self.layers.iter_mut().zip(outputs.windows(2)).rev();

        let mut local_expected_output = expected_output;

        for ((weights, bias), window) in iter {
            let [input, result]: &[_; 2] = window.try_into().unwrap();
            let cost = result.sub(&local_expected_output)?.scale(2.0);
            let sigmoid_prime = result.sigmoid_prime();
            let multiplied_mat = sigmoid_prime.mul(&cost)?.scale(self.learning_rate);

            *bias = bias.sub(&multiplied_mat)?;

            let weight_delta = multiplied_mat.try_dot(&input.transpose())?;
            *weights = weights.sub(&weight_delta)?;

            let output_delta = multiplied_mat.transpose().try_dot(weights)?;
            local_expected_output = input.sub(&output_delta)?;
        }

        Ok(())
    }

    fn prediction_steps(&self, input: ColumnVector) -> Result<Vec<ColumnVector>> {
        self.layers
            .iter()
            .try_fold(vec![input], |mut outputs, (weights, bias)| {
                let last_output = outputs.last().unwrap();
                let output = weights.try_dot(last_output)?.add(bias)?.map(sigmoid);
                outputs.push(output);
                Ok(outputs)
            })
    }

    pub fn predict(&self, input: ColumnVector) -> Result<ColumnVector> {
        self.prediction_steps(input)
            .map(|mut outputs| outputs.pop().unwrap().soft_max())
    }

    pub fn train_batch<I, T>(&mut self, batch: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<TestCase>,
    {
        let iter = batch
            .into_iter()
            .map(Into::<TestCase>::into)
            .map(TestCase::collapse);
        for (i, (input, expected_output)) in iter.enumerate() {
            if i % 100 == 0 {
                println!("Data N°{}", i);
            }
            self.train(input, expected_output)?;
        }
        Ok(())
    }

    pub fn predict_batch<F, T, I>(&self, batch: I, grade_fn: F) -> Result<f64>
    where
        I: IntoIterator<Item = T>,
        T: Into<TestCase>,
        F: Fn(ColumnVector, ColumnVector) -> f64,
    {
        let mut correct = 0.0;
        let mut count = 0.0;
        for (input, expected_output) in batch
            .into_iter()
            .map(Into::<TestCase>::into)
            .map(TestCase::collapse)
        {
            let prediction = self.predict(input)?;
            correct += grade_fn(prediction, expected_output);
            count += 1.0;
        }
        Ok(correct / count)
    }
}
