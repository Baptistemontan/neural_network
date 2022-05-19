use std::ops::{Add, Mul, Sub};

use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::matrix::{Matrix, Result, VectorAxis};

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    learning_rate: f64,
    hidden_weights: Matrix,
    output_weights: Matrix,
}

pub fn unifom_distrib(n: f64) -> Uniform<f64> {
    let max = 1.0 / n.sqrt();
    let min = -max;
    Uniform::new_inclusive(min, max)
}

impl NeuralNetwork {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: f64) -> Self {
        let mut hidden_weights = Matrix::new(hidden, input);
        let mut output_weights = Matrix::new(output, hidden);
        hidden_weights.randomize(&unifom_distrib(hidden as f64));
        output_weights.randomize(&unifom_distrib(output as f64));
        NeuralNetwork {
            learning_rate,
            hidden_weights,
            output_weights,
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

    pub fn train(&mut self, input: &Matrix, output: &Matrix) -> Result<()> {
        // // Feed forward
        // let hidden_outputs = self.hidden_weights.dot(input)?.map(sigmoid);
        // let final_outputs = self.output_weights.dot(&hidden_outputs)?.map(sigmoid);

        // // find errors
        // let outputs_errors = output.sub(&final_outputs)?;
        // let hidden_errors = self.output_weights.transpose().dot(&outputs_errors)?;

        // // backpropagate
        // self.output_weights = Self::back_propagate(
        //     &final_outputs,
        //     &outputs_errors,
        //     &hidden_outputs,
        //     self.learning_rate,
        //     &self.output_weights,
        // )?;

        // self.hidden_weights = Self::back_propagate(
        //     &hidden_outputs,
        //     &hidden_errors,
        //     input,
        //     self.learning_rate,
        //     &self.hidden_weights,
        // )?;

        let hidden_inputs = self.hidden_weights.dot(input)?;
        let hidden_outputs = hidden_inputs.map(sigmoid);
        let final_inputs = self.output_weights.dot(&hidden_outputs)?;
        let final_outputs = final_inputs.map(sigmoid);

        let outputs_errors = output.sub(&final_outputs)?;
        let hidden_errors = self.output_weights.transpose().dot(&outputs_errors)?;

        let sigmoid_primed_mat = final_outputs.sigmoid_prime();
        let multiplied_mat = outputs_errors.mul(&sigmoid_primed_mat)?;
        let transposed_mat = hidden_outputs.transpose();
        let dot_mat = multiplied_mat.dot(&transposed_mat)?;
        let scaled_mat = dot_mat.mul(self.learning_rate);
        let added_mat = self.output_weights.add(&scaled_mat)?;
        self.output_weights = added_mat;


        let sigmoid_primed_mat = hidden_outputs.sigmoid_prime();
        let multiplied_mat = hidden_errors.mul(&sigmoid_primed_mat)?;
        let transposed_mat = input.transpose();
        let dot_mat = multiplied_mat.dot(&transposed_mat)?;
        let scaled_mat = dot_mat.mul(self.learning_rate);
        let added_mat = self.hidden_weights.add(&scaled_mat)?;
        self.hidden_weights = added_mat;

        Ok(())
    }

    pub fn predict(&self, input: &Matrix) -> Result {
        let hidden_outputs = self.hidden_weights.dot(input)?.map(sigmoid);
        let final_outputs = self.output_weights.dot(&hidden_outputs)?.map(sigmoid);
        Ok(final_outputs.soft_max())
    }

    pub fn train_batch<'a, I>(&mut self, batch: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a Matrix, &'a Matrix)>,
    {
        for (i, (input, output)) in batch.into_iter().enumerate() {
            if i % 100 == 0 {
                println!("Data N°{}", i);
            }
            let flattened = input.to_vector(VectorAxis::Column);
            self.train(&flattened, output)?;
        }
        Ok(())
    }

    pub fn test_prediction<'a, F, I>(&self, batch: I, grade_fn: F) -> Result<f64>
    where
        I: IntoIterator<Item = (&'a Matrix, &'a Matrix)>,
        F: Fn(&Matrix, &Matrix) -> f64,
    {
        let mut correct = 0.0;
        let mut count = 0.0;
        for (input, output) in batch {
            let flattened = input.to_vector(VectorAxis::Column);
            let prediction = self.predict(&flattened)?;
            correct += grade_fn(&prediction, &output);
            count += 1.0;
        }
        Ok(correct / count)
    }
}
