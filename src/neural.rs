use std::{marker::PhantomData, ops::{Mul}};

use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::{
    activation::{ActivationFunction, OutputActivationFunction},
    dot_product::DotProduct,
    matrix::{Matrix, Result},
    vector::{ColumnVector, Vector},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork<D, A> {
    learning_rate: f64,
    layers: Vec<(Matrix, ColumnVector)>,
    _hidden_layer_activation: PhantomData<D>,
    _output_layer_activation: PhantomData<A>,
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
}

impl Into<(ColumnVector, ColumnVector)> for TestCase {
    fn into(self) -> (ColumnVector, ColumnVector) {
        (self.input, self.expected_output)
    }
}

impl<A: ActivationFunction, O: OutputActivationFunction> NeuralNetwork<A, O> {
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
            _hidden_layer_activation: PhantomData,
            _output_layer_activation: PhantomData,
        }
    }

    fn calculate_nabla_w(delta: &ColumnVector, activation: &ColumnVector) -> Result<Matrix> {
        delta.try_dot(&activation.transpose())
    }

    fn calculate_errors(&self, mut activations: Vec<ColumnVector>, mut zs: Vec<ColumnVector> , expected_output: ColumnVector) -> Result<(Vec<ColumnVector>, Vec<Matrix>)> {

        let last_z = zs.pop().unwrap();
        let last_z_primed = last_z.map(A::activate_prime);
        let final_output = activations.pop().unwrap();
        
        let cost_gradient = final_output.sub(&expected_output)?;
        let delta = cost_gradient.hadamard_product(&last_z_primed)?;
        let activation = activations.pop().unwrap();
        let mut nabla_w = vec![Self::calculate_nabla_w(&delta, &activation)?];
        let mut nabla_b = vec![delta];

        let iter = self.layers.iter().rev().zip(zs.iter().zip(activations.iter()).rev());
        
        
        for ((weights, _bias), (z, activation)) in iter {
            let z_primed = z.map(A::activate_prime);
            let old_delta = nabla_b.last().unwrap();
            let delta = weights.transpose().try_dot(old_delta)?.hadamard_product(&z_primed)?;
            nabla_w.push(Self::calculate_nabla_w(&delta, &activation)?);
            nabla_b.push(delta);
        }
        nabla_b.reverse();
        nabla_w.reverse();

        Ok((nabla_b, nabla_w))
    }

    fn backpropagate(&mut self, nabla_b: Vec<ColumnVector>, nabla_w: Vec<Matrix>) -> Result<()> {
        let iter = self.layers.iter_mut().zip(nabla_b.iter().zip(nabla_w.iter()));
        for ((weights, bias), (nabla_b, nabla_w)) in iter  {
            *weights -= &nabla_w.mul(self.learning_rate);
            *bias = bias.sub(&(nabla_b.scale(self.learning_rate)))?;
        }
        Ok(())
    }

    pub fn train(&mut self, input: ColumnVector, expected_output: ColumnVector) -> Result<()> {
        let (activation, zs) = self.feed_forward(input)?;

        let (nabla_b, nabla_w) = self.calculate_errors(activation, zs, expected_output)?;

        self.backpropagate(nabla_b, nabla_w)
    }

    fn feed_forward(&self, input: ColumnVector) -> Result<(Vec<ColumnVector>, Vec<ColumnVector>)> {
        self.layers.iter().try_fold(
            (vec![input], vec![]),
            |(mut activations, mut zs), (weights, bias)| {
                let last_activation = activations.last().unwrap();
                let z = weights.try_dot(last_activation)?.add(&bias)?;
                let output = O::activate(&z);
                activations.push(output);
                zs.push(z);
                Ok((activations, zs))
            },
        )
    }

    pub fn predict(&self, input: ColumnVector) -> Result<ColumnVector> {
        let (mut activations, _) = self.feed_forward(input)?;
        Ok(activations.pop().unwrap().map(A::activate))
    }

    pub fn train_batch<I, T>(&mut self, batch: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<TestCase>,
    {
        let iter = batch
            .into_iter()
            .map(Into::<TestCase>::into)
            .map(Into::into);
        for (input, expected_output) in iter {
            self.train(input, expected_output)?;
        }
        Ok(())
    }

    pub fn predict_batch<F, T, I>(&self, batch: I, mut grade_fn: F) -> Result<f64>
    where
        I: IntoIterator<Item = T>,
        T: Into<TestCase>,
        F: FnMut(ColumnVector, ColumnVector) -> f64,
    {
        let mut correct = 0.0;
        let mut count = 0.0;
        for (input, expected_output) in batch
            .into_iter()
            .map(Into::<TestCase>::into)
            .map(Into::into)
        {
            let prediction = self.predict(input)?;
            correct += grade_fn(prediction, expected_output);
            count += 1.0;
        }
        Ok(correct / count)
    }
}
