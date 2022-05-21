use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use rand::distributions::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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

    fn calculate_nabla_w(delta: &ColumnVector, activation: &ColumnVector) -> Matrix {
        delta.dot(&activation.transpose())
    }

    fn calculate_errors(
        &self,
        mut activations: Vec<ColumnVector>,
        mut zs: Vec<ColumnVector>,
        expected_output: ColumnVector,
    ) -> Result<(Vec<ColumnVector>, Vec<Matrix>)> {
        let last_z = zs.pop().unwrap();
        let last_z_primed = last_z.map(A::activate_prime);
        let final_output = activations.pop().unwrap();

        let cost_gradient = final_output - expected_output;
        let delta = cost_gradient.hadamard_product(&last_z_primed);
        let activation = activations.pop().unwrap();
        let mut nabla_w = vec![Self::calculate_nabla_w(&delta, &activation)];
        let mut nabla_b = vec![delta];

        let iter = self
            .layers
            .iter()
            .rev()
            .zip(zs.iter().zip(activations.iter()).rev());

        for ((weights, _bias), (z, activation)) in iter {
            let z_primed = z.map(A::activate_prime);
            let old_delta = nabla_b.last().unwrap();
            let delta = weights
                .transpose()
                .dot(old_delta)
                .hadamard_product(&z_primed);
            nabla_w.push(Self::calculate_nabla_w(&delta, &activation));
            nabla_b.push(delta);
        }
        nabla_b.reverse();
        nabla_w.reverse();

        Ok((nabla_b, nabla_w))
    }

    fn backpropagate(&mut self, nabla_b: Vec<ColumnVector>, nabla_w: Vec<Matrix>) -> Result<()> {
        let iter = self
            .layers
            .iter_mut()
            .zip(nabla_b.iter().zip(nabla_w.iter()));
        for ((weights, bias), (nabla_b, nabla_w)) in iter {
            *weights -= nabla_w * self.learning_rate;
            *bias -= nabla_b * self.learning_rate;
        }
        Ok(())
    }

    // pub fn train(&mut self, input: ColumnVector, expected_output: ColumnVector) -> Result<()> {
    //     let (nabla_b, nabla_w) = self.calculate_gradient(input, expected_output)?;

    //     self.backpropagate(nabla_b, nabla_w)
    // }

    pub fn calculate_gradient(
        &self,
        input: ColumnVector,
        expected_output: ColumnVector,
    ) -> Result<(Vec<ColumnVector>, Vec<Matrix>)> {
        let (activation, zs) = self.feed_forward(input);

        self.calculate_errors(activation, zs, expected_output)
    }

    fn feed_forward(&self, input: ColumnVector) -> (Vec<ColumnVector>, Vec<ColumnVector>) {
        self.layers.iter().fold(
            (vec![input], vec![]),
            |(mut activations, mut zs), (weights, bias)| {
                let last_activation = activations.last().unwrap();
                let z = &weights.dot(last_activation) + bias;
                let output = z.map(A::activate);
                activations.push(output);
                zs.push(z);
                (activations, zs)
            },
        )
    }

    pub fn predict(&self, input: ColumnVector) -> Result<ColumnVector> {
        let output = self
            .layers
            .iter()
            .try_fold(input, |input, (weights, bias)| {
                let z = &weights.try_dot(&input)? + bias;
                Ok(z.map(A::activate))
            })?;

        Ok(O::activate(&output))
    }

    pub fn train_batch<I, T>(&mut self, batch: I, mini_batch_size: usize) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<TestCase>,
    {
        let mut iter = batch
            .into_iter()
            .map(Into::<TestCase>::into)
            .map(Into::<(ColumnVector, ColumnVector)>::into);

        let mut count = usize::MAX;

        let sizes: Vec<_> = self
            .layers
            .iter()
            .map(|(weights, bias)| (weights.size(), bias.len()))
            .collect();

        let identity = || -> (Vec<ColumnVector>, Vec<Matrix>) {
            let weights: Vec<Matrix> = sizes
                .iter()
                .copied()
                .map(|((rows, cols), _)| Matrix::new(rows, cols))
                .collect();
            let biases: Vec<ColumnVector> = sizes
                .iter()
                .copied()
                .map(|(_, cols)| ColumnVector::new(cols))
                .collect();
            (biases, weights)
        };

        while count >= mini_batch_size {
            let mini_batch: Vec<(ColumnVector, ColumnVector)> =
                iter.by_ref().take(mini_batch_size).collect();
            count = mini_batch.len();
            let (mut nabla_b, mut nabla_w) = mini_batch
                .into_par_iter()
                .map(|(input, expected_output)| self.calculate_gradient(input, expected_output))
                .try_reduce(
                    identity,
                    |(mut acc_nabla_b, mut acc_nabla_w), (nabla_b, nabla_w)| {
                        acc_nabla_b
                            .iter_mut()
                            .zip(nabla_b.iter())
                            .for_each(|(a, b)| {
                                *a += b;
                            });
                        acc_nabla_w
                            .iter_mut()
                            .zip(nabla_w.iter())
                            .for_each(|(a, b)| {
                                *a += b;
                            });
                        Ok((acc_nabla_b, acc_nabla_w))
                    },
                )?;
            nabla_b
                .iter_mut()
                .for_each(|nabla_b| *nabla_b *= 1.0 / mini_batch_size as f64);
            nabla_w
                .iter_mut()
                .for_each(|nabla_w| *nabla_w *= 1.0 / mini_batch_size as f64);
            self.backpropagate(nabla_b, nabla_w)?;
        }
        Ok(())
    }

    pub fn predict_batch<F, T, I>(&self, batch: I, grade_fn: F) -> Result<f64>
    where
        I: IntoParallelIterator<Item = T>,
        T: Into<TestCase>,
        F: Fn(ColumnVector, ColumnVector) -> f64 + Send + Sync,
    {
        let count = AtomicUsize::new(0);
        let results = batch
            .into_par_iter()
            .map(|data| {
                let testcase: TestCase = data.into();
                let (input, expected_output) = testcase.into();
                let output = self.predict(input)?;
                let grade = grade_fn(output, expected_output);
                count.fetch_add(1, Ordering::SeqCst);
                Ok(grade)
            })
            .try_reduce(|| 0.0, |acc, grade| Ok(acc + grade))?;

        let total_count = count.load(Ordering::SeqCst);

        Ok(results / total_count as f64)
    }
}
