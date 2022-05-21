use std::ops::{Add, DerefMut, Div, Mul, Sub};

use crate::matrix::{MatrixOpError, Result};

mod col_vector;
mod row_vector;

pub use col_vector::ColumnVector;
use rand::{prelude::Distribution, Rng};
pub use row_vector::RowVector;

pub trait Vector: FromIterator<f64> + DerefMut<Target = [f64]> + From<Vec<f64>> {
    type TransposeTo: Vector;

    fn transpose(&self) -> Self::TransposeTo {
        self.iter().copied().collect()
    }

    fn map<F: FnMut(f64) -> f64>(&self, f: F) -> Self {
        self.iter().copied().map(f).collect()
    }

    fn new(size: usize) -> Self {
        vec![0.0; size].into()
    }

    fn new_filled(size: usize, value: f64) -> Self {
        vec![value; size].into()
    }

    fn apply<F: FnMut(f64, f64) -> f64>(&self, other: &Self, mut f: F) -> Result<Self> {
        if self.len() != other.len() {
            return Err(MatrixOpError::SizeMismatch);
        }
        Ok(self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| f(*a, *b))
            .collect())
    }

    fn soft_max(&self) -> Self {
        let total: f64 = self.to_vec().iter().copied().map(f64::exp).sum();
        self.map(|x| f64::exp(x) / total)
    }

    fn max_arg(&self) -> usize {
        let mut max_index = 0;
        let mut max_value = self[0];
        for (i, value) in self.iter().enumerate() {
            if value > &max_value {
                max_index = i;
                max_value = *value;
            }
        }
        max_index
    }

    fn scale(&self, factor: f64) -> Self {
        self.map(|x| x * factor)
    }

    fn randomize<D: Distribution<f64>, R: Rng>(&mut self, distributions: &D, rng: &mut R) {
        for value in self.iter_mut() {
            *value = distributions.sample(rng);
        }
    }

    // // suppose that the vector has sigmoid already applied
    // fn sigmoid_prime(&self) -> Self {
    //     self.iter().copied().map(|x| x - x.powi(2)).collect()
    // }

    fn add(&self, other: &Self) -> Result<Self> {
        self.apply(other, f64::add)
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        self.apply(other, f64::sub)
    }

    fn hadamard_product(&self, other: &Self) -> Result<Self> {
        self.apply(other, f64::mul)
    }

    fn div(&self, other: &Self) -> Result<Self> {
        self.apply(other, f64::div)
    }
}
