use std::ops::{Deref, DerefMut};

use crate::matrix::{MatrixOpError, Result};

mod col_vector;
mod row_vector;

pub use col_vector::ColumnVector;
use rand::{prelude::Distribution, Rng};
pub use row_vector::RowVector;

pub trait Vector: Sized + FromIterator<f64> + Deref<Target = [f64]> + DerefMut {
    fn new_filled(size: usize, value: f64) -> Self;

    fn to_owned_vec(self) -> Vec<f64>;

    fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        self.to_vec().iter().copied().map(f).collect()
    }

    fn apply<F: Fn(f64, f64) -> f64>(&self, other: &Self, f: F) -> Result<Self> {
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

    fn from_vec(data: Vec<f64>) -> Self;

    fn max_arg(&self) -> usize {
        let mut max_index = 0;
        let mut max_value = self[0];
        for (i, &value) in self.iter().enumerate() {
            if value > max_value {
                max_index = i;
                max_value = value;
            }
        }
        max_index
    }

    fn randomize<D: Distribution<f64>, R: Rng>(&mut self, distributions: &D, rng: &mut R) {
        for value in self.iter_mut() {
            *value = distributions.sample(rng);
        }
    }
}
