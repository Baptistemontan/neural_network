use std::ops::{Deref, DerefMut, Mul, Sub};

use serde::{Deserialize, Serialize};

use crate::matrix::Result;

use super::{RowVector, Vector};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnVector {
    data: Vec<f64>,
}

impl FromIterator<f64> for ColumnVector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let data: Vec<f64> = iter.into_iter().collect();
        ColumnVector { data }
    }
}

impl From<Vec<f64>> for ColumnVector {
    fn from(data: Vec<f64>) -> Self {
        ColumnVector { data }
    }
}

impl Sub for &ColumnVector {
    type Output = Result<ColumnVector>;

    fn sub(self, other: Self) -> Self::Output {
        self.apply(other, f64::sub)
    }
}

impl Mul for &ColumnVector {
    type Output = Result<ColumnVector>;

    fn mul(self, other: Self) -> Self::Output {
        self.apply(other, f64::mul)
    }
}

impl Deref for ColumnVector {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for ColumnVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Vector for ColumnVector {
    fn new_filled(size: usize, value: f64) -> Self {
        ColumnVector {
            data: vec![value; size],
        }
    }

    fn to_owned_vec(self) -> Vec<f64> {
        self.data.to_owned()
    }

    fn from_vec(data: Vec<f64>) -> Self {
        ColumnVector { data }
    }
}

impl ColumnVector {
    pub fn new(size: usize) -> Self {
        ColumnVector {
            data: vec![0.0; size],
        }
    }

    pub fn transpose(&self) -> RowVector {
        self.data.clone().into()
    }

    pub fn sigmoid_prime(&self) -> Self {
        Self::new_filled(self.len(), 1.0)
            .sub(self)
            .and_then(|ref m| m * self)
            .unwrap()
    }
}
