use std::ops::{Deref, DerefMut, Mul, Sub};

use serde::{Deserialize, Serialize};

use crate::matrix::Result;

use super::{ColumnVector, Vector};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowVector {
    data: Vec<f64>,
}

impl Deref for RowVector {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for RowVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Mul for &RowVector {
    type Output = Result<RowVector>;

    fn mul(self, other: Self) -> Self::Output {
        self.apply(other, f64::mul)
    }
}

impl From<Vec<f64>> for RowVector {
    fn from(data: Vec<f64>) -> Self {
        RowVector { data }
    }
}

impl Vector for RowVector {
    fn new_filled(size: usize, value: f64) -> Self {
        RowVector {
            data: vec![value; size],
        }
    }

    fn to_owned_vec(self) -> Vec<f64> {
        self.data.to_owned()
    }

    fn from_vec(data: Vec<f64>) -> Self {
        RowVector { data }
    }
}

impl RowVector {
    pub fn new(size: usize) -> Self {
        RowVector {
            data: vec![0.0; size],
        }
    }

    pub fn transpose(self) -> ColumnVector {
        self.data.clone().into()
    }

    pub fn sigmoid_prime(&self) -> Self {
        Self::new_filled(self.len(), 1.0)
            .sub(self)
            .and_then(|ref m| m * self)
            .unwrap()
    }
}

impl FromIterator<f64> for RowVector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let data: Vec<f64> = iter.into_iter().collect();
        RowVector { data }
    }
}

impl Sub for &RowVector {
    type Output = Result<RowVector>;

    fn sub(self, other: Self) -> Self::Output {
        self.apply(other, f64::sub)
    }
}

impl Mul<f64> for &RowVector {
    type Output = RowVector;

    fn mul(self, other: f64) -> Self::Output {
        self.map(|x| x * other)
    }
}
