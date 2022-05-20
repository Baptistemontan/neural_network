use std::ops::{Deref, DerefMut};

use super::{RowVector, Vector};
use serde::{Deserialize, Serialize};

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
    type TransposeTo = RowVector;
}
