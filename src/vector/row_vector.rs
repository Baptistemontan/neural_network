use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};

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

impl From<Vec<f64>> for RowVector {
    fn from(data: Vec<f64>) -> Self {
        RowVector { data }
    }
}

impl Vector for RowVector {
    type TransposeTo = ColumnVector;
}

impl FromIterator<f64> for RowVector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let data: Vec<f64> = iter.into_iter().collect();
        RowVector { data }
    }
}
