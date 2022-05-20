use std::ops::Mul;

use crate::vector::{ColumnVector, RowVector};

use crate::matrix::{Matrix, MatrixOpError, Result};

pub trait DotProduct<T = Self> {
    type Output;

    fn dot(&self, other: &T) -> Result<Self::Output>;
}

impl DotProduct for RowVector {
    type Output = f64;

    fn dot(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct<ColumnVector> for RowVector {
    type Output = f64;

    fn dot(&self, other: &ColumnVector) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct for ColumnVector {
    type Output = f64;

    fn dot(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct for Matrix {
    type Output = Matrix;

    fn dot(&self, other: &Self) -> Result<Self::Output> {
        if self.cols_count() != other.rows_count() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        let mut result = Matrix::new(self.rows_count(), other.cols_count());

        for (i, row) in self.iter().enumerate() {
            for (j, col) in other.cols().enumerate() {
                result[i][j] = row.dot(&col)?;
            }
        }

        Ok(result)
    }
}

impl DotProduct<ColumnVector> for Matrix {
    type Output = ColumnVector;

    fn dot(&self, other: &ColumnVector) -> Result<Self::Output> {
        self.iter().map(|row| row.dot(other)).try_collect()
    }
}

impl DotProduct<RowVector> for ColumnVector {
    type Output = Matrix;

    fn dot(&self, other: &RowVector) -> Result<Self::Output> {
        Ok(self.iter().map(|col| other.mul(*col)).collect())
    }
}
