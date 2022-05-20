use crate::vector::{ColumnVector, RowVector, Vector};

use crate::matrix::{Matrix, MatrixOpError, Result};

pub trait DotProduct<T = Self> {
    type Output;

    fn dot(&self, other: &T) -> Self::Output {
        self.try_dot(other).unwrap()
    }

    fn try_dot(&self, other: &T) -> Result<Self::Output>;
}

impl DotProduct for RowVector {
    type Output = f64;

    fn try_dot(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct<ColumnVector> for RowVector {
    type Output = f64;

    fn try_dot(&self, other: &ColumnVector) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct for ColumnVector {
    type Output = f64;

    fn try_dot(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        Ok(self.iter().zip(other.iter()).map(|(a, b)| a * b).sum())
    }
}

impl DotProduct for Matrix {
    type Output = Matrix;

    fn try_dot(&self, other: &Self) -> Result<Self::Output> {
        if self.cols_count() != other.rows_count() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        let mut result = Matrix::new(self.rows_count(), other.cols_count());

        for (i, row) in self.iter().enumerate() {
            for (j, col) in other.cols().enumerate() {
                result[i][j] = row.try_dot(&col)?;
            }
        }

        Ok(result)
    }
}

impl DotProduct<ColumnVector> for Matrix {
    type Output = ColumnVector;

    fn try_dot(&self, other: &ColumnVector) -> Result<Self::Output> {
        if self.cols_count() != other.len() {
            Err(MatrixOpError::DotProductSizeMismatch)
        } else {
            self.iter().map(|row| row.try_dot(other)).try_collect()
        }
    }
}

impl DotProduct<Matrix> for RowVector {
    type Output = ColumnVector;

    fn try_dot(&self, other: &Matrix) -> Result<Self::Output> {
        if self.len() != other.rows_count() {
            return Err(MatrixOpError::DotProductSizeMismatch);
        }

        other.cols().map(|col| self.try_dot(&col)).try_collect()
    }
}

impl DotProduct<RowVector> for ColumnVector {
    type Output = Matrix;

    fn dot(&self, other: &RowVector) -> Self::Output {
        self.iter().map(|col| other.scale(*col)).collect()
    }

    fn try_dot(&self, other: &RowVector) -> Result<Self::Output> {
        Ok(self.dot(other))
    }
}
