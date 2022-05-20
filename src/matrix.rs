use std::{
    fmt::Display,
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
};

use rand::{distributions::Distribution, Rng};
use serde::{Deserialize, Serialize};

use crate::vector::{ColumnVector, RowVector, Vector};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    data: Vec<RowVector>,
}

#[derive(Debug)]
pub enum MatrixOpError {
    SizeMismatch,
    DotProductSizeMismatch,
}

impl Display for MatrixOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixOpError::SizeMismatch => write!(f, "Size mismatch"),
            MatrixOpError::DotProductSizeMismatch => write!(f, "Dot product size mismatch"),
        }
    }
}

impl std::error::Error for MatrixOpError {}

pub type Result<R = Matrix> = std::result::Result<R, MatrixOpError>;

pub struct ColIter<'a> {
    matrix: &'a Matrix,
    index: usize,
}

impl ColIter<'_> {
    pub fn new(matrix: &Matrix) -> ColIter {
        ColIter { matrix, index: 0 }
    }
}

impl Iterator for ColIter<'_> {
    type Item = ColumnVector;

    fn next(&mut self) -> Option<Self::Item> {
        let col = self.matrix.col(self.index);
        self.index += 1;
        col
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let data = vec![RowVector::new(cols); rows];
        Matrix { data }
    }

    pub fn new_filled(rows: usize, cols: usize, value: f64) -> Matrix {
        let data = vec![RowVector::new_filled(cols, value); rows];
        Matrix { data }
    }

    pub fn randomize<D: Distribution<f64>, R: Rng>(&mut self, distribution: D, rng: &mut R) {
        for row in self.data.iter_mut() {
            row.randomize(&distribution, rng);
        }
    }

    pub fn max_arg(&self) -> (usize, usize) {
        let mut max_arg = (0, 0);
        let mut max_val = self[0][0];

        for (row, row_vec) in self.data.iter().enumerate() {
            for (col, elem) in row_vec.iter().enumerate() {
                if *elem > max_val {
                    max_val = *elem;
                    max_arg = (row, col);
                }
            }
        }

        max_arg
    }

    pub fn flatten(&self) -> impl Iterator<Item = f64> + '_ {
        self.iter().flat_map(|row| row.iter()).copied()
    }

    pub fn clear(&mut self) {
        for row in self.data.iter_mut() {
            for elem in row.iter_mut() {
                *elem = 0.0;
            }
        }
    }

    pub fn fill(&mut self, value: f64) {
        for row in self.data.iter_mut() {
            for cell in row.iter_mut() {
                *cell = value;
            }
        }
    }

    pub fn cols_count(&self) -> usize {
        self.data.first().map(|row| row.len()).unwrap_or(0)
    }

    pub fn col(&self, col_index: usize) -> Option<ColumnVector> {
        if col_index >= self.cols_count() {
            return None;
        }
        let vec: Vec<f64> = self.data.iter().map(|row| row[col_index]).collect();
        Some(vec.into())
    }

    pub fn cols(&self) -> ColIter {
        ColIter::new(self)
    }

    pub fn rows_count(&self) -> usize {
        self.data.len()
    }

    pub fn row(&self, row_index: usize) -> Option<&RowVector> {
        self.data.get(row_index)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows_count(), self.cols_count())
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols_count(), self.rows_count());
        for i in 0..self.rows_count() {
            for j in 0..self.cols_count() {
                result[j][i] = self[i][j];
            }
        }
        result
    }

    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Matrix {
        let mut mat = Matrix::new(self.rows_count(), self.cols_count());
        for (row, row_vec) in self.data.iter().enumerate() {
            for (col, elem) in row_vec.iter().enumerate() {
                mat[row][col] = f(*elem);
            }
        }
        mat
    }

    fn check_size(&self, other: &Self) -> Result<()> {
        if self.size() != other.size() {
            return Err(MatrixOpError::SizeMismatch);
        }
        Ok(())
    }

    fn apply<F: Fn(f64, f64) -> f64>(&self, other: &Matrix, f: F) -> Result {
        self.check_size(other)?;
        let mut mat = Matrix::new(self.rows_count(), self.cols_count());
        for (row, row_vec) in mat.data.iter_mut().enumerate() {
            for (col, elem) in row_vec.iter_mut().enumerate() {
                *elem = f(self[row][col], other[row][col]);
            }
        }
        Ok(mat)
    }
}

impl Deref for Matrix {
    type Target = [RowVector];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Add for &Matrix {
    type Output = Result;

    fn add(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::add)
    }
}

impl Sub for &Matrix {
    type Output = Result;

    fn sub(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::sub)
    }
}

impl Mul for &Matrix {
    type Output = Result;

    fn mul(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::mul)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: f64) -> Self::Output {
        self.map(|x| x * other)
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, other: f64) -> Self::Output {
        self.map(|x| x + other)
    }
}

impl Div<f64> for &Matrix {
    type Output = Matrix;

    fn div(self, other: f64) -> Self::Output {
        self.map(|x| x / other)
    }
}

impl FromIterator<RowVector> for Matrix {
    fn from_iter<T: IntoIterator<Item = RowVector>>(iter: T) -> Self {
        let data: Vec<RowVector> = iter.into_iter().collect();
        Matrix { data }
    }
}
