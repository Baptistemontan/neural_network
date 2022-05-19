use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub, Div},
};

use rand::{distributions::Distribution};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub enum MatrixOpError {
    SizeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
        operation: &'static str,
    },
    DotProductSizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl Display for MatrixOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixOpError::SizeMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Size mismatch: expected {:?}, actual {:?}, operation: {}",
                    expected, actual, operation
                )
            }
            MatrixOpError::DotProductSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Size mismatch: expected {:?}, actual {:?}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for MatrixOpError {}

pub type Result<R = Matrix> = std::result::Result<R, MatrixOpError>;

pub enum VectorAxis {
    Row,
    Column,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Self::from_vec(vec![vec![0.0; cols]; rows])
    }

    pub fn new_filled(rows: usize, cols: usize, value: f64) -> Matrix {
        Self::from_vec(vec![vec![value; cols]; rows])
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Matrix {
        Matrix { data }
    }

    pub fn randomize<D: Distribution<f64>>(&mut self, distribution: D) {
        for row in self.data.iter_mut() {
            for elem in row.iter_mut() {
                *elem = distribution.sample(&mut rand::thread_rng());
            }
        }
    }

    // pub fn randomize(&mut self, n: usize) {
    //     fn unifom_distribution(low: f64, high:f64) -> f64 {
    //         let range = high - low;
    //         let scale = 10000.0;
    //         let scaled_range = (range * scale) as usize;
    //         let rnd = rand::thread_rng().gen_range(0..scaled_range);
    //         ((rnd as f64) / scale) + low
    //     }

    //     let min = -1.0 / n as f64;
    //     let max = 1.0 / n as f64;
    //     for row in self.data.iter_mut() {
    //         for elem in row.iter_mut() {
    //             *elem = unifom_distribution(min, max);
    //         }
    //     }
    // }

    pub fn max_arg(&self) -> (usize, usize) {
        let mut max_arg = (0, 0);
        let mut max_val = self[(0, 0)];

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
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .copied()
    }

    pub fn to_vector(&self, axis: VectorAxis) -> Matrix {
        let mut mat = match axis {
            VectorAxis::Column => Matrix::new(self.rows() * self.cols(), 1),
            VectorAxis::Row => Matrix::new(1, self.rows() * self.cols()),
        };
        for (row, row_vec) in self.data.iter().enumerate() {
            for (col, elem) in row_vec.iter().enumerate() {
                let index = row * self.cols() + col;
                let index = match axis {
                    VectorAxis::Column => (index, 0),
                    VectorAxis::Row => (0, index),
                };
                mat[index] = *elem;
            }
        }
        mat
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

    pub fn cols(&self) -> usize {
        self.data.first().map(|row| row.len()).unwrap_or(0)
    }

    pub fn rows(&self) -> usize {
        self.data.len()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols(), self.rows());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    pub fn dot(&self, other: &Matrix) -> Result {
        if self.cols() != other.rows() {
            return Err(MatrixOpError::DotProductSizeMismatch {
                expected: self.cols(),
                actual: other.rows(),
            });
        }
        let mut result = Matrix::new(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut sum = 0.0;
                for k in 0..self.cols() {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        Ok(result)
    }

    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Matrix {
        let mut mat = Matrix::new(self.rows(), self.cols());
        for (row, row_vec) in self.data.iter().enumerate() {
            for (col, elem) in row_vec.iter().enumerate() {
                mat[(row, col)] = f(*elem);
            }
        }
        mat
    }

    fn check_size(&self, other: &Self, operation: &'static str) -> Result<()> {
        if self.size() != other.size() {
            return Err(MatrixOpError::SizeMismatch {
                expected: self.size(),
                actual: other.size(),
                operation,
            });
        }
        Ok(())
    }

    fn apply<F: Fn(f64, f64) -> f64>(
        &self,
        other: &Matrix,
        f: F,
        operation: &'static str,
    ) -> Result {
        self.check_size(other, operation)?;
        let mut mat = Matrix::new(self.rows(), self.cols());
        for (row, row_vec) in mat.data.iter_mut().enumerate() {
            for (col, elem) in row_vec.iter_mut().enumerate() {
                *elem = f(self[(row, col)], other[(row, col)]);
            }
        }
        Ok(mat)
    }

    pub fn sigmoid_prime(&self) -> Matrix {
        let result = Matrix::new_filled(self.rows(), self.cols(), 1.0)
            .sub(self)
            .and_then(|ref m| m * self);

        unsafe {
            result.unwrap_unchecked()
        }
    }

    pub fn soft_max(&self) -> Matrix {
        let total: f64 = self.flatten().map(f64::exp).sum();
        self.map(f64::exp).div(total)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        &self.data[row][col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        &mut self.data[row][col]
    }
}

impl Add for &Matrix {
    type Output = Result;

    fn add(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::add, "add")
    }
}

impl Sub for &Matrix {
    type Output = Result;

    fn sub(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::sub, "sub")
    }
}

impl Mul for &Matrix {
    type Output = Result;

    fn mul(self, other: &Matrix) -> Self::Output {
        self.apply(&other, f64::mul, "mul")
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
