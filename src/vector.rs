use serde::{Deserialize, Serialize};
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::matrix::{MatrixOpError, Result};
use rand::{prelude::Distribution, Rng};

pub trait Vector: FromIterator<f64> + DerefMut<Target = [f64]> + From<Vec<f64>> {
    type TransposeTo: Vector;

    fn transpose(&self) -> Self::TransposeTo {
        self.iter().copied().collect()
    }

    fn map<F: FnMut(f64) -> f64>(&self, f: F) -> Self {
        self.iter().copied().map(f).collect()
    }

    fn map_assign<F: FnMut(&mut f64)>(&mut self, mut f: F) {
        for elem in self.iter_mut() {
            f(elem);
        }
    }

    fn new(size: usize) -> Self {
        vec![0.0; size].into()
    }

    fn new_filled(size: usize, value: f64) -> Self {
        vec![value; size].into()
    }

    fn try_apply<F: FnMut(f64, f64) -> f64>(&self, other: &Self, mut f: F) -> Result<Self> {
        if self.len() != other.len() {
            return Err(MatrixOpError::SizeMismatch);
        }
        Ok(self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| f(*a, *b))
            .collect())
    }

    fn try_apply_assign<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) -> Result<()> {
        if self.len() != other.len() {
            return Err(MatrixOpError::SizeMismatch);
        }
        for (a, b) in self.iter_mut().zip(other.iter()) {
            f(a, *b);
        }
        Ok(())
    }

    fn apply<F: FnMut(f64, f64) -> f64>(&self, other: &Self, f: F) -> Self {
        self.try_apply(other, f).unwrap()
    }

    fn apply_assign<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, f: F) {
        self.try_apply_assign(other, f).unwrap();
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

    fn randomize<D: Distribution<f64>, R: Rng>(&mut self, distributions: &D, rng: &mut R) {
        for value in self.iter_mut() {
            *value = distributions.sample(rng);
        }
    }

    fn hadamard_product(&self, other: &Self) -> Self {
        self.apply(other, f64::mul)
    }
}

#[macro_export]
macro_rules! impl_vector {
    ($name:ident, $transpose_to:ident) => {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct $name(Vec<f64>);

        impl FromIterator<f64> for $name {
            fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
                let data: Vec<f64> = iter.into_iter().collect();
                $name(data)
            }
        }

        impl From<Vec<f64>> for $name {
            fn from(data: Vec<f64>) -> Self {
                $name(data)
            }
        }

        impl Deref for $name {
            type Target = [f64];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl Vector for $name {
            type TransposeTo = $transpose_to;
        }
    };
}

#[macro_export]
macro_rules! impl_vector_ops {
    ($name:ident) => {
        impl AddAssign<&$name> for $name {
            fn add_assign(&mut self, other: &$name) {
                self.apply_assign(other, f64::add_assign);
            }
        }

        impl SubAssign<&$name> for $name {
            fn sub_assign(&mut self, other: &$name) {
                self.apply_assign(other, f64::sub_assign);
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                *self += &other;
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                *self -= &other;
            }
        }

        impl Add for &$name {
            type Output = $name;

            fn add(self, other: &$name) -> Self::Output {
                self.apply(other, f64::add)
            }
        }

        impl Sub for &$name {
            type Output = $name;

            fn sub(self, other: &$name) -> Self::Output {
                self.apply(other, f64::sub)
            }
        }

        impl Add for $name {
            type Output = $name;

            fn add(mut self, other: $name) -> Self::Output {
                self += &other;
                self
            }
        }

        impl Sub for $name {
            type Output = $name;

            fn sub(mut self, other: $name) -> Self::Output {
                self -= &other;
                self
            }
        }

        impl Add<&$name> for $name {
            type Output = $name;

            fn add(mut self, other: &$name) -> Self::Output {
                self += other;
                self
            }
        }

        impl Sub<&$name> for $name {
            type Output = $name;

            fn sub(mut self, other: &$name) -> Self::Output {
                self -= other;
                self
            }
        }

        impl Add<$name> for &$name {
            type Output = $name;

            fn add(self, mut other: $name) -> Self::Output {
                other += self;
                other
            }
        }

        impl Sub<$name> for &$name {
            type Output = $name;

            fn sub(self, mut other: $name) -> Self::Output {
                other -= self;
                other
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.map_assign(|x| *x = -*x);
                self
            }
        }

        impl Neg for &$name {
            type Output = $name;

            fn neg(self) -> Self::Output {
                self.map(|x| -x)
            }
        }

        impl DivAssign<f64> for $name {
            fn div_assign(&mut self, other: f64) {
                self.map_assign(|x| *x /= other);
            }
        }

        impl MulAssign<f64> for $name {
            fn mul_assign(&mut self, other: f64) {
                self.map_assign(|x| *x *= other);
            }
        }

        impl Div<f64> for &$name {
            type Output = $name;

            fn div(self, other: f64) -> Self::Output {
                self.map(|x| x / other)
            }
        }

        impl Mul<f64> for &$name {
            type Output = $name;

            fn mul(self, other: f64) -> Self::Output {
                self.map(|x| x * other)
            }
        }

        impl Div<f64> for $name {
            type Output = $name;

            fn div(mut self, other: f64) -> Self::Output {
                self /= other;
                self
            }
        }

        impl Mul<f64> for $name {
            type Output = $name;

            fn mul(mut self, other: f64) -> Self::Output {
                self *= other;
                self
            }
        }
    };
}

impl_vector!(ColumnVector, RowVector);
impl_vector_ops!(ColumnVector);

impl_vector!(RowVector, ColumnVector);
impl_vector_ops!(RowVector);
