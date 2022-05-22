use serde::{Deserialize, Serialize};

use crate::vector::{ColumnVector, Vector};

pub trait ActivationFunction: Send + Sync {
    fn activate(&self, x: f64) -> f64;

    fn activate_prime(&self, x: f64) -> f64;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SigmoidActivation;

impl ActivationFunction for SigmoidActivation {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn activate_prime(&self, x: f64) -> f64 {
        self.activate(x) * (1.0 - self.activate(x))
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct TanhActivation;

impl ActivationFunction for TanhActivation {
    fn activate(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn activate_prime(&self, x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReLUActivation;

impl ActivationFunction for ReLUActivation {
    fn activate(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn activate_prime(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeakyReLUActivation(pub f64);

impl ActivationFunction for LeakyReLUActivation {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            x * self.0
        }
    }

    fn activate_prime(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            self.0
        }
    }
}

pub trait OutputActivationFunction: Send + Sync {
    fn activate(&self, x: ColumnVector) -> ColumnVector;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftMax;

impl OutputActivationFunction for SoftMax {
    fn activate(&self, mut x: ColumnVector) -> ColumnVector {
        let sum: f64 = x.iter().copied().map(f64::exp).sum();
        x.map_assign(|x| *x = x.exp() / sum);
        x
    }
}
