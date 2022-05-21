use crate::vector::{ColumnVector, Vector};

pub trait ActivationFunction: Send + Sync {
    fn activate(x: f64) -> f64;

    fn activate_prime(x: f64) -> f64;
}

pub struct SigmoidActivation;

impl ActivationFunction for SigmoidActivation {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn activate_prime(x: f64) -> f64 {
        Self::activate(x) * (1.0 - Self::activate(x))
    }
}

pub struct TanhActivation;

impl ActivationFunction for TanhActivation {
    fn activate(x: f64) -> f64 {
        x.tanh()
    }

    fn activate_prime(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

pub struct ReLUActivation;

impl ActivationFunction for ReLUActivation {
    fn activate(x: f64) -> f64 {
        x.max(0.0)
    }

    fn activate_prime(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub trait OutputActivationFunction: Send + Sync {
    fn activate(x: &ColumnVector) -> ColumnVector;
}

pub struct SoftMax;

impl OutputActivationFunction for SoftMax {
    fn activate(x: &ColumnVector) -> ColumnVector {
        let sum = x.iter().sum::<f64>();
        x.map(|x| x / sum)
    }
}