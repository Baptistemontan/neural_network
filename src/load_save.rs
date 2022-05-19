use std::{error::Error, fmt::Display, path::Path};

use serde::{de::DeserializeOwned, Serialize};

#[derive(Debug)]
pub enum LoadSaveError {
    Io(std::io::Error),
    Serde(serde_json::Error),
}

impl Display for LoadSaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadSaveError::Io(e) => write!(f, "IO error: {}", e),
            LoadSaveError::Serde(e) => write!(f, "Serde error: {}", e),
        }
    }
}

impl Error for LoadSaveError {}

pub trait Savable: Serialize {
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), LoadSaveError> {
        let contents = serde_json::to_string(self).map_err(LoadSaveError::Serde)?;
        std::fs::write(path, contents).map_err(LoadSaveError::Io)
    }
}

impl<T> Savable for T where T: Serialize {}

pub trait Loadable: DeserializeOwned {
    fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoadSaveError> {
        let contents = std::fs::read_to_string(path).map_err(LoadSaveError::Io)?;
        serde_json::from_str(&contents).map_err(LoadSaveError::Serde)
    }
}

impl<T> Loadable for T where T: DeserializeOwned {}
