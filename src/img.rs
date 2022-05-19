use std::fmt::Display;
use std::num::{ParseIntError, ParseFloatError};
use std::{path::Path, error::Error, fs::File};

use csv::{Reader, StringRecord};
use csv::Error as CsvError;
use csv::StringRecordsIntoIter;

use crate::matrix::Matrix;

pub struct Img {
    pixels: Matrix,
    label: usize,
}

pub struct ImgLoader {
    records: StringRecordsIntoIter<File>,
}

#[derive(Debug)]
pub enum ParsingError {
    Csv(CsvError),
    ParseInt(ParseIntError),
    ParseFloat(ParseFloatError),
    EmptyRecord
}

impl Display for ParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsingError::Csv(e) => write!(f, "CSV error: {}", e),
            ParsingError::ParseInt(e) => write!(f, "ParseInt error: {}", e),
            ParsingError::ParseFloat(e) => write!(f, "ParseFloat error: {}", e),
            ParsingError::EmptyRecord => write!(f, "Empty record"),
        }
    }
}

impl Error for ParsingError {}

impl Img {
    pub fn new(pixels: Matrix, label: usize) -> Self {
        Img { pixels, label }
    }

    pub fn get_label(&self) -> usize {
        self.label
    }

    pub fn get_pixels(&self) -> &Matrix {
        &self.pixels
    }

    pub fn load_from_csv<P: AsRef<Path>>(path: P) -> Result<ImgLoader, CsvError> {
        let reader = Reader::from_path(path)?;
        let records = reader.into_records();
        Ok(ImgLoader::new(records))
    }
}

impl ImgLoader {
    pub fn new(records: StringRecordsIntoIter<File>) -> Self {
        ImgLoader { records }
    }
}

impl Iterator for ImgLoader {
    type Item = Result<Img, ParsingError>;

    fn next(&mut self) -> Option<Self::Item> {
        fn parse(record: StringRecord) -> Result<Img, ParsingError> {
            let label = record.get(0)
                .ok_or(ParsingError::EmptyRecord)?
                .parse::<usize>()
                .map_err(ParsingError::ParseInt)?;
            let pixels = record.iter().skip(1).enumerate().try_fold(
                Matrix::new(28, 28),
                |mut mat, (index, pixel)| {
                    let pixel: f64 = match pixel.parse() {
                        Ok(pixel) => pixel,
                        Err(e) => return Err(ParsingError::ParseFloat(e)),
                    };
                    mat[(index / 28, index % 28)] = pixel / 256.0;
                    Ok(mat)
                },
            )?;
            Ok(Img::new(pixels, label))
        }
        self.records.next().map(|record| {
            record.map_err(ParsingError::Csv).and_then(parse)
        })
    }
}
