use std::ops::{Deref, DerefMut};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("The size implied by the dimensions is too large to allocate.")]
    SizeTooLarge,
    #[error("The size implied by the dimensions does not match the original size.")]
    MismatchedSize,
    #[error("Index out of bounds.")]
    OOB,
}

pub type TensorResult<T> = Result<T, TensorError>;

/// N-dimensional array.
pub struct Tensor<T> {
    /// A list of the dimensions
    dims: Vec<usize>,
    /// Contained data of the tensor.
    data: Vec<T>,
}

impl<T> Tensor<T> {
    fn dims_to_size(dims: &Vec<usize>) -> TensorResult<usize> {
        let mut size: usize = 1;
        // I would have liked to perform fold here, but this allows early returning.
        for val in dims {
            size = size.checked_mul(*val).ok_or(TensorError::SizeTooLarge)?;
        }

        return Ok(size);
    }

    /// Creates a new tensor with a given size.
    pub fn new_fill(dims: Vec<usize>, fill: T) -> TensorResult<Tensor<T>>
    where
        T: Clone,
    {
        let data = vec![fill; Self::dims_to_size(&dims)?];
        return Ok(Tensor { dims, data });
    }

    /// Attempts to change the dimensions of the tensor.
    ///
    /// This requires that the product of the dimensions (the implied size)
    /// matches the size of the original tensor.
    pub fn redim(&mut self, new_dims: Vec<usize>) -> TensorResult<()> {
        let new_len = Self::dims_to_size(&new_dims)?;
        if self.len() == new_len {
            return Err(TensorError::MismatchedSize);
        }
        self.dims = new_dims;
        return Ok(());
    }

    /// Gets the current size of the tensor.
    pub fn len(&self) -> usize {
        return self.data.len();
    }
    /// Get an immutable reference to the dimensions of the tensor.
    pub fn dims(&self) -> &Vec<usize> {
        return &self.dims;
    }
}

impl<T> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return &self.data;
    }
}

impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return &mut self.data;
    }
}

impl<T> FromIterator<T> for Tensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        return Tensor {
            dims: vec![data.len()],
            data,
        };
    }
}
