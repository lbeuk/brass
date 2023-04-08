use std::{ops::{Deref, DerefMut}};
use crate::{ndarray::{NDArray}, tensor::Tensor};

mod tests;

pub struct Matrix<T> {
    data: NDArray<T>
}

impl<T> Matrix<T> {
    /// Creates a diaganol [Matrix], using the value of [Default::default()] to fill in
    /// the values not along the diaganol.
    pub fn diaganol(dvalue: T, nrow: usize, ncol: usize) -> Matrix<T>
    where T: Clone + Default {
        return Self::diaganol_default(dvalue, T::default(), nrow, ncol);
    }

    /// Creates a diaganol [Matrix] of an arbitrary row and column number.
    pub fn diaganol_default(dvalue: T, default: T, nrow: usize, ncol: usize) -> Matrix<T>
    where T: Clone {
        let mut data = NDArray::new_filled(vec![nrow, ncol], default);
        let mut i = 0;
        while  i < nrow*ncol {
            data[i] = dvalue.clone();
            i += 1 + ncol;
        }
        return Matrix { data };
    }
}

impl<T> Deref for Matrix<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return self.data.deref();
    }
}

impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return self.data.deref_mut();
    }
}

impl<T> Tensor<T> for Matrix<T> {
    fn dims(&self) -> Vec<usize> {
        return self.data.dims();
    }
}