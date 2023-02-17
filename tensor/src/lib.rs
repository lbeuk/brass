use std::ops::{Deref};

pub mod ndarray;

pub trait Tensor<T> {
    /// Constant to define how many dimensions there are. By default, this is
    /// -1, which implies that the number of dimensions is handled dynamically
    /// by the ndims function.
    const NDIMS: isize = -1;

    /// Returns the number of spaces in the Tensor, which is by default the product
    /// of the dimensions. Implementations may override this function for a more efficient
    /// implementation to return the size.
    fn size(&self) -> usize {
        return self.dims_vec().iter().product();
    }

    fn dims_vec(&self) -> Vec<usize>;

    fn ndims(&self) -> isize {
        return Self::NDIMS;
    }
}

/// Some types, such as Vec, do not allow direct access to an underlying,
/// referencable dimension type. Because of this, DimsBetter is not a included
/// in the Tensor trait
pub trait DimsBetter {
    fn dims(&self) -> &[usize];
}

impl<T> Tensor<T> for Vec<T> {
    const NDIMS: isize = 1;

    fn size(&self) -> usize {
        return self.len();
    }

    fn dims_vec(&self) -> Vec<usize> {
        return vec![self.len()];
    }
}