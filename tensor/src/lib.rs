use std::ops::{Deref};

pub mod ndarray;

/// A general abstraction of an [NDArray](ndarray::NDArray) that
/// permits for [Tensor] operations to be applied on other types,
/// such as vectors
pub trait Tensor<T> {
    /// Constant to define how many dimensions there are. By default, this is
    /// -1, which implies that the number of dimensions is handled dynamically
    /// by the ndims function.
    const NDIMS: isize = -1;

    /// Returns the number of spaces in the [Tensor], which is by default the product
    /// of the dimensions. Implementations may override this function for a more efficient
    /// implementation to return the size.
    fn size(&self) -> usize {
        return self.dims_vec().iter().product();
    }

    /// Returns a vector that contains the dimensions of the Tensor.
    /// The [dims](DimsBetter::dims) function, from the [DimsBetter] trait
    /// is preferable if an underlying type may implement it.
    fn dims_vec(&self) -> Vec<usize>;

    /// Returns the number of dimensions that a [Tensor] has. The default
    /// implementation should be overrided when [NDIMS](Tensor::NDIMS) is not
    /// implemented and creating a new size vector is inefficient.
    fn ndims(&self) -> isize {
        if Self::NDIMS != -1 {
            return Self::NDIMS;
        }
        return self.dims_vec().len().try_into().unwrap();
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