use std::{ops::{DerefMut, Mul, Add}};

mod tests;

pub trait Tensor<T>: DerefMut<Target = [T]> {
    /// Return an array of the dimensions
    fn dims(&self) -> Vec<usize>;

    /// Return the number of dimensions
    fn ndims(&self) -> usize {
        return self.dims().len();
    }

    /// Return the total number of elements
    fn size(&self) -> usize {
        return self.dims().iter().product();
    }

    /// Returns a tuple containing the results of [dims](Tensor::dims), [ndims](Tensor::dims), as well as the length
    /// of each dimension.
    ///
    /// The primary purpose of this function if to calculate dimlens, but since dims and ndims
    /// are already calculated and are likely to be needed if this function is needed, they are
    /// also included.
    fn dims_ndim_dimlens(&self) -> (Vec<usize>, usize, Vec<usize>) {
        let dims = self.dims();
        let ndims = dims.len();
        let mut lens = vec![1; ndims];
        for j in 1..ndims {
            let i = ndims - j - 1;
            lens[i] = dims[i + 1] * lens[i + 1];
        }
        return (dims, ndims, lens);
    }

    /// Converts a multi-dimensions index into a single dimensional index.
    fn flatten_index(&self, index: &[usize]) -> Option<usize> {
        let (dims, ndims, dimlens) = self.dims_ndim_dimlens();
        if index.len() != ndims {
            return None;
        }

        let mut return_index = 0;
        for i in 0..ndims {
            if index[i] >= dims[i] {
                return None;
            }
            return_index += dimlens[i] * index[i];
        }

        return Some(return_index);
    }

    /// Indexes a vector using multiple dimensions
    fn tensor_index(&self, index: &[usize]) -> Option<&T> {
        return match self.flatten_index(index) {
            Some(idx) => Some(&self[idx]),
            None => None,
        };
    }

    /// Mutably indexes a vector using multiple dimensions
    fn tensor_index_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        return match self.flatten_index(index) {
            Some(idx) => Some(&mut self[idx]),
            None => None,
        };
    }

    /// Returns a dot product of two tensors
    fn dot<'a, Rhs, RhsType>(&'a self, rhs: &'a Rhs) -> Option<<T as Mul<RhsType>>::Output>
    where
        Rhs: Tensor<RhsType>,
        RhsType: Copy,
        T: Mul<RhsType> + Copy,
        <T as Mul<RhsType>>::Output: Add<Output = <T as Mul<RhsType>>::Output> + Default,
    {
        return self.dot_default(<T as Mul<RhsType>>::Output::default(), rhs);
    }

    /// Returns a dot product of two tensors
    fn dot_default<'a, Rhs, RhsType>(&'a self, default: <T as Mul<RhsType>>::Output, rhs: &'a Rhs) -> Option<<T as Mul<RhsType>>::Output>
    where
        Rhs: Tensor<RhsType>,
        RhsType: Copy,
        T: Mul<RhsType> + Copy,
        <T as Mul<RhsType>>::Output: Add<Output = <T as Mul<RhsType>>::Output>,
    {
        // Return early if the dimensions of the two tensors do not match
        if self.dims() != rhs.dims() {
            return None;
        }
        
        let mut dot_iter = self.iter().zip(rhs.iter());

        // Return early if the tensors have no first element, otherwise calculate the
        // first element of the sum. It is easier (and I suspect faster) to calculate
        // the first element seperately than to handle the initial case seperately in the for loop.
        let (l, r) = match dot_iter.next() {
            Some(x) => x,
            None => return None
        };
        let mut sum = (*l)*(*r) + default;

        for (l, r) in dot_iter {
            let prod = (*l)*(*r);
            let prev_plus_prod = sum + prod;
            sum = prev_plus_prod;

        }

        return Some(sum);
    }
}

impl<T> Tensor<T> for Vec<T> {
    fn dims(&self) -> Vec<usize> {
        return vec![self.len()];
    }

    fn ndims(&self) -> usize {
        return self.dims().len();
    }

    fn size(&self) -> usize {
        return self.dims().iter().product();
    }

    fn dims_ndim_dimlens(&self) -> (Vec<usize>, usize, Vec<usize>) {
        let dims = self.dims();
        let ndims = dims.len();
        let mut lens = vec![1; ndims];
        for j in 1..ndims {
            let i = ndims - j - 1;
            lens[i] = dims[i + 1] * lens[i + 1];
        }
        return (dims, ndims, lens);
    }

    fn flatten_index(&self, index: &[usize]) -> Option<usize> {
        let (dims, ndims, dimlens) = self.dims_ndim_dimlens();
        if index.len() != ndims {
            return None;
        }

        let mut return_index = 0;
        for i in 0..ndims {
            if index[i] >= dims[i] {
                return None;
            }
            return_index += dimlens[i] * index[i];
        }

        return Some(return_index);
    }

    fn tensor_index(&self, index: &[usize]) -> Option<&T> {
        return match self.flatten_index(index) {
            Some(idx) => Some(&self[idx]),
            None => None,
        };
    }

    fn tensor_index_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        return match self.flatten_index(index) {
            Some(idx) => Some(&mut self[idx]),
            None => None,
        };
    }
}
