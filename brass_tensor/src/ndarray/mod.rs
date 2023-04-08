use std::{
    alloc::{self, Layout},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::NonNull,
    slice::SliceIndex,
};
use crate::tensor::Tensor;

pub struct NDArray<T> {
    ptr: NonNull<T>,
    dims: Vec<usize>,
    size: usize,
}

/// An n-dimensional array that is allocated at runtime.
/// 
/// This is the primary implementation of the [Tensor].
impl<T> NDArray<T> {
    /// Allocates the space for the [NDArray], but does not fill. This
    /// means that the array may contain random garbage.
    fn _allocate(dims: Vec<usize>) -> NDArray<T> {
        let size = dims.iter().product();

        let mut new_ndarray: NDArray<T> = NDArray {
            ptr: NonNull::dangling(),
            dims,
            size,
        };

        let layout = Layout::array::<T>(new_ndarray.size).unwrap();

        let new_ptr = unsafe { alloc::alloc(layout) } as *mut T;
        let new_ptr = match NonNull::new(new_ptr) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };

        new_ndarray.ptr = new_ptr;

        return new_ndarray;
    }

    /// Changes the dimension of the [Tensor], which is permitted so long
    /// as the product of the dimensions remain the same.
    pub fn reshape(&mut self, new_dims: Vec<usize>) -> bool {
        let new_size: &usize = &new_dims.iter().product();
        if new_size != &self.size {
            return false;
        }
        self.dims = new_dims;
        return true;
    }

    /// Creates a new [NDArray] with a default value in each spot.
    pub fn new_filled(dims: Vec<usize>, fill: T) -> NDArray<T>
    where T: Clone {
        let mut new_ndarray = NDArray::<T>::_allocate(dims);

        for elem in new_ndarray.deref_mut() {
            *elem = fill.clone();
        }

        return new_ndarray;
    }
}

impl<T> Deref for NDArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) };
    }
}

impl<T> DerefMut for NDArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) };
    }
}

impl<T> Tensor<T> for NDArray<T> {
    fn dims(&self) -> Vec<usize> {
        return self.dims.clone();
    }
}

impl<T, Idx> Index<Idx> for NDArray<T>
where
    Idx: SliceIndex<[T], Output = T>,
{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        return &self.deref()[index];
    }
}

impl<T, Idx> IndexMut<Idx> for NDArray<T>
where
    Idx: SliceIndex<[T], Output = T>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        return &mut self.deref_mut()[index];
    }
}

impl<T: Copy> From<&Vec<T>> for NDArray<T> {
    fn from(value: &Vec<T>) -> Self {
        let dims = vec![value.len()];
        let mut new_tensor = NDArray::_allocate(dims);

        let zipped_iter = new_tensor.iter_mut().zip(value.iter());
        for (to, from) in zipped_iter {
            *to = *from;
        }

        return new_tensor;
    }
}

impl<T> From<Vec<T>> for NDArray<T> {
    fn from(value: Vec<T>) -> Self {
        let dims = vec![value.len()];
        let mut new_tensor = NDArray::_allocate(dims);

        let zipped_iter = new_tensor.iter_mut().zip(value.into_iter());
        for (to, from) in zipped_iter {
            *to = from;
        }

        return new_tensor;
    }
}
