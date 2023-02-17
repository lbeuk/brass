use std::{ptr::NonNull, ops::{Deref, DerefMut}, alloc::{self, Layout}};
use crate::Tensor;

pub struct NDArray<T> {
    ptr: NonNull<T>,
    dims: Vec<usize>,
    size: usize
}

impl<T> NDArray<T> {
    /// Allocates the space for the NDArray, but does not fill. This
    /// means that the array may contain random garbage.
    fn allocate(dims: Vec<usize>) -> Option<NDArray<T>> {
        // Create empty NDArray, useful to get size
        let mut new_ndarray: NDArray<T> = NDArray {
            ptr: NonNull::dangling(),
            dims,
            size: 0
        };

        new_ndarray.size = new_ndarray.size();
        let layout = match Layout::array::<T>(new_ndarray.size) {
            Ok(l) => l,
            Err(_) => return None
        };

        let new_ptr = unsafe {alloc::alloc(layout)} as *mut T;
        let new_ptr = match NonNull::new(new_ptr) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout)
        };
        
        new_ndarray.ptr = new_ptr;

        return Some(new_ndarray);
    }
}

impl<T: Clone> NDArray<T> {
    /// Creates a new NDArray with a default value in each spot.
    pub fn new_fill(dims: Vec<usize>, fill: T) -> Option<NDArray<T>> {
        let mut new_ndarray = match NDArray::<T>::allocate(dims) {
            Some(n) => n,
            None => return None
        };

        for elem in new_ndarray.deref_mut() {
            *elem = fill.clone();
        }

        return Some(new_ndarray);
    }
}

impl<T> Tensor<T> for NDArray<T> {
    fn size(&self) -> usize {
        return self.dims.iter().fold(1, |size, i| size * i);
    }

    fn dims_vec(&self) -> Vec<usize> {
        return self.dims.clone();
    }

    fn ndims(&self) -> isize {
        return self.dims.len().try_into().unwrap();    
    }
}

impl<T> Deref for NDArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
        };
    }
}

impl<T> DerefMut for NDArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
        };
    }
}

impl<T> Drop for NDArray<T> {
    fn drop(&mut self) {
        todo!("Implement Drop");
    }
}