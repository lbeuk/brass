use std::{ptr::NonNull, ops::{Deref, DerefMut}, alloc::{self, Layout}};
use crate::{Tensor, DimsBetter};

pub struct NDArray<T> {
    ptr: NonNull<T>,
    dims: Vec<usize>,
    size: usize
}

impl<T> NDArray<T> {
    /// Changes the dimension of the [NDArray], which is permitted so long
    /// as the product of the dimensions remain the same.
    pub fn reshape(&mut self, new_dims: Vec<usize>) -> bool {
        let new_size: &usize = &new_dims.iter().product();
        if new_size != &self.size {
            return false;
        }
        self.dims = new_dims;
        return true;
    } 

    /// Allocates the space for the [NDArray], but does not fill. This
    /// means that the array may contain random garbage.
    fn _allocate(dims: Vec<usize>) -> NDArray<T> {
        let size = dims.iter().product();

        let mut new_ndarray: NDArray<T> = NDArray {
            ptr: NonNull::dangling(),
            dims,
            size
        };

        let layout = Layout::array::<T>(new_ndarray.size).unwrap();

        let new_ptr = unsafe {alloc::alloc(layout)} as *mut T;
        let new_ptr = match NonNull::new(new_ptr) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout)
        };
        
        new_ndarray.ptr = new_ptr;

        return new_ndarray;
    }
}

impl<T: Clone> NDArray<T> {
    /// Creates a new [NDArray] with a default value in each spot.
    pub fn new_fill(dims: Vec<usize>, fill: T) -> NDArray<T> {
        let mut new_ndarray = NDArray::<T>::_allocate(dims);

        for elem in new_ndarray.deref_mut() {
            *elem = fill.clone();
        }

        return new_ndarray;
    }
}

impl<T> Tensor<T> for NDArray<T> {
    fn size(&self) -> usize {
        return self.size;
    }

    fn dims_vec(&self) -> Vec<usize> {
        return self.dims.clone();
    }

    fn ndims(&self) -> isize {
        return self.dims.len().try_into().unwrap();    
    }
}

impl<T> DimsBetter for NDArray<T> {
    fn dims(&self) -> &[usize] {
        return &self.dims;
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