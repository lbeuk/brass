#[cfg(test)]

use std::vec;

use crate::{ndarray::NDArray, tensor::Tensor};

fn test_ndarray() -> NDArray<usize> {
    return NDArray::new_filled(vec![1, 2, 3, 4, 5], 1);
}

#[test]
fn test_dim_lens() {
    let x = test_ndarray();
    let (_, _, dimlens) = x.dims_ndim_dimlens();
    println!("{:?}", dimlens);
    assert_eq!(dimlens, vec![120, 60, 20, 5, 1]);
}

#[test]
fn test_flatten_index() {
    let x = test_ndarray();
    let passing_indexes = [([0, 1, 2, 3, 4], 119), ([0, 0, 0, 0, 0], 0)];
    let failing_indexes = [
        [1, 1, 2, 3, 4],
        [0, 2, 2, 3, 4],
        [0, 1, 3, 3, 4],
        [0, 1, 2, 4, 4],
        [0, 1, 2, 3, 5],
    ];

    for (idx, idx_calc) in passing_indexes {
        assert_eq!(x.flatten_index(&idx), Some(idx_calc));
    }

    for idx in failing_indexes {
        assert_eq!(x.flatten_index(&idx), None);
    }
}

#[test]
fn test_dot() {
    let left = vec![1,2,3];
    let right = vec![2,3,4];
    let prod = left.dot(&right);
    assert_eq!(prod, Some(20));
}