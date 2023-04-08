#[cfg(test)]

#[test]
fn default_int() {
    use crate::tensor::Tensor;

    use super::Matrix;
    let width = 4;
    let diaganol_val = 20;
    // The desired behavior is that a ints default is 0
    let default_val = 0;

    let matrix = Matrix::diaganol(diaganol_val, width, width);
    for row in 0..width {
        for col in 0..width {
            let val = matrix.tensor_index(&[row, col]).unwrap();
            if row == col {
                assert_eq!(*val, diaganol_val);
            }
            else {
                assert_eq!(*val, default_val);
            }
        }
    }
}