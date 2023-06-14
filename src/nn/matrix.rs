use rand::{thread_rng, Rng};
use std::fmt::{Debug, Formatter, Result};

fn random_from_to(from: f64, to: f64) -> f64 {
    let mut rng = thread_rng();
    (rng.gen::<f64>() * (to - from)) + from
}

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl PartialEq<Matrix> for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.cols != other.cols || self.rows != other.rows {
            return false;
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != other.data[i][j] {
                    return false;
                }
            }
        }
        return true;
    }
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize, from: f64, to: f64) -> Matrix {
        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = random_from_to(from, to);
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn dot_product(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!(
                "Attempted to multiply by matrix of incorrect dimensions. {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }

        let mut res = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }

        res
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if (self.rows != other.rows && other.rows != 1 && self.rows != 1) || (self.cols != other.cols && other.cols != 1 && self.cols != 1)
        {
            panic!(
                "Attempted to add matrix of incorrect dimensions. {}x{} + {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..res.rows {
            for j in 0..res.cols {
                if other.rows == 1 {
                    res.data[i][j] = self.data[i][j] + other.data[0][j];
                } else if other.cols == 1 {
                    res.data[i][j] = self.data[i][j] + other.data[i][0];
                } else {
                    res.data[i][j] = self.data[i][j] + other.data[i][j];
                }
            }
        }

        res
    }

    pub fn sum_by_axis(&self, axis: usize) -> Matrix {
        let mut res: Vec<Vec<f64>> = vec![];

        match axis {
            0 => {
                let mut row: Vec<f64> = vec![];
                for j in 0..self.cols {
                    let mut acc = 0.0;
                    for i in 0..self.rows {
                        acc += self.data[i][j];
                    }
                    row.push(acc);
                }
                res.push(row);
            }
            1 => {
                let mut row: Vec<f64> = vec![];
                for i in 0..self.rows {
                    let mut acc = 0.0;
                    for j in 0..self.cols {
                        acc += self.data[i][j];
                    }
                    row.push(acc);
                }
                res.push(row);
            }
            _ => {
                panic!("Unreachable");
            }
        }

        Matrix::from(res)
    }

    pub fn scalar_multiplication(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to dot multiply by matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        res
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Attempted to subtract matrix of incorrect dimensions. {}x{} - {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }

    pub fn square(&self) -> Matrix {
        self.scalar_multiplication(self)
    }

    // pub fn mean(&self, axis: usize) -> Matrix {
    //     if ![0,1,2].contains(&axis) {
    //         panic!("axis can be only 0, 1 or 2, get = {}", axis);
    //     }
    //     return match axis {
    //         0 => self.clone(),
    //         1 => self.clone(),
    //         2 => self.clone(),
    //         _ => panic!("Unreachable")
    //     }
    // }

    pub fn count(&self) -> usize {
        self.rows * self.cols
    }

    pub fn collect_sum(&self) -> f64 {
        let mut result = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                result += self.data[i][j];
            }
        }
        result
    }

    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect(),
        )
    }

    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "Matrix {{\n{}\n}}",
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<String>>()
                        .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::matrix::Matrix;

    #[test]
    fn add() {
        let A: Matrix = Matrix::from(vec![vec![0.1, 0.2], vec![2.0, 1.0]]);
        let B = Matrix::from(vec![vec![5.0, 4.0], vec![0.5, 0.4]]);
        let result = A.add(&B);
        assert_eq!(result, Matrix::from(vec![vec![5.1, 4.2], vec![2.5, 1.4]]));
    }

    #[test]
    fn add_with_different_dimentions() {
        let A: Matrix = Matrix::from(vec![vec![0.1, 0.2], vec![2.0, 1.0]]);
        let B = Matrix::from(vec![vec![1.0, 1.0]]);
        let result = A.add(&B);
        assert_eq!(result, Matrix::from(vec![vec![1.1, 1.2], vec![3.0, 2.0]]));
    }

    #[test]
    fn sum_by_axis() {
        let A: Matrix = Matrix::from(vec![vec![0.0, 1.0], vec![0.0, 5.0]]);
        let result = A.sum_by_axis(0);
        assert_eq!(result, Matrix::from(vec![vec![0.0, 6.0]]));

        let B: Matrix = Matrix::from(vec![vec![0.0, 1.0], vec![0.0, 5.0]]);
        let result = B.sum_by_axis(1);
        assert_eq!(result, Matrix::from(vec![vec![1.0, 5.0]]));
    }
}
