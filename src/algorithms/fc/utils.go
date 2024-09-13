package fc

import (    "fmt"
)

// DenseMatrix representa una matriz densa.
type DenseMatrix struct {
    data []float64
    rows int
    cols int
}

// MakeDenseMatrix crea una nueva matriz densa con los datos proporcionados.
func MakeDenseMatrix(data []float64, rows, cols int) *DenseMatrix {
    if len(data) != rows*cols {
        panic("data length does not match dimensions")
    }
    return &DenseMatrix{
        data: data,
        rows: rows,
        cols: cols,
    }
}

// Array devuelve los datos de la matriz como un slice unidimensional.
func (m *DenseMatrix) Array() []float64 {
    return m.data
}

// Rows devuelve el número de filas de la matriz.
func (m *DenseMatrix) Rows() int {
    return m.rows
}

// Cols devuelve el número de columnas de la matriz.
func (m *DenseMatrix) Cols() int {
    return m.cols
}

// Get devuelve el valor en la posición (i, j).
func (m *DenseMatrix) Get(i, j int) float64 {
    if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
        panic("index out of range")
    }
    return m.data[i*m.cols+j]
}

// Set establece el valor en la posición (i, j).
func (m *DenseMatrix) Set(i, j int, value float64) {
    if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
        panic("index out of range")
    }
    m.data[i*m.cols+j] = value
}

// GetRowVector devuelve la fila i como un vector.
func (m *DenseMatrix) GetRowVector(i int) *DenseMatrix {
    if i < 0 || i >= m.rows {
        panic("index out of range")
    }
    row := make([]float64, m.cols)
    copy(row, m.data[i*m.cols:(i+1)*m.cols])
    return MakeDenseMatrix(row, 1, m.cols)
}

// GetColVector devuelve la columna j como un vector.
func (m *DenseMatrix) GetColVector(j int) *DenseMatrix {
    if j < 0 || j >= m.cols {
        panic("index out of range")
    }
    col := make([]float64, m.rows)
    for i := 0; i < m.rows; i++ {
        col[i] = m.data[i*m.cols+j]
    }
    return MakeDenseMatrix(col, m.rows, 1)
}

// String devuelve una representación en cadena de la matriz.
func (m *DenseMatrix) String() string {
    result := ""
    for i := 0; i < m.rows; i++ {
        for j := 0; j < m.cols; j++ {
            result += fmt.Sprintf("%f ", m.Get(i, j))
        }
        result += "\n"
    }
    return result
}