package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"time"
)

// DataFrame es una estructura que almacena los datos del CSV
type DataFrame struct {
	Columns map[string]int
	Rows    [][]string
}

// Load_dataset carga el archivo CSV y lo convierte en un DataFrame
func Load_dataset(path string) *DataFrame {
	// Abre el archivo CSV
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("Error al abrir el archivo:", err)
		return nil
	}
	defer file.Close()

	// Crea un lector CSV
	reader := csv.NewReader(file)

	// Lee todas las filas
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error al leer el archivo:", err)
		return nil
	}

	if len(records) == 0 {
		fmt.Println("Archivo CSV vacío")
		return nil
	}

	// Crear un mapa para las columnas
	columns := make(map[string]int)
	for i, columnName := range records[0] {
		columns[columnName] = i
	}

	// Crear un DataFrame
	dataFrame := &DataFrame{
		Columns: columns,
		Rows:    records[1:], // Ignorar la primera fila que contiene los nombres de las columnas
	}

	return dataFrame
}

// GetRow retorna una fila por su índice
func (df *DataFrame) GetRow(index int) ([]string, error) {
	if index < 0 || index >= len(df.Rows) {
		return nil, fmt.Errorf("índice fuera de rango")
	}
	return df.Rows[index], nil
}

// GetColumn retorna una columna por su nombre
func (df *DataFrame) GetColumn(name string) ([]string, error) {
	colIndex, exists := df.Columns[name]
	if !exists {
		return nil, fmt.Errorf("columna no encontrada")
	}

	column := make([]string, len(df.Rows))
	for i, row := range df.Rows {
		column[i] = row[colIndex]
	}

	return column, nil
}

// GetValue retorna un valor específico dado el índice de la fila y el nombre de la columna
func (df *DataFrame) GetValue(rowIndex int, columnName string) (string, error) {
	row, err := df.GetRow(rowIndex)
	if err != nil {
		return "", err
	}

	colIndex, exists := df.Columns[columnName]
	if !exists {
		return "", fmt.Errorf("columna no encontrada")
	}

	return row[colIndex], nil
}

// measureExecutionTime mide el tiempo de ejecución de una función
func MeasureExecutionTime(name string, f func()) {
    start := time.Now()
    f()
    duration := time.Since(start)
    fmt.Printf("Tiempo de ejecución para %s: %v\n", name, duration)
}