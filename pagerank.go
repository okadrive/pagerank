package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main() {
	d := 0.85
	ε := 0.00001
	n := 6

	data := []float64{
		0, 0, 1, 0, 0, 0,
		1, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 1,
		0, 0, 0, 0, 1, 0,
	}
	N := mat.NewDense(n, n, data)
	fmt.Println("original:")
	matPrint(N)

	// n × n の空行列を作成
	A1 := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		col := N.ColView(i)
		colSum := mat.Norm(col, 1)
		// deal with dangling node
		if colSum == 0 {
			src := make([]float64, n)
			for j := 0; j < n; j++ {
				src[j] = 1 / float64(n)
			}
			A1.SetCol(i, src)
		} else {
			src := make([]float64, n)
			for j := 0; j < n; j++ {
				src[j] = N.At(j, i) / colSum
			}
			A1.SetCol(i, src)
		}
	}

	A2 := mat.NewDense(n, n, nil)

	// receiver matrix
	r1 := mat.NewDense(n, n, nil)
	r2 := mat.NewDense(n, n, nil)
	r3 := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			r3.Set(i, j, float64(1))
		}
	}

	r1.Scale(d, A1)
	r2.Scale((1-d)/float64(n), r3)
	A2.Add(r1, r2)

	randVector := make([]float64, n)
	for i := range randVector {
		randVector[i] = rand.Float64()
	}
	r0 := mat.NewDense(n, 1, randVector)

	prevR := r0
	for {
		nextR := mat.NewDense(n, 1, nil)
		nextR.Mul(A2, prevR)
		// normalization for sum of norm equal to 1
		sum := float64(0)
		for i := 0; i < n; i++ {
			sum += math.Abs(nextR.At(i, 0))
		}
		nextR.Scale(1 / sum, nextR)
		diff := mat.NewDense(n, 1, nil)
		diff.Sub(nextR, prevR)

		sum = float64(0)
		for i := 0; i < n; i++ {
			sum += math.Abs(diff.At(i, 0))
		}
		if sum < ε {
			fmt.Println("pageRank:")
			matPrint(nextR)
			return
		}
		prevR = nextR
	}
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
	fmt.Println()
}
