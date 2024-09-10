package fc

import (
	"math"
	"sort"
	"sync"
)

// Estructura para almacenar las calificaciones
type RatingConcurrent struct {
	data map[string]map[string]float64
}

// Constructor para la estructura RatingConcurrent
func NewRatingsConcurrent() *RatingConcurrent {
	return &RatingConcurrent{data: make(map[string]map[string]float64)}
}

// Añadir una calificación
func (r *RatingConcurrent) AddRatingConcurrent(user, item string, rating float64) {
	if _, exists := r.data[user]; !exists {
		r.data[user] = make(map[string]float64)
	}
	r.data[user][item] = rating
}

// Calcular la similitud del coseno entre dos usuarios
func CosineSimilarityConcurrent(ratings *RatingConcurrent, user1, user2 string) float64 {
	commonItems := make(map[string]bool)
	for item := range ratings.data[user1] {
		if _, exists := ratings.data[user2][item]; exists {
			commonItems[item] = true
		}
	}

	if len(commonItems) == 0 {
		return 0.0
	}

	var sum1, sum2, sumProduct float64
	for item := range commonItems {
		sum1 += ratings.data[user1][item] * ratings.data[user1][item]
		sum2 += ratings.data[user2][item] * ratings.data[user2][item]
		sumProduct += ratings.data[user1][item] * ratings.data[user2][item]
	}

	return sumProduct / (math.Sqrt(sum1) * math.Sqrt(sum2))
}

// Tarea para calcular recomendaciones
type RecommendationTaskConcurrent struct {
	item       string
	score      float64
	similarity float64
}

// Generar recomendaciones para un usuario concurrentemente
func RecommendConcurrent(ratings *RatingConcurrent, user string, k int) []string {
	scores := make(map[string]float64)
	similaritySums := make(map[string]float64)
	var mu sync.Mutex
	var wg sync.WaitGroup

	taskCh := make(chan RecommendationTaskConcurrent)

	// Función para procesar cada tarea de recomendación
	go func() {
		for task := range taskCh {
			mu.Lock()
			scores[task.item] += task.similarity * task.score
			similaritySums[task.item] += task.similarity
			mu.Unlock()
		}
	}()

	// Llenar las tareas concurrentes
	for otherUser := range ratings.data {
		if otherUser == user {
			continue
		}

		wg.Add(1)
		go func(otherUser string) {
			defer wg.Done()
			similarity := CosineSimilarityConcurrent(ratings, user, otherUser)
			if similarity <= 0 {
				return
			}

			for item, rating := range ratings.data[otherUser] {
				if _, exists := ratings.data[user][item]; exists {
					continue
				}
				taskCh <- RecommendationTaskConcurrent{item: item, score: rating, similarity: similarity}
			}
		}(otherUser)
	}

	wg.Wait()
	close(taskCh)

	recommendations := make([]string, 0, k)
	for item := range scores {
		scores[item] /= similaritySums[item]
		recommendations = append(recommendations, item)
	}

	// Ordenar las recomendaciones por puntuación (descendente)
	sort.Slice(recommendations, func(i, j int) bool {
		return scores[recommendations[i]] > scores[recommendations[j]]
	})

	if len(recommendations) > k {
		return recommendations[:k]
	}
	return recommendations
}