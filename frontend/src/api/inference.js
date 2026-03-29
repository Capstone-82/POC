// src/api/inference.js
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const getRecommendation = async ({ prompt, use_case, current_model }) => {
  const response = await api.post('/inference/recommend', {
    prompt,
    use_case,
    current_model
  })
  return response.data;
}

export const getRecommendationOptions = async () => {
  const response = await api.get('/inference/options')
  return response.data
}
