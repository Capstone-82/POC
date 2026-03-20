// src/api/training.js
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const startTrainingJob = async ({ prompt, evaluator_model }) => {
  const response = await api.post('/training/run', { prompt, evaluator_model })
  return response.data; // { job_id }
}

export const startCSVTrainingJob = async ({ file, evaluator_model }) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('evaluator_model', evaluator_model);
  
  const response = await api.post('/training/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data; // { job_id }
}
