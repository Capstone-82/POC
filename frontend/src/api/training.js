// src/api/training.js
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const startTrainingJob = async ({ prompt, prompt_complexity, use_case, clarity }) => {
  const response = await api.post('/training/run', {
    prompt,
    prompt_complexity,
    use_case,
    clarity,
  })
  return response.data; // { job_id }
}

export const startCSVTrainingJob = async ({ file, prompt_complexity, use_case }) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('prompt_complexity', prompt_complexity);
  formData.append('use_case', use_case);
  
  const response = await api.post('/training/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data; // { job_id }
}

export const startMultiCSVTrainingJob = async ({ files, prompt_complexity, use_case, delay_ms }) => {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  formData.append('prompt_complexity', prompt_complexity)
  formData.append('use_case', use_case)
  formData.append('delay_ms', String(delay_ms))

  const response = await api.post('/training/upload-multi', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}
