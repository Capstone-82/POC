import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const startClarityJob = async ({ file, prompt_complexity, use_case }) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('auto_forward', 'false')
  formData.append('prompt_complexity', prompt_complexity)
  formData.append('use_case', use_case)

  const response = await api.post('/clarity/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}
