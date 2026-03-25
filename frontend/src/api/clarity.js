import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const startClarityJob = async ({ file }) => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/clarity/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}
