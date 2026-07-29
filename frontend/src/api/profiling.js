import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

export const profileAndRoute = async ({
  prompt,
  max_tokens = null,
  include_legacy = false,
  top_n = 3,
}) => {
  const response = await api.post('/profiling/route', {
    prompt,
    max_tokens: max_tokens ? parseInt(max_tokens, 10) : null,
    include_legacy,
    top_n: parseInt(top_n, 10),
  })
  return response.data
}
