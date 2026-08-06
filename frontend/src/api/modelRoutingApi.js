const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/** Fetch all app configs from the backend (SSM or local fallback). */
export async function fetchApps() {
  const res = await fetch(`${BASE}/api/model-routing/apps`)
  if (!res.ok) throw new Error(`Failed to fetch apps: ${res.status}`)
  return res.json()
}

/** Create or update an app config. */
export async function upsertApp(config) {
  const res = await fetch(`${BASE}/api/model-routing/apps`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to save app: ${res.status}`)
  }
  return res.json()
}

/** Delete an app config by app_id and env. */
export async function deleteApp(appId, env) {
  const res = await fetch(`${BASE}/api/model-routing/apps/${appId}/${env}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to delete app: ${res.status}`)
  }
  return res.json()
}

/** Dry-run the routing decision for a prompt + model. */
export async function testRoute(payload) {
  const res = await fetch(`${BASE}/api/model-routing/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Test route failed: ${res.status}`)
  }
  return res.json()
}

/** Return all model IDs from the registry. */
export async function fetchAvailableModels() {
  const res = await fetch(`${BASE}/api/model-routing/models`)
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`)
  return res.json()
}
