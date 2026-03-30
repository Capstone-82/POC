# Frontend Specification

## Stack

- React 19
- Vite 8
- React Router 7
- Tailwind CSS
- Framer Motion
- Axios
- React Dropzone
- Lucide React

The frontend is a single-page application rooted in [frontend/src/App.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\App.jsx).

## Routes

The app defines three user-facing routes:

- `/training`
- `/inference`
- `/clarity`

The root path redirects to `/training`.

Global chrome is supplied by [frontend/src/components/Navbar.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\Navbar.jsx).

## Visual System

The UI uses a dark, glassmorphism-style presentation with:

- translucent cards via `.glass` and `.glass-card`
- blue/cyan gradient accents
- large dashboard-like telemetry layouts
- Framer Motion transitions for route and section entry

Base styling lives in [frontend/src/index.css](c:\Users\Musharraf\Documents\POC\frontend\src\index.css).

## API Layer

### Training API

File: [frontend/src/api/training.js](c:\Users\Musharraf\Documents\POC\frontend\src\api\training.js)

Exports:

- `startTrainingJob`
- `startCSVTrainingJob`
- `startMultiCSVTrainingJob`

Base URL:

- `http://localhost:8000/api`

### Inference API

File: [frontend/src/api/inference.js](c:\Users\Musharraf\Documents\POC\frontend\src\api\inference.js)

Exports:

- `getRecommendation`
- `getRecommendationOptions`

### Clarity API

File: [frontend/src/api/clarity.js](c:\Users\Musharraf\Documents\POC\frontend\src\api\clarity.js)

Exports:

- `startClarityJob`

The current clarity UI always sends `auto_forward=false`.

## Page Details

### Benchmark page

File: [frontend/src/pages/Training.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\Training.jsx)

This page is the main benchmark orchestration surface.

State it manages:

- prompt complexity
- use case
- clarity for single prompt mode
- input mode: single or CSV
- single prompt text
- one selected CSV
- optional CSV queue
- delay between queued CSV files
- streaming logs
- failed model logs
- prompt progress
- file progress
- running, done, and error flags

User controls:

- Use case selector cards
- Prompt complexity selector cards
- Prompt clarity selector cards in single-prompt mode
- Input strategy toggle
- Prompt textarea
- CSV uploader for one file
- CSV uploader for multiple sequential files
- Delay input for queued CSV execution
- Run and stop controls

Backend interactions:

- Starts a job with one of the training endpoints
- Opens an `EventSource` to `/api/training/stream/{job_id}`

Handled stream events:

- `progress`
- `file_started`
- `file_done`
- `file_delay`
- `model_failed`
- `done`
- `error`

Displayed results:

- overall progress bar
- per-file queue status
- live model telemetry
- list of failed models
- success or failure toast

### Recommendation page

File: [frontend/src/pages/Inference.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\Inference.jsx)

This page is the recommendation workspace.

Startup behavior:

- Calls `getRecommendationOptions()` on mount
- Populates available use cases and compatible baseline models
- Auto-selects the first use case when catalog data loads
- Auto-selects the first compatible model if the previous choice no longer fits the selected use case

User controls:

- Prompt textarea
- Use-case card selector
- Searchable baseline model menu
- Recommendation submit button

Displayed baseline metadata:

- average accuracy
- median cost
- median latency
- sample count

Displayed recommendation data:

- recommended model/provider
- complexity
- clarity
- filter level
- data source
- comparison against the current baseline
- switching policy summary
- warnings

### Clarity page

File: [frontend/src/pages/ClarityLabeling.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\ClarityLabeling.jsx)

This page supports dataset preparation for prompt clarity labeling.

User controls:

- upload a CSV with a `prompt` column
- start or stop the stream

Backend interactions:

- Calls `/api/clarity/upload`
- Opens an `EventSource` to `/api/clarity/stream/{job_id}`

Handled stream events:

- `started`
- `chunk_ready`
- `done`
- `error`

Displayed results:

- chunk progress
- per-chunk downloads
- ZIP download
- final success or failure state

## Shared Components

### `Navbar`

File: [frontend/src/components/Navbar.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\Navbar.jsx)

Responsibilities:

- global top navigation
- route highlighting
- app brand display

### `PromptInput`

File: [frontend/src/components/PromptInput.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\PromptInput.jsx)

Responsibilities:

- benchmark prompt entry
- character count display

### `CSVUpload`

File: [frontend/src/components/CSVUpload.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\CSVUpload.jsx)

Responsibilities:

- drag-and-drop CSV selection
- single-file or multi-file mode
- simple file type validation by `.csv` extension
- selected file list display

Important note:

- In single-file mode it returns one `File`
- In multi-file mode it returns an array of `File` objects

### `LiveLog`

File: [frontend/src/components/LiveLog.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\LiveLog.jsx)

Responsibilities:

- render benchmark SSE log events
- support filtering by prompt complexity
- auto-scroll as new log items arrive

Each log card shows:

- prompt progress index
- model ID
- complexity
- accuracy score
- latency
- estimated cost
- provider

### `RecommendationOutput`

File: [frontend/src/components/RecommendationOutput.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\RecommendationOutput.jsx)

Responsibilities:

- render recommendation response details
- compare current baseline against recommended model
- format delta cards for accuracy, cost, and latency
- display warnings and policy rationale

### `EvaluatorDropdown`

File: [frontend/src/components/EvaluatorDropdown.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\EvaluatorDropdown.jsx)

Status:

- Present in the codebase
- Not currently used by the active training page

It appears to be a legacy component from an earlier version of the UI where the user explicitly chose a response judge model.

## Frontend State and Data Contracts

### Benchmarking contract

The training UI sends:

- `prompt`
- `prompt_complexity`
- `use_case`
- `clarity`

For CSV mode, the frontend sends:

- uploaded file
- `prompt_complexity`
- `use_case`

The CSV itself is expected to contain prompt rows and clarity labels for the single-file and multi-file upload routes.

### Recommendation contract

The inference UI sends:

- `prompt`
- `use_case`
- `current_model`

It expects a rich response containing enough detail to explain why the recommendation was made.

### Clarity contract

The clarity UI uploads a CSV and expects chunk-based progress with downloadable artifacts.

## Operational Notes

- Backend URLs are currently hardcoded to `http://localhost:8000/api`.
- There is no environment-based frontend API configuration yet.
- SSE streams are opened directly with `EventSource`.
- If the backend disconnects, pages show a user-visible error message but do not automatically reconnect.
- The training page includes a "View Results" button in the success state, but it is currently presentation-only and does not navigate anywhere.

## Running the Frontend

From the repo root:

```bash
cd frontend
npm install
npm run dev
```

Vite runs by default on:

- `http://localhost:5173`
