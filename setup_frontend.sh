#!/bin/bash

# LLM Recommender — Frontend Scaffold
# Run from project root: bash setup_frontend.sh

echo "Setting up frontend..."

# Create Vite React app
npm create vite@latest frontend -- --template react
cd frontend

# Install dependencies
npm install react-router-dom axios react-dropzone
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Create folder structure
mkdir -p src/pages
mkdir -p src/components
mkdir -p src/api

# Create empty files
touch src/pages/Training.jsx
touch src/pages/Inference.jsx

touch src/components/Navbar.jsx
touch src/components/EvaluatorDropdown.jsx
touch src/components/PromptInput.jsx
touch src/components/CSVUpload.jsx
touch src/components/LiveLog.jsx
touch src/components/RecommendationOutput.jsx

touch src/api/training.js
touch src/api/inference.js

echo ""
echo "Done. Structure created:"
echo ""
find src -type f | sort
echo ""
echo "Next: cd frontend && npm run dev"