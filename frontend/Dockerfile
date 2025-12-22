# Multi-stage build for the React (Vite) frontend
FROM node:18-alpine AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --silent

# Copy source and build
COPY . .
RUN npm run build

# Production image
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# Provide SPA fallback and proxy /api to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
