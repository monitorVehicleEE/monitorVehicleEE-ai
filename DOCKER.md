# Docker deploy

## CPU

Build and run the sequential server, `src.app.app:app`:

```powershell
docker compose up -d --build
```

Open:

```text
http://localhost:8001
```

## GPU

Requires NVIDIA driver and NVIDIA Container Toolkit.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Environment

The default compose file mounts:

- `./model` to `/app/model`
- `./dataset/vehicle/videos` to `/app/dataset/vehicle/videos`
- `./dataset/output_test` to `/app/dataset/output_test`

Important variables:

- `DEVICE=cpu` for CPU, or `DEVICE=0` for the first GPU
- `CAMERA_API_URL=http://host.docker.internal:8000` when the backend runs on the host machine
- `VIDEO_BASE_DIR=/app/dataset/vehicle/videos`
- `OUTPUT_DIR=/app/dataset/output_test`
- `EVENT_IMAGES_DIR=/app/dataset/output_test/events`

## Manual image build

```powershell
docker build -t monitor-vehicle-ee-ai:latest .
docker run --rm -p 8001:8001 `
  -e DEVICE=cpu `
  -e CAMERA_API_URL=http://host.docker.internal:8000 `
  -v ${PWD}/model:/app/model:ro `
  -v ${PWD}/dataset/vehicle/videos:/app/dataset/vehicle/videos:ro `
  -v ${PWD}/dataset/output_test:/app/dataset/output_test `
  monitor-vehicle-ee-ai:latest
```

## Deploy to EzyCloudX

EzyCloudX has GPU cloud options and a Docker GPU product. If you use a full rented machine, clone this repository on the machine and run the GPU compose command above. If you use the Docker GPU product, build and push the image to a registry first:

```powershell
docker build -t <your-registry>/monitor-vehicle-ee-ai:latest .
docker push <your-registry>/monitor-vehicle-ee-ai:latest
```

Run the container with these settings:

```text
Port: 8001
GPU: enabled
DEVICE=0
CAMERA_API_URL=<your backend API URL>
AI_PUBLIC_URL=http://<public server ip or domain>:8001
VIDEO_BASE_DIR=/app/dataset/vehicle/videos
OUTPUT_DIR=/app/dataset/output_test
EVENT_IMAGES_DIR=/app/dataset/output_test/events
```

For video file sources, upload/mount videos into `/app/dataset/vehicle/videos`. For RTSP/HTTP camera sources, make sure the rented GPU machine can reach those camera URLs.

## Backend Server

This backend server is separate from the AI server and talks to it through HTTP APIs.

Build and run backend only:

```powershell
docker compose -f docker-compose.backend.yml up -d --build
```

Build and run AI + backend in the same Docker network:

```powershell
docker compose -f docker-compose.yml -f docker-compose.backend.yml up -d --build
```

The backend image defaults to:

```text
APP_MODULE=src.app.main:main
PORT=8000
AI_PUBLIC_URL=http://ai-server:8001
```

If your FastAPI object is named `app` instead of `main`, set:

```text
APP_MODULE=src.app.main:app
```

## Full Stack Local/VPS

Run AI + backend + PostgreSQL together from this repo, with the backend repo as a sibling folder:

```text
workspace/
  monitorVehicleEE-ai/
  monitorVehicleEE-be-fta/
```

Create your env file:

```powershell
Copy-Item .env.stack.example .env
```

Local CPU:

```powershell
docker compose -f docker-compose.stack.yml --env-file .env up -d --build
```

Local/VPS GPU:

```powershell
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env up -d --build
```

Services:

```text
Backend: http://localhost:8000
AI:      http://localhost:8001
DB:      127.0.0.1:5432
```

For VPS, update `.env`:

```text
POSTGRES_PASSWORD=<strong-password>
JWT_SECRET_KEY=<strong-secret>
CORS_ORIGINS=https://www.nqmovee.site,https://nqmovee.site
AI_PUBLIC_URL=https://ai.nqmovee.site
DEVICE=0
```

Then point your frontend environment to the public backend URL:

```text
VITE_API_URL=https://api.nqmovee.site
VITE_WS_URL=wss://api.nqmovee.site
```
