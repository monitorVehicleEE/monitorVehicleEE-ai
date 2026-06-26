# Deploy VPS Scenarios

Frontend is already deployed at:

```text
https://www.nqmovee.site
```

The backend and AI stack should expose:

```text
https://api.nqmovee.site -> backend API
https://ai.nqmovee.site  -> AI server, event images, streams
```

In Cloudflare DNS, create:

```text
A  api  -> <VPS_PUBLIC_IP>
A  ai   -> <VPS_PUBLIC_IP>
```

After backend is public, update frontend environment and redeploy:

```text
VITE_API_URL=https://api.nqmovee.site
VITE_WS_URL=wss://api.nqmovee.site
```

## Scenario 1: Ubuntu VPS, Recommended

Use this when the provider has Ubuntu 22.04 or 24.04. This path matches the local Docker workflow.

### 1. SSH into the server

```bash
ssh root@<VPS_PUBLIC_IP>
```

### 2. Install Docker and Git

```bash
apt update
apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sh

docker --version
docker compose version
```

### 3. Check GPU

```bash
nvidia-smi
```

If Docker cannot use the GPU yet, install NVIDIA Container Toolkit:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update
apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

Test Docker GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 4. Clone source code

```bash
mkdir -p /opt/nqmovee
cd /opt/nqmovee

git clone <AI_REPO_URL> monitorVehicleEE-ai
git clone <BE_REPO_URL> monitorVehicleEE-be-fta
```

Expected structure:

```text
/opt/nqmovee/
  monitorVehicleEE-ai/
  monitorVehicleEE-be-fta/
```

### 5. Upload AI models if they are not in Git

Check on VPS:

```bash
ls /opt/nqmovee/monitorVehicleEE-ai/model/pytorch/vehicle/best.pt
ls /opt/nqmovee/monitorVehicleEE-ai/model/pytorch/plate/best.pt
ls /opt/nqmovee/monitorVehicleEE-ai/model/pytorch/char/best.pt
```

If missing, upload from your dev machine:

```powershell
scp -r D:\IT\DATN\workspace\monitorVehicleEE-ai\model root@<VPS_PUBLIC_IP>:/opt/nqmovee/monitorVehicleEE-ai/
```

### 6. Create production environment file

```bash
cd /opt/nqmovee/monitorVehicleEE-ai
cp .env.stack.example .env
nano .env
```

Use:

```env
POSTGRES_DB=movee
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<strong_database_password>
JWT_SECRET_KEY=<strong_jwt_secret>

CORS_ORIGINS=https://www.nqmovee.site,https://nqmovee.site
AI_PUBLIC_URL=https://ai.nqmovee.site

DEVICE=0
STREAM_WIDTH=640
JPEG_QUALITY=75
```

For CPU-only testing, use:

```env
DEVICE=cpu
```

### 7. Start the stack

GPU:

```bash
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env up -d --build
```

CPU:

```bash
docker compose -f docker-compose.stack.yml --env-file .env up -d --build
```

### 8. Test services locally on VPS

```bash
curl http://localhost:8000
curl http://localhost:8001
docker compose -f docker-compose.stack.yml --env-file .env logs -f
```

Expected:

```text
Backend: {"message":"Backend API running"}
AI:      {"message":"AI server running"}
```

### 9. Install Caddy for HTTPS reverse proxy

```bash
apt install -y debian-keyring debian-archive-keyring apt-transport-https curl

curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
  gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg

curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
  tee /etc/apt/sources.list.d/caddy-stable.list

apt update
apt install -y caddy
```

Edit Caddyfile:

```bash
nano /etc/caddy/Caddyfile
```

Use:

```caddyfile
api.nqmovee.site {
    reverse_proxy 127.0.0.1:8000
}

ai.nqmovee.site {
    reverse_proxy 127.0.0.1:8001
}
```

Reload:

```bash
systemctl reload caddy
```

Test:

```bash
curl https://api.nqmovee.site
curl https://ai.nqmovee.site
```

### 10. Update frontend

In Vercel or Cloudflare Pages, set:

```text
VITE_API_URL=https://api.nqmovee.site
VITE_WS_URL=wss://api.nqmovee.site
```

Redeploy frontend.

### 11. Common maintenance commands

```bash
cd /opt/nqmovee/monitorVehicleEE-ai

docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env ps
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env logs -f
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env restart
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env down
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env up -d --build
```

Reset database:

```bash
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env down -v
docker compose -f docker-compose.stack.yml -f docker-compose.gpu.yml --env-file .env up -d --build
```

## Scenario 2: Windows GPU VM

Use this only if Ubuntu is unavailable. The most reliable Windows path is running Python services directly, not Docker GPU.

### 1. Remote into the server

Use Remote Desktop:

```text
mstsc -> <VPS_PUBLIC_IP>
```

### 2. Install required software

Install:

```text
NVIDIA driver
Python 3.11
Node.js 22.12+ only if you need local frontend build
Git
PostgreSQL 16
Caddy for Windows
NSSM
```

Open PowerShell as Administrator.

Install common tools with `winget`:

```powershell
winget install -e --id Git.Git
winget install -e --id Python.Python.3.11
winget install -e --id PostgreSQL.PostgreSQL.16
winget install -e --id CaddyServer.Caddy
winget install -e --id NSSM.NSSM
```

If `winget` asks for agreement, accept it. If a package ID is not found, search it:

```powershell
winget search python
winget search git
winget search postgresql
winget search caddy
winget search nssm
```

Close and reopen PowerShell, then check:

```powershell
python --version
git --version
psql --version
caddy version
nssm version
```

If `psql` is not found, add PostgreSQL to PATH for the current terminal:

```powershell
$env:Path += ";C:\Program Files\PostgreSQL\16\bin"
```

To persist PostgreSQL PATH system-wide:

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\Program Files\PostgreSQL\16\bin",
  "Machine"
)
```

NVIDIA driver is usually preinstalled on GPU VMs. Check first:

```powershell
nvidia-smi
```

If `nvidia-smi` is not found or fails, install the NVIDIA driver from the EzyCloudX control panel or the official NVIDIA driver page, reboot the VPS, then check again.

Check GPU:

```powershell
nvidia-smi
```

### 3. Clone source code

```powershell
mkdir D:\deploy
cd D:\deploy

git clone <AI_REPO_URL> monitorVehicleEE-ai
git clone <BE_REPO_URL> monitorVehicleEE-be-fta
```

Expected structure:

```text
D:\deploy\
  monitorVehicleEE-ai\
  monitorVehicleEE-be-fta\
```

### 4. Prepare PostgreSQL

Create database:

```powershell
psql -U postgres
```

Inside psql:

```sql
CREATE DATABASE movee;
\q
```

Import schema:

```powershell
psql -U postgres -d movee -f D:\deploy\monitorVehicleEE-be-fta\docs\db\create\query.sql
```

### 5. Run backend directly

```powershell
cd D:\deploy\monitorVehicleEE-be-fta
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set environment:

```powershell
$env:DATABASE_URL="postgresql://postgres:<postgres_password>@localhost:5432/movee"
$env:CORS_ORIGINS="https://www.nqmovee.site,https://nqmovee.site"
$env:JWT_SECRET_KEY="<strong_jwt_secret>"
```

Run:

```powershell
uvicorn src.app.main:main --host 127.0.0.1 --port 8000
```

Test:

```powershell
curl http://127.0.0.1:8000
```

### 6. Run AI directly

```powershell
cd D:\deploy\monitorVehicleEE-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Make sure models exist:

```powershell
dir D:\deploy\monitorVehicleEE-ai\model\pytorch\vehicle\best.pt
dir D:\deploy\monitorVehicleEE-ai\model\pytorch\plate\best.pt
dir D:\deploy\monitorVehicleEE-ai\model\pytorch\char\best.pt
```

Set environment:

```powershell
$env:DEVICE="0"
$env:CAMERA_API_URL="http://127.0.0.1:8000"
$env:AI_PUBLIC_URL="https://ai.nqmovee.site"
$env:VIDEO_BASE_DIR="D:\deploy\monitorVehicleEE-ai\dataset\vehicle\videos"
$env:OUTPUT_DIR="D:\deploy\monitorVehicleEE-ai\dataset\output_test"
$env:EVENT_IMAGES_DIR="D:\deploy\monitorVehicleEE-ai\dataset\output_test\events"
$env:SHOW_WINDOW="false"
```

Run sequential AI server:

```powershell
uvicorn src.app.app:app --host 127.0.0.1 --port 8001
```

Test:

```powershell
curl http://127.0.0.1:8001
```

### 7. Install Caddy on Windows

Download Caddy Windows binary from:

```text
https://caddyserver.com/download
```

Place it at:

```text
C:\caddy\caddy.exe
```

Create:

```text
C:\caddy\Caddyfile
```

Use:

```caddyfile
api.nqmovee.site {
    reverse_proxy 127.0.0.1:8000
}

ai.nqmovee.site {
    reverse_proxy 127.0.0.1:8001
}
```

Test:

```powershell
cd C:\caddy
.\caddy.exe run --config Caddyfile
```

When OK, install as service:

```powershell
.\caddy.exe service install --config C:\caddy\Caddyfile
.\caddy.exe service start
```

### 8. Keep backend and AI running on Windows

Use NSSM or Windows Task Scheduler. NSSM is simpler.

Create two `.bat` files.

Backend:

```bat
@echo off
cd /d D:\deploy\monitorVehicleEE-be-fta
call .venv\Scripts\activate.bat
set DATABASE_URL=postgresql://postgres:<postgres_password>@localhost:5432/movee
set CORS_ORIGINS=https://www.nqmovee.site,https://nqmovee.site
set JWT_SECRET_KEY=<strong_jwt_secret>
uvicorn src.app.main:main --host 127.0.0.1 --port 8000
```

AI:

```bat
@echo off
cd /d D:\deploy\monitorVehicleEE-ai
call .venv\Scripts\activate.bat
set DEVICE=0
set CAMERA_API_URL=http://127.0.0.1:8000
set AI_PUBLIC_URL=https://ai.nqmovee.site
set VIDEO_BASE_DIR=D:\deploy\monitorVehicleEE-ai\dataset\vehicle\videos
set OUTPUT_DIR=D:\deploy\monitorVehicleEE-ai\dataset\output_test
set EVENT_IMAGES_DIR=D:\deploy\monitorVehicleEE-ai\dataset\output_test\events
set SHOW_WINDOW=false
uvicorn src.app.app:app --host 127.0.0.1 --port 8001
```

Install with NSSM:

```powershell
nssm install nqmovee-be D:\deploy\run-be.bat
nssm install nqmovee-ai D:\deploy\run-ai.bat
nssm start nqmovee-be
nssm start nqmovee-ai
```

### 9. Update frontend

In Vercel or Cloudflare Pages, set:

```text
VITE_API_URL=https://api.nqmovee.site
VITE_WS_URL=wss://api.nqmovee.site
```

Redeploy frontend.

### 10. Windows warning

Windows can run this project, but it is not the preferred Docker deployment path. Use Ubuntu whenever possible. Windows Docker GPU may require WSL2, NVIDIA CUDA WSL support, Docker Desktop, and provider support for virtualization.

## Quick Recommendation

Prefer:

```text
Ubuntu GPU VM + Docker Compose
```

Fallback:

```text
Windows GPU VM + direct Python services + PostgreSQL + Caddy
```

