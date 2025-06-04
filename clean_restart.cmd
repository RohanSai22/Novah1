@echo off
echo Cleaning up Docker containers and images...

REM Stop all containers
docker-compose down --volumes --remove-orphans

REM Remove all containers, networks, and images created by docker-compose
docker-compose down --rmi all --volumes --remove-orphans

REM Prune system to clean up
docker system prune -f

REM Clean Docker build cache
docker builder prune -f

echo Starting services with fresh build...
docker-compose up --build --force-recreate

if %ERRORLEVEL% neq 0 (
    echo Error: Failed to start containers. Check Docker logs with 'docker compose logs'.
    echo Possible fixes: Ensure Docker Desktop is running or check if port 8080 is free.
    exit /b 1
)
