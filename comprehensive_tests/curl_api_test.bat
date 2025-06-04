@echo off
echo Testing Nova AI API with curl...
echo.

echo 1. Testing health endpoint...
curl -X GET "http://localhost:8000/health" -H "Content-Type: application/json" -w "\nStatus: %%{http_code}\n" -s
echo.

echo 2. Testing CORS preflight...
curl -X OPTIONS "http://localhost:8000/query" -H "Origin: http://localhost:5173" -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: content-type" -w "\nStatus: %%{http_code}\n" -s -D -
echo.

echo 3. Testing query endpoint...
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -H "Origin: http://localhost:5173" -d "{\"query\":\"Test query\",\"tts_enabled\":false}" -w "\nStatus: %%{http_code}\n" -s
echo.

echo Test completed.
