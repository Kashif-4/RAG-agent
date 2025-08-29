import httpx

url = "https://4ae1bf46-a1be-419c-8f7a-751a29d868c2.eu-west-1-0.aws.cloud.qdrant.io/collections"
headers = {
    "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.3ufGsS34vNHGU_KvlSrSSv2LjsSLBLJ4q9J_z8BY_P8"  # Same one that worked in curl
}

try:
    response = httpx.get(url, headers=headers, timeout=10)
    print("Status code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Failed:", e)
