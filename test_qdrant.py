from qdrant_client import QdrantClient



client = QdrantClient(
    url="https://4ae1bf46-a1be-419c-8f7a-751a29d868c2.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.3ufGsS34vNHGU_KvlSrSSv2LjsSLBLJ4q9J_z8BY_P8",
    prefer_grpc=False,
    timeout=10.0,
)

print(client.get_collections())
