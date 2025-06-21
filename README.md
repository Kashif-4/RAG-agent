# RAG-agent

✅ What Has Been Done So Far
📄 Data Collection & Preprocessing

Scraped reported judgments from the LHC website.

Downloaded and extracted text from PDFs using BeautifulSoup and PyMuPDF.

📦 Vector Store Setup (Qdrant)

Set up and connected to a Qdrant cloud instance (hosted).

Embedded judgment text using Cohere embeddings (embed-english-light-v3.0).

Uploaded embeddings in batches into Qdrant with appropriate metadata.

🔍 Search Functionality

Implemented a vector similarity search using client.search() (later advised to switch to query_points due to deprecation warning).

Successfully retrieved the top-k matching results based on user input.

🧠 LLM Integration

Integrated Groq API with the model llama3-8b-8192.

Built a function to pass context + query as a prompt to Groq and parse the response.

🤖 Full Query-to-Answer Pipeline

Created a working RAG pipeline:

Take user query → embed → search in Qdrant → format prompt → send to Groq → return answer.

Successfully answered legal queries based on real LHC judgments.



⚠️ Challenges Faced & How They Were Solved
1. Cohere SDK Errors and Exception Handling
One of the first issues was figuring out how to properly handle API rate-limiting from Cohere. The original plan was to catch a specific error like TooManyRequestsError, but this caused an import error because that class doesn’t exist in the Cohere SDK. Trying another option, CohereAPIError, also didn’t work and resulted in another import error.

How it was fixed:
The solution was to handle general exceptions using Python's Exception class, then check whether the error had a status_code of 429 (which indicates too many requests). This allowed retrying the request after a short delay.


2. Rate Limiting from Cohere API
Once the correct error handling was in place, the Cohere API still sometimes returned HTTP 429 errors due to too many requests being sent too quickly.

How it was fixed:
A retry loop was implemented. If a rate limit error occurred, the script would wait a few seconds and then try again—up to 10 times. This ensured the embedding process could continue even under temporary limits.

3. Qdrant vectors_count Showing None
After uploading all the embeddings to Qdrant, checking the number of stored vectors using collection_info.vectors_count returned None, which was confusing.

How it was fixed:
It turned out that the right property to check was points_count, not vectors_count.


4. Deleting & Recreating the Qdrant Collection Didn’t Seem to Work
Even after calling client.delete_collection(...), old data still seemed to persist. This was confusing, especially when verifying if the new upload worked.

How it was fixed:
The full process of deleting and then recreating the collection was placed clearly at the beginning of the code. Then, the rest of the workflow (chunking, embedding, uploading) was re-run from scratch, ensuring everything was reset cleanly.

try:
    client.delete_collection(collection_name)
except:
    pass  # It might not exist yet

client.recreate_collection(...)


5. Not Knowing Whether Qdrant Was Running Locally or on the Cloud
At one point, it wasn’t clear whether the Qdrant client was connecting to a local instance or the cloud. This caused uncertainty about where the data was actually going.

How it was fixed:
Using the QdrantClient constructor with the correct api_key and ensuring it was pointing to the Qdrant Cloud resolved the confusion:


6. Trying to Check Qdrant Version (Unnecessary Method)
An attempt was made to check the Qdrant version using client.get_version(), but this failed with an AttributeError because that method isn’t available in the installed version of the client.

How it was fixed:
This was deemed non-critical and skipped, since the Qdrant instance was functioning correctly.
