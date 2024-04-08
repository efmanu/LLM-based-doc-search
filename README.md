# Document Search using LLM

## Install llama-index 
```commandline
 pip install llama-index 
```
## Ollama
To run LLMs locally

### Pull Docker Image
```commandline
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
### Run LLMs
```commandline
docker exec -it ollama ollama run llama2
```
Replace `llama2` with required model name

## Install Ollama Client
```commandline
pip install ollama
```






