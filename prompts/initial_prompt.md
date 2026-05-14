# Loom

i want to write a python command line tool for llm batch processing.

* allows text processing only  
* the user specifies  
* a JSON file with a specific format, that is basically a list of prompts  
* an llm provider (e.g., google) and model (gemini-3.5-flash-preview)  
* the tool saves the ids to reference the batch, as well as the input file, in a temp directory  
* the tool also has a fetch function to retrieve a batch. when the batch is finished, the tool generates a JSON with all outputs  
* the tool can also process CSV. in this case, the prompts will be defined in a column of the CSV. the output will be the input csv \+ a column "llm\_response"  
* it supports openai, google and anthropic for the start  
* API keys will be read  
  * from .env file  
  * as environment variables  
  * can also be passed as parameters

## **2\. Software Architecture & Flow**

The tool functions as a **State Machine**. Because batch jobs can take up to 24 hours (OpenAI/Anthropic standards), the tool must store the batch\_id locally to "remember" what it is waiting for.

### **A. Input Data Formats**

The tool expects one of two formats:

**JSON Format (input.json)** A simple list of objects containing a unique ID and the prompt.

JSON  
\[  
  {"id": "task-001", "prompt": "Summarize this: ..."},  
  {"id": "task-002", "prompt": "Translate this to French: ..."}  
\]

**CSV Format (input.csv)** The user must specify which column contains the prompt via a CLI flag (e.g., \--col text).

Code-Snippet  
id,text,metadata  
1,"Explain quantum physics",low\_priority  
2,"Write a poem about rust",high\_priority

---

## **3\. Example CLI Calls**

### **Starting a Batch**

When you run this, the tool:

1. Converts your file to the provider's specific format (e.g., .jsonl for OpenAI).  
2. Uploads it and triggers the batch.  
3. Saves a file in \~/.loom/batches/{batch\_id}.json containing metadata.

Bash  
\# Process a JSON file with Gemini  
loom run \--file prompts.json \--provider google \--model gemini-1.5-flash

\# Process a CSV file with Claude  
loom run \--file data.csv \--col "text" \--provider anthropic \--model claude-3-5-sonnet

### **Fetching Results**

Since batches are asynchronous, you check back later. The tool looks into its temp directory, finds the active IDs, and polls the provider.

Bash  
\# Fetch a specific batch  
loom fetch \--id "batch\_67890abc"

\# Fetch all pending batches in the temp directory  
loom fetch \--all

---

## **4\. Output Logic**

* **JSON Input:** Generates prompts\_results.json where each object now includes an "llm\_response" field.  
* **CSV Input:** Generates data\_results.csv. It preserves all original columns (metadata, IDs) and appends the llm\_response column at the end.

## **5\. Technical Implementation Tip**

To support multiple providers, use a **Strategy Pattern**. Create a base Provider class and subclasses for OpenAIProvider, GoogleProvider, and AnthropicProvider. This makes it easy to add new models or custom local LLM backends later.

Python  
\# Conceptual Strategy Pattern  
class BatchProvider:  
    def submit(self, file\_path): pass  
    def check\_status(self, batch\_id): pass  
    def download\_results(self, batch\_id): pass

## **1\. Project Structure Recommendation**

To keep the tool modular, I'd recommend a directory structure that separates the "Orchestrator" from the "Provider" logic.

Plaintext

loom/

├── main.py           \# Entry point (Click or Typer)

├── core/

│   ├── orchestrator.py  \# Handles file IO and state management

│   └── models.py        \# Pydantic models for validation

├── providers/

│   ├── base.py       \# Abstract Base Class

│   ├── openai.py     \# OpenAI Batch API implementation

│   ├── anthropic.py  \# Anthropic Message Batches

│   └── google.py     \# Google Vertex/AI Studio Batch

└── utils/

    ├── storage.py    \# Manages the \~/.loom/ temp directory

    └── converters.py \# Converts CSV/JSON to Provider-specific JSONL

---

## **2\. Managing the "Temp" State**

Since you want to save IDs to a temp directory, avoid /tmp (which clears on reboot). Instead, use the user's home directory.

* **Location:** \~/.loom/batches/  
* **Filename:** {provider}\_{batch\_id}.json  
* **Metadata to store:**  
  * original\_file\_path: So you know where to merge the results back to.  
  * file\_type: (JSON or CSV).  
  * column\_mapping: (If CSV, which column was the prompt).  
  * created\_at: For sorting by newest.

---

## **3\. Handling the "Merge" Logic**

The trickiest part of LLM batching is that **providers do not guarantee order.** OpenAI and Anthropic return results in a .jsonl file where lines may be shuffled.

### **For JSON:**

You should include a custom\_id in your batch request. When fetching, use that ID to map the response back to the original index in your input list.

### **For CSV:**

1. Read the original CSV into a Pandas DataFrame or a list of dicts.  
2. Assign a temporary UUID to each row.  
3. After fetching, join the LLM responses back to the original rows based on that UUID.  
4. Export to {original\_name}\_processed.csv.

---

## **4\. Provider Nuances**

Be aware that "Batch" means different things to different APIs:

| Feature | OpenAI | Anthropic | Google (Gemini) |
| :---- | :---- | :---- | :---- |
| **Endpoint** | /v1/batches | /v1/messages/batches | jobs.batchPredictionJobs |
| **Input Format** | JSONL (Required) | JSONL (Required) | JSONL (Cloud Storage / Local) |
| **Timeout** | 24 Hours | 24 Hours | Varies |
| **Price** | 50% Off | 50% Off | Standard (usually) |

**Direct Correction:** Note that for Google Gemini, batch processing via the Google AI SDK often requires the input file to be hosted on **Google Cloud Storage (GCS)** if the dataset is large, though some newer preview features allow direct uploads. You'll want to check if your tool needs to handle a "GCS Upload" step or if it can pipe it directly.

## **5\. Potential "Wit" for your CLI**

Since you are calling this a "batch" tool, you could add some fun status messages while it waits:

* loom run: "Spinning the threads... Batch submitted to OpenAI."  
* loom fetch: "Checking the loom... Your data is 45% woven."  
* loom fetch \--done: "Fabric complete. Results saved to output.csv."

Find the batch API documentations here

* Google: https://ai.google.dev/gemini-api/docs/batch-api  
* Anthropic: https://platform.claude.com/docs/en/build-with-claude/batch-processing  
* OpenAI: https://developers.openai.com/api/docs/guides/batch