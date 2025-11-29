import os
import json
import time
import traceback
import base64
import requests
import uvicorn
import shutil
import mimetypes
import threading
from urllib.parse import urlparse, unquote
from typing import List, Optional, Any, Dict
from urllib.parse import urljoin
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# --- Configuration ---
MY_SECRET = "prasana-ramanaa-tds-p2"

# 1. Main LLM (Generation - Logic & Coding)
LLM_GEN_BASE_URL = "https://aipipe.org/openrouter/v1"
LLM_MODEL_ANALYST = "x-ai/grok-4.1-fast:free" 
LLM_MODEL_DEV = "x-ai/grok-code-fast-1" 
LLM_API_KEY = ""
LLM_MODEL_FALLBACK = "google/gemini-3-pro-preview" # [ADD THIS]

# 2. Media LLM (Vision & Audio Transcription only)
LLM_MEDIA_BASE_URL = "https://aipipe.org/openai/v1"
MEDIA_MODEL_VISION = "gpt-4o" # or compatible vision model
MEDIA_MODEL_AUDIO = "whisper-1"

# Clients
client_gen = OpenAI(base_url=LLM_GEN_BASE_URL, api_key=LLM_API_KEY)
client_media = OpenAI(base_url=LLM_MEDIA_BASE_URL, api_key=LLM_API_KEY)

app = FastAPI()

# --- Data Models ---

# Pass 1: Analysis
class TaskBlueprint(BaseModel):
    submit_url: str
    #submission_payload_keys: List[str] 
    step_by_step_plan: str # Detailed plan including file paths and specific logic
    #data_format_hint: str 

# Pass 2: Coding
class GeneratedCode(BaseModel):
    python_code: str

class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str

# --- Helper: Media Processing ---

def analyze_image(image_path: str) -> str:
    """Gets text description of an image."""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client_media.chat.completions.create(
            model=MEDIA_MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. If there is text, transcribe it exactly. If there are objects to count, count them."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error describing image {image_path}: {str(e)}"

def transcribe_audio(audio_path: str) -> str:
    """Transcribes audio file."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client_media.audio.transcriptions.create(
                model=MEDIA_MODEL_AUDIO, 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio {audio_path}: {str(e)}"

# --- Helper: File Downloader ---

def download_files_from_html(html_content: str, base_url: str, download_dir: str) -> Dict[str, str]:
    """
    Downloads ALL files found in href or src tags.
    Returns a dict: { original_url : local_file_path }
    """
    soup = BeautifulSoup(html_content, "html.parser")
    downloaded_map = {}
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Gather all potential links
    tags = soup.find_all(['a', 'img', 'source', 'link', 'script', 'iframe', 'video', 'audio', 'embed', 'object'])
    
    for tag in tags:
        url = tag.get('href') or tag.get('src')
        if not url: continue
        
        # Absolute URL
        abs_url = urljoin(base_url, url)
        parsed = urlparse(abs_url)
        
        # Filter out anchors, javascript, or blank
        if abs_url.startswith("#") or abs_url.startswith("javascript"):
            continue

        # Get filename
        filename = os.path.basename(parsed.path)
        if not filename: 
            # Try to guess extension if missing
            filename = f"file_{int(time.time()*1000)}"

        # Clean filename
        filename = unquote(filename)
        local_path = os.path.join(download_dir, filename)

        # Download
        try:
            # simple check to avoid downloading the page itself recursively if it acts like a file
            # Ideally we check Content-Type headers, but for speed we accept most things here
            # We skip if it looks like just a standard nav link (no extension), unless specifically requested?
            # User said "ANY FILE", usually implies assets. 
            
            print(f"[*] Attempting download: {abs_url}")
            r = requests.get(abs_url, stream=True, timeout=5)
            if r.status_code == 200:
                # Basic check to ensure we aren't downloading html pages as files unless they have extensions
                content_type = r.headers.get('content-type', '')
                if 'text/html' in content_type and not (filename.endswith('.html') or filename.endswith('.txt')):
                    continue 

                with open(local_path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                
                downloaded_map[abs_url] = local_path
                print(f"    -> Saved to: {local_path}")
        except Exception as e:
            print(f"    -> Download failed: {e}")
            continue

    return downloaded_map

# --- Helper: HTML Sanitizer ---
def clean_and_absolutize_html(html_content: str, base_url: str) -> str:
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup.find_all(href=True):
            tag['href'] = urljoin(base_url, tag['href'])
        for tag in soup.find_all(src=True):
            tag['src'] = urljoin(base_url, tag['src'])
        for tag in soup.find_all(action=True):
            tag['action'] = urljoin(base_url, tag['action'])
        return str(soup)
    except Exception:
        return html_content

# --- Helper: Code Execution Sandbox ---
def execute_generated_code(code: str, context: dict) -> Any:
    local_scope = context.copy()
    # Pre-import common libs
    import pandas as pd
    import numpy as np
    import requests
    import json
    import re
    import base64
    from bs4 import BeautifulSoup
    
    local_scope.update({
        "pd": pd, "np": np, "requests": requests, 
        "json": json, "re": re, "BeautifulSoup": BeautifulSoup,
        "base64": base64, "os": os
    })

    try:
        exec(code, globals(), local_scope)
        if "answer" not in local_scope:
            raise ValueError("Code did not define 'answer' variable.")
        return local_scope["answer"]
    except Exception as e:
        raise e

# --- Core Logic ---
def process_quiz(email: str, secret: str, task_url: str):
    print(f"[*] Processing URL: {task_url}")
    
    # Create a workspace for this run
    workspace_id = str(int(time.time()))
    download_dir = os.path.join(os.getcwd(), "downloads", workspace_id)

    # 1. Scrape HTML
    raw_html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(task_url)
            page.wait_for_load_state("networkidle") 
            time.sleep(2)
            raw_html = page.content()
            browser.close()
    except Exception as e:
        print(f"[!] Scraping failed: {e}")
        return

    # 2. Download ALL Files
    print("[*] Downloading Assets...")
    # We clean HTML first to ensure links are absolute for the downloader
    page_content = clean_and_absolutize_html(raw_html, task_url)
    file_map = download_files_from_html(page_content, task_url, download_dir)

    # 3. Process Media (Transcribe/Describe)
    media_context = []
    print("[*] Processing Media Content...")
    
    for url, local_path in file_map.items():
        mime_type, _ = mimetypes.guess_type(local_path)
        if not mime_type: continue

        info_text = ""
        if mime_type.startswith("image"):
            desc = analyze_image(local_path)
            info_text = f"IMAGE ({local_path}): {desc}"
        elif mime_type.startswith("audio"):
            trans = transcribe_audio(local_path)
            info_text = f"AUDIO ({local_path}) Transcription: {trans}"
        
        if info_text:
            media_context.append(info_text)
            print(f"    -> Analyzed: {os.path.basename(local_path)}")

    # 4. Construct Comprehensive Input for Analyst
    media_context_str = "\n".join(media_context)
    files_list_str = json.dumps(file_map, indent=2)
    
    # [ADD THIS]
    retry_count = 0 
    server_suggested_url = None
    skip_after = 2

    # 1. Get the JSON Schema from your Pydantic model
    schema_def = TaskBlueprint.model_json_schema()

    # --- PASS 1: THE ANALYST (Updated System Prompt) ---
    print("[*] PASS 1: Analyzing Task with Media Context...")
    
    analyst_system_prompt = """
    You are an Elite Data Extraction & Logic Analyst.
    
    Your Goal:
    1. Identify the Question in the provided context.
    2. Identify the 'Submission URL'.
    3. Create a PRECISE STEP-BY-STEP PLAN to get the answer using python.
    
    PLANNING RULES:
    0. **PLAN ONLY TO GET ANSWER**: You are creating the plan to get the final answer using python. DO NOT create a plan to post the json itself. JUST FOCUS ON GETTING THE ANSWER.  
    1. **NO CODE NEEDED?**: If the answer is obvious from the Media Analysis (e.g., "Count red bottles" and the analysis says "7 red bottles") or simple calculations, the plan should simply be: "Write python code to set answer = 7".
    2. **SCRAPING**: If the question can be answered by scraping the HTML content, the plan must include steps to parse the HTML ONLY using PLAYWRIGHT and NOT BEAUTIFULSOUP.
    3. **MEDIA CONTEXT**: If the question can be answered using the media analysis (image descriptions, audio transcriptions), extract the answer yourself and make the plan simply be to set answer = (your found answer).
    4. **PAGE CONTENT**: If the question can be answered using the raw HTML content (e.g., HIDDEN TEXT, text in paragraphs, tables), extract the answer yourself and make the plan simply be to set answer = (your found answer).
    5. **FILES**: If the question relies on a file (Example: CSV, Excel, Zip or any other file), refer to the LOCAL PATH provided in the 'Downloaded Files' section. Do not ask to download it again.
    6. **LLM REQUIRED?**: If and ONLY IF the solution requires AI reasoning (e.g., summarizing text, sentiment analysis), the plan must instruct to use the following specific Ollama endpoint:
       - Model: "gemma3:12b"
       - URL: "http://192.168.1.35:11434/api/generate"
       - Method: Use Python `requests` to POST to this local IP.
    7. **FILE SUBMISSION**: If the answer requires uploading a file or an image, the plan must instruct to read the file from the local path, convert it to a Base64 string, and set that as the `answer`.
    
    RESPONSE FORMAT:
    Output strict JSON complying with the TaskBlueprint schema.
    """


    full_context_input = f"""
    TARGET URL: {task_url}
    
    --- RAW HTML CONTENT (Read and Understand it fully. Try to get answer from it.) ---
    {page_content}
    
    --- DOWNLOADED FILES (URL -> LOCAL PATH) ---
    {files_list_str}
    
    --- MEDIA ANALYSIS (Visual Descriptions / Audio Transcriptions) ---
    {media_context_str}
    """


    planning_history = [
        {"role": "system", "content": analyst_system_prompt},
        {"role": "user", "content": full_context_input}
    ]

    while retry_count <= skip_after:
        try:
            # [CHANGE 3: Model Selection Logic]
            # Use Standard Dev model for attempts 0-3, use Fallback (Gemini 3 Pro) for attempt 4
            current_model = LLM_MODEL_DEV
            if retry_count == skip_after:
                print(f"[*] 4th Retry: Activating Fallback Model {LLM_MODEL_FALLBACK}...")
                current_model = LLM_MODEL_FALLBACK
                
                # Define the specialized "Hail Mary" System Prompt
                new_system_prompt = {
                    "role": "system", 
                    "content": (
                        "You are an Advanced Python Developer and Advanced Problem Solving agent. The Goal is to Generate an error-free and correct code to get a correct 'ANSWER' for the Question in the RAW HTML Content.\n"
                        "Rules:\n"
                        "0. DO NOT give code to post submission json. Generate code to GET ONLY THE ANSWER.\n"
                        "1. Output valid Python code only.\n"
                        "2. Define variable `answer` with the final result.\n"
                        "3. **FILES**: Use the local file paths provided in the input. DO NOT attempt to download files again. They are already on disk.\n"
                        "4. **OLLAMA**: If needed, use `requests` to hit `http://192.168.1.35:11434`.\n"
                        "5. **IMPORTS**: Import whatever you need inside the script (pandas, requests, etc).\n"
                        "Below are the conversation between a User and the Assistant. Analyse it throughly, find what is the issue and give the correct code for the question."
                    )
                }

                # Overwrite indices 0 and 1 (The original setup) 
                # but KEEP the subsequent conversation history (errors/code) for context.
                if len(planning_history) >= 2:
                    planning_history[0] = new_system_prompt

            try:
                if retry_count == skip_after:
                    print(f"[*] Using Fallback Model {current_model} for Analysis...")
                    analysis_completion = client_gen.beta.chat.completions.create(
                        model=current_model,
                        messages=planning_history,
                        response_format={
                            "type": "json_schema", 
                            "json_schema": {
                                "name": "TaskBlueprint", 
                                "schema": schema_def,
                                "strict": True # Attempt to force strictness
                            }
                        },
                        extra_body = {
                            "reasoning": {
                                # One of the following (not both):
                                "effort": "low", # Can be "high", "medium", "low", "minimal" or "none" (OpenAI-style)
                                #"max_tokens": 2000, # Specific token limit (Anthropic-style)
                                # Optional: Default is false. All models support this.
                                "exclude": True, # Set to true to exclude reasoning tokens from response
                                # Or enable reasoning with the default parameters:
                                "enabled": True # Default: inferred from `effort` or `max_tokens`
                            }
                        },
                    )

                else:               
                    analysis_completion = client_gen.beta.chat.completions.create(
                        model=LLM_MODEL_ANALYST,
                        messages=planning_history,
                        response_format={
                            "type": "json_schema", 
                            "json_schema": {
                                "name": "TaskBlueprint", 
                                "schema": schema_def,
                                "strict": True # Attempt to force strictness
                            }
                        },
                        #extra_body = {
                        #    "reasoning": {
                                # One of the following (not both):
                                #"effort": "low", # Can be "high", "medium", "low", "minimal" or "none" (OpenAI-style)
                                #"max_tokens": 2000, # Specific token limit (Anthropic-style)
                                # Optional: Default is false. All models support this.
                                #"exclude": True, # Set to true to exclude reasoning tokens from response
                                # Or enable reasoning with the default parameters:
                        #        "enabled": False # Default: inferred from `effort` or `max_tokens`
                        #    }
                        #},
                    )


                # 3. Get Raw Content
                raw_content = analysis_completion.choices[0].message.content
                
                # 4. THE FIX: Manually strip Markdown before parsing
                # This handles the case where DeepSeek wraps valid JSON in ```json ... ```
                clean_json = raw_content.replace("```json", "").replace("```", "").strip()
                
                # 5. Convert to Pydantic Object
                blueprint = TaskBlueprint.model_validate_json(clean_json)
                
                print(f"[*] Blueprint Created. Target: {blueprint.submit_url}")
                print(f"[*] Plan: {blueprint.step_by_step_plan}")

                planning_history.append({"role": "assistant", "content": blueprint.step_by_step_plan})

            except Exception as e:
                print(f"[!] Analysis Failed: {e}")
                # Cleanup
                shutil.rmtree(download_dir, ignore_errors=True)
                return 

            # --- PASS 2: THE DEVELOPER (Updated Logic) ---
            
            print("[*] PASS 2: Generating Code to Solve Task...")
            coding_message = [
                {"role": "system", "content": (
                    "You are a Python Developer. Generate code based on the Analyst's plan.\n"
                    "Rules:\n"
                    "1. Output valid Python code only.\n"
                    "2. Define variable `answer` with the final result.\n"
                    "3. **FILES**: Use the local file paths provided in the input. DO NOT attempt to download files again. They are already on disk.\n"
                    "4. **OLLAMA**: If instructed, use `requests` to hit `http://192.168.1.35:11434`.\n"
                    "5. **IMPORTS**: Import whatever you need inside the script (pandas, requests, etc)."
                    "6. DO NOT EVER leave the answer empty untill otherwise explicitly instructed in the plan."
                )},
                {"role": "user", "content": f"""
                CONTEXT: (Use only if needed)
                - Downloaded Files Map: {files_list_str}
                - Media Info: {media_context_str}
                
                ANALYST PLAN (Follow this exactly):
                {blueprint.step_by_step_plan}

                DO NOT SET THE ANSWER TO NONE OR EMPTY UNLESS EXPLICITLY INSTRUCTED IN THE PLAN.
                """}
            ]


            # Generate Code
            code_completion = client_gen.beta.chat.completions.parse(
                model=current_model,
                messages=coding_message,
                response_format=GeneratedCode,
            )
            generated_code = code_completion.choices[0].message.parsed.python_code

            print(f"[*] Generated Code (Model: {current_model}):\n{generated_code}")    
        
            planning_history.append({"role": "assistant", "content": generated_code})

            # Execute Code
            try:
                # We pass the download_dir as current_dir just in case
                calculated_answer = execute_generated_code(generated_code, {"current_dir": download_dir})
                print(f"[*] Answer Calculated: {calculated_answer}")
            except Exception as exec_error:
                full_traceback = traceback.format_exc()
                print(f"[!] Execution Error:\n{full_traceback}")

                # [CHANGE 4: Handle Execution Failures as Retries too]
                retry_count += 1
                if retry_count > skip_after:
                    break # Break loop to trigger skip logic below

                planning_history.append(
                    {
                        "role": "user", 
                        "content": f"""
                        \n\n!!! CRITICAL: PREVIOUS PLAN FAILED !!!
                        The previous plan you created led to an error in the code.
                        FAILURE REASON: {full_traceback}
                    
                        INSTRUCTION: Analyze why the previous approach failed. Create a NEW, DIFFERENT plan to solve this.
                        """
                    }
                )
                continue

            # Submit Answer
            payload = {
                "email": email,
                "secret": secret,
                "url": task_url,
                "answer": calculated_answer
            }
            print(f"[*] Submitting Payload: {payload}")
            try:
                response = requests.post(blueprint.submit_url, json=payload, timeout=10)
                res_data = response.json()
                
                # CRITICAL FIX: Capture the URL immediately if it exists
                if res_data.get("url"):
                    server_suggested_url = res_data.get("url")
                    
            except Exception as req_e:
                print(f"[!] Network/JSON Error: {req_e}")
                res_data = {"text": "Network Error"}

            print(f"[*] Result Payload: {res_data}")

            if res_data.get("correct") is True:
                # Success Logic
                next_url = res_data.get("url")
                shutil.rmtree(download_dir, ignore_errors=True)
                if next_url:
                    planning_history = [{"role": "system", "content": analyst_system_prompt},
                                        {"role": "user", "content": full_context_input}] # Reset history for next
                    print(f"[*] Answer Correct! Moving to Next URL: {next_url}")
                    process_quiz(email, secret, next_url)
                else:
                    print("[*] Quiz Complete!")
                return
            else:
                # Failure Logic
                retry_count += 1
                reason = res_data.get("reason", "Unknown")
                
                # Check if we are done
                if retry_count > skip_after:
                    print(f"[!] Max retries reached. Moving to Skip Logic.")
                    break

                # Normal Retry - Loop continues
                print(f"[!] Answer Incorrect. Retry {retry_count}/{skip_after}.")
                msg = f"""
                        \n\n!!! CRITICAL: PREVIOUS PLAN FAILED !!!
                        The Code executed but did not yield the correct answer.
                        Calculated Answer from executed code: {calculated_answer}
                        SERVER REASON: {reason}
                    
                        INSTRUCTION: Analyze why the previous approach failed. Create a NEW, DIFFERENT plan to solve this.
                        """

                # Final Retry
                if retry_count == skip_after:
                    msg += f"""\n
                    THIS IS YOUR LAST ATTEMPT. Use your superior reasoning to fix this.\n
                    Revisit the question, the raw HTML content, the media analysis, and the downloaded files.\n
                    Identify where your previous logic/code went wrong.\n
                    You may change the entire approach if needed.\n
                    Generate the correct code to get the right answer this time.\n
                    Remember, DO NOT give code to post submission json. Just focus on getting the correct 'answer' variable.\n
                    """
                
                planning_history.append({"role": "user", "content": msg})

        except Exception as e:
            print(f"[!] Loop Error: {e}")
            time.sleep(2)

    # [CHANGE 6: SKIP LOGIC (Outside the Loop)]
    # --- SKIP LOGIC (Outside Loop) ---
    print(f"[!] Failed to solve question after {retry_count} attempts. Attempting Skip...")
    
    shutil.rmtree(download_dir, ignore_errors=True)
    
    # Use the persistent variable 'server_suggested_url'
    if server_suggested_url:
        # Prevent infinite loop if server points back to current page
        if server_suggested_url != task_url:
            print(f"[*] Skipping to next URL: {server_suggested_url}")
            planning_history = [
                {"role": "system", "content": analyst_system_prompt},
                {"role": "user", "content": full_context_input}
            ] # Reset history for next
            process_quiz(email, secret, server_suggested_url)
        else:
            print("[!] CRITICAL: Server URL points to current page. Cannot skip to avoid infinite loop.")
    else:
        print("[!] Cannot skip: No valid 'url' found in any server responses.")

# --- FastAPI Setup ---
@app.post("/solve-quiz")
async def handle_task(payload: TaskPayload):
    if payload.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # Run the long-running worker in a daemon thread so Ctrl+C can exit the process.
    t = threading.Thread(target=process_quiz, args=(payload.email, payload.secret, payload.url), daemon=True)
    t.start()
    return JSONResponse(status_code=200, content={"message": "Task started."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
