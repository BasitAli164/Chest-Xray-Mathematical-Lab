"""Generative AI explanation module using Groq API.

Yeh module image processing ke baad non-technical explanation generate karta hai.
Yeh API ko poori image nahi bhejta, balke sirf operation ka metadata aur
statistical differences bhejta hai taake token limit cross na ho aur AI
ko imaginary numbers banane ki zaroorat na pade.
"""

from __future__ import annotations

import os
import ssl
import certifi
import httpx

from typing import Any
import streamlit as st
from groq import Groq

# Default model (Groq par yeh bohat fast aur free hai)
DEFAULT_MODEL = "openai/gpt-oss-120b"

def _get_client() -> Groq | None:
    """Streamlit secrets ya environment variable se Groq client banata hai."""
    api_key = None
    
    # Streamlit secrets se try karo
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    # Environment variable se try karo
    elif "GROQ_API_KEY" in os.environ:
        api_key = os.environ["GROQ_API_KEY"]
        
    if not api_key:
        return None
    
    # ---------------------------------------------------------
    # WINDOWS SSL ERROR FIX
    # Agar system ka SSL certificate kharab ho, toh hum force
    # karenge ke httpx certifi library ka sahi certificate use kare.
    # ---------------------------------------------------------
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        custom_http_client = httpx.Client(verify=ssl_context)
        return Groq(api_key=api_key, http_client=custom_http_client)
    except Exception:
        # Agar kisi wajah se custom client fail ho, toh normal client try karein
        return Groq(api_key=api_key)

def _extract_math_metadata(trace_dict: dict[str, Any]) -> dict[str, Any]:
    """Trace se sirf zaroori numeric data nikalta hai (taake prompt chota rahe)."""
    
    # Level 1: Formula info
    op = trace_dict.get("operation", "Unknown Operation")
    formula = trace_dict.get("formula_plain", "N/A")
    params = trace_dict.get("parameters", {})
    
    # Level 2: Derivation aur Spot checks se numbers nikalne ki koshish
    # Hum dhoondte hain wo steps jahan actual values di gayi hain (jaise values={"A_00": 10, "C_00": 40})
    spot_checks = []
    for step in trace_dict.get("derivation", []):
        if "values" in step and step["values"]:
            vals = step["values"]
            spot_checks.append(f"Step '{step.get('title', '')}': {vals}")
            
    # Agar pixel-level trace hai (Convolution wagera ke liye)
    pixel_trace = None
    for inter in trace_dict.get("intermediate_results", []):
        if isinstance(inter, dict) and "result" in inter and "kernel_values" in inter:
            pixel_trace = {
                "sum": inter.get("sum"),
                "result": inter.get("result"),
                "neighborhood_preview": inter.get("neighborhood", [])[:1], # sirf pehli row
            }
            break

    return {
        "operation_name": op,
        "applied_parameters": params,
        "mathematical_formula": formula,
        "calculation_steps": spot_checks,
        "pixel_level_calculation": pixel_trace
    }

def generate_explanation(trace_dict: dict[str, Any]) -> str:
    """Groq API call karke simple explanation return karta hai."""
    client = _get_client()
    if not client:
        return "⚠️ **API Key Missing:** AI explanation ke liye `.streamlit/secrets.toml` mein `GROQ_API_KEY` set karein."

    # 1. Extract compact data
    meta = _extract_math_metadata(trace_dict)
    
    # 2. Build Prompt
        # 2. Build Prompt
    system_prompt = (
        "You are an expert Computer Vision Professor and a Linear Algebra teacher. "
        "Your task is to explain image processing operations to a first-semester student who has basic math knowledge but no prior experience with digital image processing. "
        "You must strictly use the actual numbers provided in the user prompt. NEVER invent or hallucinate numbers. "
        "If the provided data is missing some values, explain the concept using the available data and the general formula.\n\n"
        
        "Write the explanation in clear, simple, and friendly language. "
        "You MUST format your response using EXACTLY these 6 sections with bold headings. "
        "DO NOT include a separate section for formula breakdown or variable definitions (as the user interface already displays the math formula). Instead, focus on explaining the practical impact.\n\n"
        
        "**1. The Core Concept (What & Why)**: Explain what this operation actually does in real life. Why was it needed? (e.g., Why do we blur? Why do we detect edges?).\n"
        "**2. The Linear Algebra Behind It**: Name the exact mathematical concept (e.g., Convolution, Matrix Transpose, Eigenvalue Decomposition). Explain the concept simply, as if teaching in a classroom.\n"
        "**3. What Changed in the Image**: Compare the image before and after using the provided spot-check numbers. Explain how the actual pixel intensity, contrast, or edges changed based on those specific numbers.\n"
        "**4. Parameters Used**: Explain the parameters used (e.g., Kernel Size, Sigma, Offset). What happens if we increase or decrease this parameter?\n"
        "**5. Purpose of Operation (Practical Use Case)**: Where is this operation used in the real world (e.g., Medical imaging, self-driving cars, photo editing)? Why is it useful?\n"
        "**6. Processing Sequence**: If this operation builds upon a previous state or fits into a larger pipeline, explain how this specific step helps prepare the image for the next steps.\n\n"
        
        "Keep the tone encouraging, academic, and easy to understand. Use bullet points and bold text to make it highly readable. "
        "Do not exceed 200 words. Be direct and avoid generic filler text."
    )
    
    user_prompt = (
        f"Here is the actual data from the image operation performed:\n\n"
        f"- **Operation**: {meta['operation_name']}\n"
        f"- **Parameters Used**: {meta['applied_parameters']}\n"
        f"- **Formula Applied**: {meta['mathematical_formula']}\n"
        f"- **Actual Numbers from Image (Spot Checks)**: {meta['calculation_steps']}\n"
    )
    if meta["pixel_level_calculation"]:
        user_prompt += f"- **Pixel Level Calculation**: {meta['pixel_level_calculation']}\n"
        
    user_prompt += "\nPlease explain this operation to me based on these exact numbers."

    # 3. API Call
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_MODEL,
            temperature=0.3, # Kam temperature = zyada factual, kam creative (taake numbers ghalat na kare)
            max_tokens=1000,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ **AI Error:** API call fail ho gayi. Details: {str(e)}"