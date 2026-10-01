import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

# --- 1. SETUP & TUNABLE PARAMETERS (ARGPARSE) ---
parser = argparse.ArgumentParser(description="Medical AI Chatbot with Custom Personas")
parser.add_argument("--persona", type=str, choices=["clinical", "patient", "pharma"], default="clinical", 
                    help="Choose the bot's behavior: 'clinical' (strict), 'patient' (simple language), or 'pharma' (drugs/medication).")
parser.add_argument("--temp", type=float, default=0.2, help="Set the inference temperature (0.0 to 1.0).")
parser.add_argument("--tokens", type=int, default=400, help="Set the maximum generated tokens.")
args = parser.parse_args()

# --- 2. CUSTOM PERSONAS ---
PERSONAS = {
    "clinical": (
        "You are a strict Clinical Medical Assistant. Provide highly technical, exact medical definitions, "
        "anatomical context, and etiology. Use professional medical terminology. "
        "Strictly no procedural, surgical, or dosage execution steps."
    ),
    "patient": (
        "You are a friendly, empathetic Patient Education Assistant. Explain medical conditions and concepts "
        "using simple, easy-to-understand language. Use analogies when helpful. Reassure the patient, but "
        "strictly refuse to give medical advice, self-treatment steps, or surgical procedures."
    ),
    "pharma": (
        "You are a Pharmacology Reference Assistant. Focus specifically on explaining medications, drug classes, "
        "mechanisms of action, and common side effects. Strictly refuse to prescribe dosages or give administration instructions."
    )
}

selected_system_prompt = PERSONAS[args.persona]

load_dotenv()
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)
console = Console()

# --- 3. SESSION PERSISTENCE (Isolated by Persona) ---
HISTORY_FILE = f"chat_history_{args.persona}.json"
MAX_TURNS = 5 

if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)
        console.print(f"[bold yellow]Loaded previous '{args.persona}' session history.[/bold yellow]")
    except Exception:
        messages = [{"role": "system", "content": selected_system_prompt}]
else:
    messages = [{"role": "system", "content": selected_system_prompt}]

# Welcome UI showing active configurations
console.print(
    Panel.fit(
        f"[bold green]Medical Assistant Initialized[/bold green]\n"
        f"[dim]Persona: {args.persona.upper()} | Temp: {args.temp} | Max Tokens: {args.tokens}[/dim]\n"
        "Type [bold red]'quit'[/bold red] or [bold red]'exit'[/bold red] to end the session.",
        border_style="cyan"
    )
)

# --- 4. MAIN CHAT LOOP ---
while True:
    user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        console.print("[bold yellow]Goodbye![/bold yellow]")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        # Apply the tunable parameters to the API call
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            max_tokens=args.tokens,
            temperature=args.temp,
            stream=True,
        )

        console.print("\n[bold green]Bot:[/bold green]")
        full_reply = ""

        with Live(console=console, refresh_per_second=12) as live:
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_reply += content
                    live.update(Markdown(full_reply))

        messages.append({"role": "assistant", "content": full_reply})

        if len(messages) > 1 + (MAX_TURNS * 2):
            messages = [messages[0]] + messages[-(MAX_TURNS * 2):]

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")