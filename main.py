import os
import openai
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import re
import keyboard
import ast
import random
import time
import heapq
import logging
import pickle

# Load environment variables
load_dotenv()

# Constants for API keys and search limits
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
TAVILY_SEARCH_LIMIT = 100

# Counter for Tavily searches
tavily_search_count = 0

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Add error checking for other API keys
if not TAVILY_API_KEY:
    raise ValueError("No Tavily API key found. Please set the TAVILY_API_KEY environment variable.")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Google Search API key or Custom Search Engine ID not found. Please set the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")

console = Console()

# Set up logging
logging.basicConfig(filename='gpt4_mini_engineer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_json(s: str) -> Optional[Dict]:
    """Extract JSON-like content from a string."""
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(match.group())
            except:
                return None
    return None

class TaskQueue:
    def __init__(self):
        self.tasks = []
        self.completed_tasks = []

    def add_task(self, task: str, priority: int = 0) -> None:
        heapq.heappush(self.tasks, (priority, task))

    def get_next_task(self) -> Optional[str]:
        if self.tasks:
            return heapq.heappop(self.tasks)[1]
        return None

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def view_tasks(self) -> None:
        if not self.tasks:
            console.print("[bold yellow]No tasks in the queue.[/bold yellow]")
            return
        console.print("[bold cyan]Current Tasks:[/bold cyan]")
        for index, (priority, task) in enumerate(sorted(self.tasks)):
            console.print(f"{index + 1}. [Priority {priority}] {task}")

    def add_new_task(self) -> None:
        task = console.input("[bold yellow]Enter new task: [/bold yellow]")
        priority = int(console.input("[bold yellow]Enter priority (0=highest): [/bold yellow]"))
        self.add_task(task, priority)
        console.print(f"[bold green]Task added: {task} with priority {priority}[/bold green]")

    def modify_task(self) -> None:
        self.view_tasks()
        task_index = int(console.input("[bold yellow]Enter the task index to modify: [/bold yellow]")) - 1
        if task_index < 0 or task_index >= len(self.tasks):
            console.print("[bold red]Invalid task index.[/bold red]")
            return
        new_task = console.input("[bold yellow]Enter new task description: [/bold yellow]")
        new_priority = int(console.input("[bold yellow]Enter new priority (0=highest): [/bold yellow]"))
        self.tasks[task_index] = (new_priority, new_task)
        heapq.heapify(self.tasks)
        console.print(f"[bold green]Task modified: {new_task} with priority {new_priority}[/bold green]")

    def manage_tasks(self) -> None:
        while True:
            action = console.input("[bold yellow]Task Management: (v)iew, (a)dd, (m)odify, or (q)uit: [/bold yellow]")
            if action.lower() == 'v':
                self.view_tasks()
            elif action.lower() == 'a':
                self.add_new_task()
            elif action.lower() == 'm':
                self.modify_task()
            elif action.lower() == 'q':
                break

class ProjectContext:
    def __init__(self):
        self.current_focus = None
        self.constraints = []
        self.completed_milestones = []

    def update(self, focus: Optional[str] = None, constraint: Optional[str] = None, milestone: Optional[str] = None) -> None:
        if focus:
            self.current_focus = focus
        if constraint:
            self.constraints.append(constraint)
        if milestone:
            self.completed_milestones.append(milestone)

class Assistant:
    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": """You are specialized in software development with access to a variety of tools and the ability to instruct and direct a coding agent. Your capabilities include:
            1. Creating and managing project structures
            2. Writing, debugging, and improving code across multiple languages
            3. Providing architectural insights and applying design patterns
            4. Staying current with the latest technologies and best practices
            5. Analyzing and manipulating files within the project directory
            6. Performing web searches for up-to-date information

            When assisting users:
            1. Take initiative to solve problems and implement solutions promptly.
            2. Use available tools proactively to create, edit, or manage code and project structures.
            3. Always use the create_folder tool to create necessary directories before creating files.
            4. Always use the create_file tool to create and populate code files instead of just providing code snippets.
            5. Use the edit_and_apply tool to make changes to existing files when necessary.
            6. Provide step-by-step progress updates as you work on solutions.
            7. Only ask for clarification if it's absolutely necessary to proceed.
            8. If multiple approaches are possible, choose the most suitable one based on common best practices and efficiency.
            9. Strive to deliver working solutions, even if simplified, that can be expanded upon later.

            Available tools and their optimal use cases:
            1. create_folder: Create new directories in the project structure.
            2. create_file: Generate new files with specified content. Make the file as complete and useful as possible.
            3. edit_and_apply: Examine and modify existing files by instructing a separate AI coding agent.
            4. read_file: Read the contents of an existing file.
            5. read_multiple_files: Read the contents of multiple existing files at once.
            6. list_files: List all files and directories in a specified folder.
            7. websearch: Google_search and tavily_search available to Perform a web search using the API for up-to-date information.

            Remember, your primary goal is to help users accomplish their tasks effectively and efficiently. Always aim to provide practical, implementable solutions by creating actual project structures and files."""}
        ]
        self.total_tokens_used = 0
        self.project_context = ProjectContext()
        self.cache = {}
        self.interaction_memory = []
        self.file_contents = {}
        self.code_editor_memory = []
        self.code_editor_files = set()
        self.running_processes = {}
        self.task_queue = TaskQueue()

    def process_message(self, user_input: str) -> str:
        """
        Process a user message and return the assistant's response.

        Args:
            user_input (str): The user's input message.

        Returns:
            str: The assistant's response.
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        max_retries = 5
        retry_delay = 1  # Start with a 1 second delay

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.conversation_history,
                    tools=[
                        {"type": "function", "function": {"name": "create_folder", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
                        {"type": "function", "function": {"name": "create_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
                        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
                        {"type": "function", "function": {"name": "edit_and_apply", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "instructions": {"type": "string"}, "project_context": {"type": "string"}}, "required": ["path", "instructions", "project_context"]}}},
                        {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
                    ]
                )
                assistant_message = response.choices[0].message

                # Update the token usage counter
                self.total_tokens_used += response.usage.total_tokens

                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = extract_json(tool_call.function.arguments)

                        if arguments is None:
                            print(f"Failed to parse arguments for function {function_name}. Arguments: {tool_call.function.arguments}")
                            continue

                        if function_name == 'create_folder':
                            output = self.create_folder(arguments.get('path', ''))
                        elif function_name == 'create_file':
                            output = self.create_file(arguments.get('path', ''), arguments.get('content', ''))
                        elif function_name == 'read_file':
                            output = self.read_file(arguments.get('path', ''))
                        elif function_name == 'edit_and_apply':
                            output = self.edit_and_apply(arguments.get('path', ''), arguments.get('instructions', ''), arguments.get('project_context', ''))
                        elif function_name == 'list_files':
                            output = self.list_files(arguments.get('path', ''))
                        else:
                            output = f"Unknown function: {function_name}"

                        self.conversation_history.append({"role": "function", "name": function_name, "content": output})

                    # Get a new response from the assistant with the function output
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=self.conversation_history
                    )
                    assistant_message = response.choices[0].message

                    # Update the token usage counter
                    self.total_tokens_used += response.usage.total_tokens

                # Check for loops by comparing with cache
                if self.check_for_loops(assistant_message.content):
                    print("Detected a loop in responses. Adjusting the request to avoid repetition.")
                    user_input = self.adjust_request(user_input)
                    self.conversation_history.append({"role": "user", "content": user_input})
                    continue

                # Update cache and interaction memory
                self.update_cache(user_input, assistant_message.content)
                self.interaction_memory.append(assistant_message.content)

                self.conversation_history.append({"role": "assistant", "content": assistant_message.content})

                # Log the response
                logging.info(f"Response: {assistant_message.content}")
                
                return assistant_message.content

            except openai.error.OpenAIError as e:
                print(f"Error occurred: {e}")
                if isinstance(e, openai.error.RateLimitError):
                    console.print("[bold red]Rate limit exceeded. Waiting before retrying...[/bold red]")
                    time.sleep(20)  # Wait for 20 seconds before retrying
                    continue
                elif isinstance(e, openai.error.APIError):
                    console.print(f"[bold red]OpenAI API error: {e}[/bold red]")
                    return f"Error: {e}"
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Please try again later.")
                    return f"Error: {e}"

    def tool_checker_agent(self, task: str) -> str:
        response = client.chat.completions.create(
            model="toolchecker-model",
            messages=[{"role": "system", "content": "You are a tool checker agent."}, {"role": "user", "content": task}]
        )
        return response.choices[0].message.content

    def code_editor_agent(self, task: str) -> str:
        response = client.chat.completions.create(
            model="codeeditor-model",
            messages=[{"role": "system", "content": "You are a code editor agent."}, {"role": "user", "content": task}]
        )
        return response.choices[0].message.content

    def code_execution_agent(self, task: str) -> str:
        response = client.chat.completions.create(
            model="codeexecution-model",
            messages=[{"role": "system", "content": "You are a code execution agent."}, {"role": "user", "content": task}]
        )
        return response.choices[0].message.content

    def update_project_state(self, state_update: Dict[str, Optional[str]]) -> None:
        self.project_context.update(**state_update)

    def check_for_loops(self, new_response: str) -> bool:
        return new_response in self.cache.values()

    def adjust_request(self, user_input: str) -> str:
        return f"{user_input} (avoiding repetition)"

    def update_cache(self, user_input: str, response: str) -> None:
        self.cache[user_input] = response

    def reset_conversation(self) -> None:
        self.conversation_history = []
        self.file_contents = {}
        self.code_editor_memory = []
        self.code_editor_files = set()
        self.running_processes = {}
        self.task_queue = TaskQueue()
        self.project_context = ProjectContext()

    def reset_code_editor_memory(self) -> None:
        self.code_editor_memory = []

    def create_folder(self, folder_name: str) -> str:
        try:
            os.makedirs(folder_name, exist_ok=True)
            return f"Folder '{folder_name}' created successfully."
        except Exception as e:
            return f"Error creating folder '{folder_name}': {str(e)}"

    def create_file(self, file_name: str, content: str = "") -> str:
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(content)
            return f"File '{file_name}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_name}': {str(e)}"

    def read_file(self, file_name: str) -> str:
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"File '{file_name}' not found."
        except Exception as e:
            return f"Error reading file '{file_name}': {str(e)}"

    def edit_and_apply(self, path: str, instructions: str, project_context: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()

            prompt = f"""
            Given the following file content and instructions, update the file content accordingly. 
            Ensure the changes are contextually relevant to the project context.

            Project Context:
            {project_context}

            Current File Content:
            {content}

            Instructions:
            {instructions}

            Updated File Content:
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a code editing assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                n=1,
                temperature=0.7,
            )

            updated_content = response.choices[0].message.content.strip()

            with open(path, 'w', encoding='utf-8') as file:
                file.write(updated_content)

            return f"File '{path}' has been updated with project-specific professor input functionality."
        except Exception as e:
            return f"Error in edit_and_apply: {str(e)}"

    def list_files(self, path: str) -> str:
        try:
            files = os.listdir(path)
            return "\n".join(files)
        except FileNotFoundError:
            return f"Path '{path}' not found."
        except Exception as e:
            return f"Error listing files in '{path}': {str(e)}"

    def extract_tasks_from_response(self, response: str) -> List[str]:
        """Extract tasks from the assistant's response."""
        task_pattern = r'(?:^|\. )([A-Z][^.!?]*(?:implement|create|develop|design|fix|update|improve)[^.!?]*\.)'
        tasks = re.findall(task_pattern, response)
        return tasks

    def prioritize_tasks(self, tasks: List[str]) -> List[str]:
        """Prioritize tasks based on current project context."""
        prioritized_tasks = []
        for task in tasks:
            priority = 0
            # Increase priority for tasks related to current focus
            if self.project_context.current_focus and self.project_context.current_focus in task.lower():
                priority += 1
            # Increase priority for tasks that seem more urgent
            if any(word in task.lower() for word in ['urgent', 'important', 'critical']):
                priority += 2
            prioritized_tasks.append((priority, task))
        return [task for _, task in sorted(prioritized_tasks, reverse=True)]  # Higher priority first

    def extract_focus(self, response: str) -> Optional[str]:
        """Extract the current focus from the assistant's response."""
        focus_pattern = r'(?:focus|priority|main task) (?:is|should be) ([\w\s]+)'
        match = re.search(focus_pattern, response, re.IGNORECASE)
        return match.group(1) if match else None

    def generate_next_tasks(self, assistant_response: str) -> List[str]:
        """Generate next tasks based on the assistant's response."""
        tasks = self.extract_tasks_from_response(assistant_response)
        prioritized_tasks = self.prioritize_tasks(tasks)
        return prioritized_tasks

    def save_state(self, filename: str) -> None:
        """Save the current state of the assistant to a file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'conversation_history': self.conversation_history,
                    'project_context': self.project_context,
                    'task_queue': self.task_queue,
                    'total_tokens_used': self.total_tokens_used
                }, f)
            logging.info(f"State saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving state to {filename}: {str(e)}")
            raise

    def load_state(self, filename: str) -> None:
        """Load the state of the assistant from a file."""
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.conversation_history = state['conversation_history']
                self.project_context = state['project_context']
                self.task_queue = state['task_queue']
                self.total_tokens_used = state['total_tokens_used']
            logging.info(f"State loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading state from {filename}: {str(e)}")
            raise

def web_search(query: str) -> str:
    global tavily_search_count
    if tavily_search_count >= TAVILY_SEARCH_LIMIT:
        return google_search(query)
    
    tavily_search_count += 1
    url = "https://api.tavily.com/search"
    params = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_images": False,
        "max_results": 5
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
    else:
        print(f"Tavily search failed with status code: {response.status_code}")
        results = google_search(query)
    
    return "\n".join([f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n" for r in results])

def google_search(query: str) -> List[Dict[str, str]]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": 5
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return [{"title": item['title'], "url": item['link'], "snippet": item['snippet']} for item in data.get('items', [])]
    else:
        print(f"Google search failed with status code: {response.status_code}")
        return []

def file_search(query: str) -> str:
    results = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append(f"Found in {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return "\n".join(results) if results else "No results found."

def display_help() -> None:
    help_text = """
    Available commands:
    - exit: Quit the program
    - manage: Manage tasks (view, add, modify)
    - automode: Enter automode with optional iteration count
    - save: Save current project state
    - load: Load a saved project state
    - help: Display this help message
    """
    console.print(Panel(help_text, title="Help", style="bold green"))

def main() -> None:
    assistant = Assistant()
    console.print(Panel("Welcome to GPT-4 Mini Engineer!", style="bold green"))

    # Get user input for project request
    project_request = console.input("[bold yellow]Enter your project request: [/bold yellow]")
    
    # Get user input for number of iterations (optional)
    iterations_input = console.input("[bold yellow]Enter the number of interactions (default is 15): [/bold yellow]")
    max_iterations = int(iterations_input) if iterations_input.isdigit() else 15

    console.print(Panel(f"Project Request: {project_request}", style="bold yellow"))

    def pause_check() -> bool:
        if keyboard.is_pressed("space"):
            console.print(Panel("[bold red]Development paused. Please provide new input or type 'exit' to quit.[/bold red]"))
            while True:
                user_input = console.input("[bold yellow]Enter next task or type 'exit' to quit: [/bold yellow]")
                if user_input.lower() == "exit":
                    console.print(Panel("Exiting GPT-4 Mini Engineer. Goodbye!", style="bold red"))
                    exit()
                else:
                    assistant_response = assistant.process_message(user_input)
                    console.print(Panel(assistant_response, title="GPT-4 Engineer", style="bold blue"))
                    console.print(Panel(f"Total Tokens Used: {assistant.total_tokens_used}", style="bold magenta"))
                    return True
        return False

    def automode(max_iterations: int = 25) -> None:
        console.print("[bold yellow]Entering automode...[/bold yellow]")
        goal = console.input("[bold yellow]Enter the automode goal: [/bold yellow]")
        assistant.task_queue.add_task(goal)
        
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Running automode...", total=max_iterations)
            for i in range(max_iterations):
                console.print(f"\n[bold cyan]Automode Iteration {i+1}:[/bold cyan]")
                
                if pause_check():
                    return

                if assistant.task_queue.is_empty():
                    console.print(Panel("All tasks completed.", style="bold green"))
                    return
                
                current_task = assistant.task_queue.get_next_task()
                assistant_response = assistant.process_message(current_task)
                console.print(Panel(assistant_response, title="GPT-4 Engineer", style="bold blue"))
                console.print(Panel(f"Total Tokens Used: {assistant.total_tokens_used}", style="bold magenta"))
                
                if "AUTOMODE_COMPLETE" in assistant_response:
                    console.print(Panel("Automode completed.", style="bold green"))
                    return

                assistant.task_queue.completed_tasks.append(current_task)

                # Generate specific, actionable tasks based on the response
                new_tasks = assistant.generate_next_tasks(assistant_response)
                for task in new_tasks:
                    assistant.task_queue.add_task(task)
                
                console.print(Panel(f"New Tasks: {new_tasks}", style="bold yellow"))

                progress.update(task_id, advance=1)
                if keyboard.is_pressed('q'):  # Allow interruption
                    if console.input("[bold yellow]Interrupt automode? (y/n): [/bold yellow]").lower() == 'y':
                        break

        console.print(Panel("Maximum iterations reached. Automode concluded.", style="bold green"))

    for i in range(max_iterations):
        console.print(f"\n[bold cyan]Iteration {i+1}:[/bold cyan]")
        
        if pause_check():
            project_request = f"{project_request} (next step)"
            console.print(Panel(f"New Task: {project_request}", style="bold yellow"))
            continue
        
        assistant_response = assistant.process_message(f"Iteration {i+1}/{max_iterations}: {project_request}")
        console.print(Panel(assistant_response, title="GPT-4 Engineer", style="bold blue"))
        console.print(Panel(f"Total Tokens Used: {assistant.total_tokens_used}", style="bold magenta"))
        
        if "AUTOMODE_COMPLETE" in assistant_response:
            console.print(Panel("Project development completed.", style="bold green"))
            break

        new_tasks = assistant.generate_next_tasks(assistant_response)
        for task in new_tasks:
            assistant.task_queue.add_task(task)

        console.print(Panel(f"New Tasks: {new_tasks}", style="bold yellow"))
        
    else:
        console.print(Panel("Maximum iterations reached. Project development concluded.", style="bold green"))

    while True:
        user_input = console.input("[bold yellow]Enter next task, 'manage' for task management, 'save' to save state, 'load' to load state, 'help' for commands, or 'exit' to quit: [/bold yellow]")
        if user_input.lower() == "exit":
            console.print(Panel("Exiting GPT-4 Mini Engineer. Goodbye!", style="bold red"))
            break
        elif user_input.lower() == "manage":
            assistant.task_queue.manage_tasks()
        elif user_input.lower() == "save":
            filename = console.input("[bold yellow]Enter filename to save state: [/bold yellow]")
            assistant.save_state(filename)
        elif user_input.lower() == "load":
            filename = console.input("[bold yellow]Enter filename to load state: [/bold yellow]")
            assistant.load_state(filename)
        elif user_input.lower() == "help":
            display_help()
        elif user_input.startswith("automode"):
            iterations = user_input.split()[1] if len(user_input.split()) > 1 else 25
            automode(int(iterations))
        else:
            assistant_response = assistant.process_message(user_input)
            console.print(Panel(assistant_response, title="GPT-4 Engineer", style="bold blue"))
            console.print(Panel(f"Total Tokens Used: {assistant.total_tokens_used}", style="bold magenta"))

if __name__ == "__main__":
    main()
