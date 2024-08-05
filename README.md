# GPT-4o-mini-Engineer

GPT-4 Mini Engineer is an advanced AI-powered software development assistant designed to help manage and execute complex development tasks. It leverages the capabilities of OpenAI's GPT-4 model to provide architectural insights, create and manage project structures, write and debug code, and much more.

## Features

- **Task Management**: Efficiently manage and prioritize tasks using a task queue.
- **Project Context Handling**: Maintain project context to track the current state of development.
- **Automode**: Automate repetitive tasks with dynamic task generation and prioritization.
- **Error Handling**: Robust error handling and retry mechanisms for API calls.
- **State Saving/Loading**: Save and load the project state to resume work later.
- **Interactive Task Management**: View, add, and modify tasks interactively.
- **Help Command**: Display available commands and their descriptions for easy reference.

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gpt4_mini_engineer.git
    cd gpt4_mini_engineer
    ```

2. **Set up the virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the root directory of the project with the following content:
    ```
    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CSE_ID=your_google_cse_id
    ```

## Usage

### Running the Assistant

1. **Start the assistant**:
    ```bash
    python main.py
    ```

2. **Interact with the assistant**:
    - Enter your project request when prompted.
    - Use the available commands to manage tasks, save/load state, and more.

### Available Commands

- `exit`: Quit the program
- `manage`: Manage tasks (view, add, modify)
- `automode [iterations]`: Enter automode with optional iteration count (default is 25)
- `save`: Save the current project state
- `load`: Load a saved project state
- `help`: Display available commands and their descriptions

## Project Structure

gpt4_mini_engineer/
│
├── main.py # Main entry point of the project
├── requirements.txt # List of project dependencies
├── .env # Environment variables (not included in the repository)
├── README.md # Project documentation
└── gpt4_mini_engineer.log # Log file for tracking the development process


## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or suggestions, please reach out to [tod-84@mail.ru].
