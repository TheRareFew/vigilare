# Vigilare

Vigilare is an intelligent productivity tracking system that integrates with ActivityWatch to capture and analyze your work patterns, with a focus on LLM interactions and prompt management.

## Features

- Automatic screenshot capture based on user activity
- Sensitive information detection and blurring
- LLM prompt extraction and analysis
- Integration with ActivityWatch
- Vector-based prompt storage and similarity search
- Productivity analysis and reporting

## Requirements

- Python 3.8+
- ActivityWatch instance running
- Windows OS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vigilare.git
cd vigilare
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy and configure the example config files:
```bash
cp config/config.yaml.example config/config.yaml
cp config/logging_config.yaml.example config/logging_config.yaml
```

5. Edit the configuration files to match your setup.

## Usage

1. Ensure ActivityWatch is running
2. Start Vigilare:
```bash
python src/cli/main.py
```

## Project Structure

```
vigilare/
├── src/              # Source code
├── config/           # Configuration files
├── tests/            # Test files
├── data/             # Data storage
│   ├── screenshots/  # Captured screenshots
│   ├── vectors/      # Vector embeddings
│   └── logs/         # Application logs
└── scripts/          # Utility scripts
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Sort imports: `isort .`
- Lint code: `flake8`

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
