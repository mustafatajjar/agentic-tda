# Agentic Tabular Data Augmentation (TDA)

## Setup Instructions

### 1. Environment Setup
```bash
make env
make data
export KG_INDEX_DIR=$(pwd)/data/kg-index
```

### 2. API Key Configuration
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. GRASP Configuration
Set the API key in `grasp/configs/single_kg.yaml`

## How to Run

### Main Pipeline
Run the complete agentic TDA pipeline:
```bash
python -m src.run_agents --data_path ./data/your_dataset --num_columns_to_add 10
```

### Parameters
- `--data_path`: Path to your dataset file
- `--num_columns_to_add`: Number of columns to add during each augmentation iteration (default: 10)

## References