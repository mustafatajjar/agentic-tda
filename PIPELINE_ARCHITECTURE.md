# TDA Pipeline Architecture

## Overview

The TDA (Tabular Data Augmentation) system has been refactored from a monolithic `TDAAgent` class into a modular, pipeline-based architecture. This new design provides better separation of concerns, improved testability, and more flexible orchestration.

## Architecture Components

### 1. Core Pipeline Classes

#### `TDAPipeline` (Route-Based)
- Uses the existing `RoutePipeline` infrastructure
- Each agent decides the next step in the pipeline
- More modular and testable
- Better for complex routing logic

#### `SimplifiedTDAPipeline` (Sequential)
- Maintains the original logic flow
- Easier to understand and debug
- Better for simple, linear workflows
- More predictable execution

### 2. Specialized Agents

#### `DataPreparationAgent`
- **Responsibility**: Data loading, metadata extraction, fold splitting
- **Input**: Data path, number of folds
- **Output**: DataFrame, metadata, fold indices, original columns
- **Dependencies**: `arff_to_dataframe`, `extract_arff_metadata`

#### `BaselineEvaluationAgent`
- **Responsibility**: Establish baseline performance before augmentation
- **Input**: DataFrame, target column, evaluation parameters
- **Output**: Original evaluation scores, baseline score
- **Dependencies**: `EvaluationAgent`

#### `DomainAgent`
- **Responsibility**: Analyze dataset domain and context
- **Input**: DataFrame, ARFF metadata
- **Output**: Domain context (primary domain, column descriptions)
- **Dependencies**: OpenAI GPT-4, domain prompt template

#### `AugmentAgent`
- **Responsibility**: Generate new features using LLM reasoning
- **Input**: DataFrame, domain context, augmentation history
- **Output**: Augmented DataFrame, added columns, suggestions
- **Dependencies**: OpenAI GPT-4, reasoning prompt template

#### `FeaturePruningAgent`
- **Responsibility**: Feature selection and pruning
- **Input**: DataFrame, target column
- **Output**: Pruned DataFrame, selected features, pruning effectiveness
- **Dependencies**: `prune_features_binary_classification`

#### `PerformanceTrackingAgent`
- **Responsibility**: Track performance improvements and decide continuation
- **Input**: DataFrame, evaluation parameters, iteration info
- **Output**: Current score, improvement, continuation decision
- **Dependencies**: `EvaluationAgent`

#### `LoggingAgent`
- **Responsibility**: Comprehensive logging and monitoring
- **Input**: Iteration details, prompts, responses, scores
- **Output**: Log file paths, log entry counts
- **Dependencies**: `write_to_logs`, file system

#### `ConfigAgent`
- **Responsibility**: Configuration validation and management
- **Input**: Pipeline configuration parameters
- **Output**: Validation results, errors, warnings
- **Dependencies**: File system, environment variables

## Pipeline Flow

### Route-Based Pipeline (`TDAPipeline`)

```
Data Prep → Baseline Eval → Domain Analysis → [Augment → Prune → Performance Track]*
```

**Routing Logic:**
- `data_prep` → `baseline_eval`
- `baseline_eval` → `domain_analysis`
- `domain_analysis` → `augment`
- `augment` → `prune`
- `prune` → `performance_track`
- `performance_track` → `augment` (if continuing) or `None` (if stopping)

### Simplified Pipeline (`SimplifiedTDAPipeline`)

```
Data Prep → Baseline Eval → [Domain Analysis → Augment → Prune → Performance Track]*
```

**Execution:**
- Linear execution with explicit control flow
- While loop for iteration management
- Direct agent calls without routing

## Data Flow

### Agent Input/Output Pattern

All agents follow the standard `AgentInput`/`AgentOutput` pattern:

```python
@dataclass
class AgentInput:
    data: Any
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentOutput:
    result: Any
    metadata: Optional[Dict[str, Any]] = None
```

### Data Transformation Chain

1. **Raw Data** → `DataPreparationAgent` → **Structured DataFrame**
2. **DataFrame** → `BaselineEvaluationAgent` → **Baseline Scores**
3. **DataFrame + Metadata** → `DomainAgent` → **Domain Context**
4. **DataFrame + Context** → `AugmentAgent` → **Augmented DataFrame**
5. **Augmented DataFrame** → `FeaturePruningAgent` → **Pruned DataFrame**
6. **Pruned DataFrame** → `PerformanceTrackingAgent` → **Performance Metrics**

## Configuration

### `TDAPipelineConfig`

```python
@dataclass
class TDAPipelineConfig:
    data_path: str
    num_columns_to_add: int = 20
    target_column: str = "class"
    n_folds: int = 10
    test_size: float = 0.2
    model: str = "tabpfn"
    max_augmentations: int = 10
    verbose: bool = True
```

### Configuration Validation

The `ConfigAgent` validates:
- File existence and format
- Parameter ranges and types
- Environment variables
- System resources

## Usage Examples

### Basic Usage

```python
from src.agents.tda_pipeline import SimplifiedTDAPipeline, TDAPipelineConfig

# Create configuration
config = TDAPipelineConfig(
    data_path="./data/dataset.arff",
    num_columns_to_add=10,
    max_augmentations=5
)

# Create and run pipeline
pipeline = SimplifiedTDAPipeline(config)
results = pipeline.run()

print(f"Final score: {results['final_score']:.4f}")
print(f"Improvement: {results['final_score'] - results['baseline_score']:.4f}")
```

### Route-Based Pipeline

```python
from src.agents.tda_pipeline import TDAPipeline

pipeline = TDAPipeline(config)
results = pipeline.run()
```

### Command Line

```bash
# Use simplified pipeline (default)
python -m src.run_agents --data_path ./data/dataset.arff --num_columns_to_add 10

# Use route-based pipeline
python -m src.run_agents --data_path ./data/dataset.arff --num_columns_to_add 10 --use_simplified
```

## Testing

### Running Tests

```bash
# Run all pipeline tests
python -m pytest tests/test_tda_pipeline.py -v

# Run specific test
python -m pytest tests/test_tda_pipeline.py::TestTDAPipeline::test_config_agent_validation -v
```

### Test Coverage

The test suite covers:
- Individual agent functionality
- Configuration validation
- Pipeline integration
- Error handling
- Mock external dependencies

## Benefits of New Architecture

### 1. **Modularity**
- Each agent has a single, well-defined responsibility
- Easy to add, remove, or modify agents
- Clear interfaces between components

### 2. **Testability**
- Individual agents can be tested in isolation
- Mock external dependencies easily
- Comprehensive test coverage

### 3. **Maintainability**
- Smaller, focused classes
- Clear separation of concerns
- Easier to debug and modify

### 4. **Flexibility**
- Multiple pipeline implementations
- Easy to add new routing logic
- Configurable agent behavior

### 5. **Reusability**
- Agents can be used in different pipelines
- Standard input/output interfaces
- Easy to compose new workflows

## Migration Guide

### From Old `TDAAgent`

1. **Replace direct instantiation:**
   ```python
   # Old
   agent = TDAAgent(data_path="...", ...)
   agent.run()
   
   # New
   config = TDAPipelineConfig(data_path="...", ...)
   pipeline = SimplifiedTDAPipeline(config)
   results = pipeline.run()
   ```

2. **Access results:**
   ```python
   # Old
   # Results were printed to logs
   
   # New
   results = pipeline.run()
   print(f"Final score: {results['final_score']}")
   print(f"Total iterations: {results['total_iterations']}")
   ```

3. **Configuration:**
   ```python
   # Old
   # Parameters passed directly to constructor
   
   # New
   config = TDAPipelineConfig(
       data_path="...",
       num_columns_to_add=10,
       # ... other parameters
   )
   ```

## Future Enhancements

### 1. **Advanced Routing**
- Conditional routing based on performance
- Dynamic agent selection
- A/B testing different augmentation strategies

### 2. **Monitoring & Observability**
- Real-time performance metrics
- Agent execution timing
- Resource usage tracking

### 3. **Distributed Execution**
- Parallel agent execution
- Distributed feature generation
- Multi-GPU evaluation

### 4. **Configuration Management**
- YAML/JSON configuration files
- Environment-specific configs
- Dynamic parameter tuning

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   - Check `ConfigAgent` output for validation errors
   - Verify environment variables (especially `OPENAI_API_KEY`)
   - Ensure data file exists and is readable

2. **Agent Failures**
   - Check agent metadata for error details
   - Verify input data format
   - Check external service availability (OpenAI API)

3. **Pipeline Stuck**
   - Check routing logic in route-based pipeline
   - Verify loop termination conditions
   - Check `max_steps` safety limit

### Debug Mode

Enable verbose logging:
```python
config = TDAPipelineConfig(..., verbose=True)
```

Check agent outputs:
```python
results = pipeline.run()
print(f"Agent metadata: {results['final_output'].metadata}")
```
