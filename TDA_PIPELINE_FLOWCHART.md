# TDA Pipeline Architecture Flowchart

## Simplified Pipeline (Working)

```mermaid
flowchart TD
    A[Start: run_agents.py] --> B[Create TDAPipelineConfig]
    B --> C[Initialize SimplifiedTDAPipeline]
    
    C --> D[DataPreparationAgent]
    D --> D1[Load ARFF file]
    D1 --> D2[Extract metadata]
    D2 --> D3[Create CV folds]
    D3 --> D4[Save fold indices]
    
    D4 --> E[BaselineEvaluationAgent]
    E --> E1[Create EvaluationAgent]
    E1 --> E2[Test on holdout]
    E2 --> E3[Nested CV evaluation]
    E3 --> E4[Calculate baseline score]
    
    E4 --> F[DomainAgent]
    F --> F1[Analyze dataset]
    F1 --> F2[Generate domain context]
    F2 --> F3[Extract column descriptions]
    
    F3 --> G[AugmentAgent]
    G --> G1[Load reasoning prompt]
    G1 --> G2[Generate feature suggestions]
    G2 --> G3[Execute feature generation]
    G3 --> G4[Add new columns]
    
    G4 --> H[FeaturePruningAgent]
    H --> H1[Perform feature selection]
    H1 --> H2[AutoGluon pruning]
    H2 --> H3[Majority vote selection]
    
    H3 --> I[PerformanceTrackingAgent]
    I --> I1[Evaluate pruned dataset]
    I1 --> I2[Compare with previous]
    I2 --> I3[Decide continuation]
    
    I3 --> J{Continue?}
    J -->|Yes| F
    J -->|No| K[Final Evaluation]
    
    K --> L[Save Results]
    L --> L1[Save augmented columns]
    L1 --> L2[Log performance history]
    L2 --> M[End]
    
    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style J fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#fff8e1
    style G fill:#fce4ec
    style H fill:#e0f2f1
    style I fill:#f1f8e9
```

## Route-Based Pipeline (Has Data Flow Issues)

```mermaid
flowchart TD
    A[Start: TDAPipeline] --> B[Initialize Agents]
    B --> B1[DataPreparationAgent]
    B --> B2[BaselineEvaluationAgent]
    B --> B3[DomainAgent]
    B --> B4[AugmentAgent]
    B --> B5[FeaturePruningAgent]
    B --> B6[PerformanceTrackingAgent]
    B --> B7[EvaluationAgent]
    
    C[RoutePipeline] --> D[data_prep]
    D --> E[baseline_eval]
    E --> F[domain_analysis]
    F --> G[augment]
    G --> H[prune]
    H --> I[performance_track]
    I --> J{Continue?}
    J -->|Yes| G
    J -->|No| K[End]
    
    style A fill:#ffcdd2
    style C fill:#ffcdd2
    style J fill:#fff3e0
    style K fill:#c8e6c9
```

## Agent Data Flow

```mermaid
flowchart LR
    subgraph "Input Data"
        A1[ARFF File]
        A2[Configuration]
    end
    
    subgraph "DataPreparationAgent"
        B1[arff_to_dataframe]
        B2[extract_arff_metadata]
        B3[KFold splitting]
    end
    
    subgraph "BaselineEvaluationAgent"
        C1[Create EvaluationAgent]
        C2[Holdout evaluation]
        C3[Nested CV]
    end
    
    subgraph "DomainAgent"
        D1[summarize_dataframe]
        D2[GPT-4 analysis]
        D3[Domain context]
    end
    
    subgraph "AugmentAgent"
        E1[Load prompt template]
        E2[GPT-4 reasoning]
        E3[Feature generation]
        E4[Column addition]
    end
    
    subgraph "FeaturePruningAgent"
        F1[AutoGluon pruning]
        F2[Feature selection]
        F3[Majority voting]
    end
    
    subgraph "PerformanceTrackingAgent"
        G1[Performance evaluation]
        G2[Improvement check]
        G3[Continuation decision]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> C1
    B2 --> C1
    B3 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> F1
    F1 --> G1
    G1 --> E1
    
    style A1 fill:#e3f2fd
    style A2 fill:#e3f2fd
    style B1 fill:#f3e5f5
    style C1 fill:#e8f5e8
    style D1 fill:#fff8e1
    style E1 fill:#fce4ec
    style F1 fill:#e0f2f1
    style G1 fill:#f1f8e9
```

## Data Types and Interfaces

```mermaid
flowchart TD
    subgraph "Agent Input/Output Pattern"
        A[AgentInput] --> A1[data: Any]
        A --> A2[metadata: Dict]
        
        B[AgentOutput] --> B1[result: Any]
        B --> B2[metadata: Dict]
    end
    
    subgraph "DataPreparationAgent"
        C[DataPreparationInput]
        C1[data_path: str]
        C2[n_folds: int]
        C3[random_state: int]
        
        D[DataPreparationOutput]
        D1[df: DataFrame]
        D2[metadata: str]
        D3[fold_indices: list]
        D4[original_columns: list]
    end
    
    subgraph "BaselineEvaluationAgent"
        E[BaselineEvaluationInput]
        E1[df: DataFrame]
        E2[target_column: str]
        E3[n_folds: int]
        E4[test_size: float]
        E5[model: str]
        
        F[BaselineEvaluationOutput]
        F1[original_eval: float]
        F2[original_nested_cv_scores: List]
        F3[baseline_score: float]
    end
    
    subgraph "EvaluationAgent"
        G[EvaluationInput]
        G1[df: DataFrame]
        G2[target_column: str]
        G3[evaluation_type: str]
        G4[n_splits: int]
        G5[device: str]
        
        H[EvaluationOutput]
        H1[scores: List[float]]
        H2[mean_score: float]
        H3[std_score: float]
        H4[evaluation_type: str]
        H5[n_splits: int]
    end
    
    A --> C
    D --> E
    F --> G
    H --> B
    
    style A fill:#e1f5fe
    style B fill:#c8e6c9
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#fff8e1
    style H fill:#fff8e1
```

## Configuration Flow

```mermaid
flowchart TD
    A[Command Line Args] --> B[TDAPipelineConfig]
    B --> B1[data_path: str]
    B --> B2[num_columns_to_add: int]
    B --> B3[target_column: str]
    B --> B4[n_folds: int]
    B --> B5[test_size: float]
    B --> B6[model: str]
    B --> B7[max_augmentations: int]
    B --> B8[verbose: bool]
    
    B --> C[SimplifiedTDAPipeline]
    C --> D[DataPreparationAgent]
    C --> E[BaselineEvaluationAgent]
    C --> F[DomainAgent]
    C --> G[AugmentAgent]
    C --> H[FeaturePruningAgent]
    C --> I[PerformanceTrackingAgent]
    C --> J[EvaluationAgent]
    
    E --> E1[config: target_column, n_folds, test_size, model]
    I --> I1[config: target_column, n_folds, test_size, model]
    J --> J1[config: label, n_folds, test_size, model]
    
    style A fill:#e3f2fd
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style E1 fill:#e8f5e8
    style I1 fill:#e8f5e8
    style J1 fill:#e8f5e8
```

## Error Handling and Logging

```mermaid
flowchart TD
    A[Agent Execution] --> B{Success?}
    B -->|Yes| C[Return AgentOutput]
    B -->|No| D[Error Handling]
    
    D --> D1[Log Error]
    D1 --> D2[Return Error Output]
    D2 --> D3[Metadata: status=failed]
    
    C --> C1[Metadata: status=success]
    C1 --> E[Continue Pipeline]
    
    subgraph "Logging System"
        F[LoggingAgent]
        F1[write_to_logs]
        F2[Structured JSON logs]
        F3[Performance tracking]
    end
    
    E --> F
    D1 --> F
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style D fill:#ffcdd2
    style F fill:#fff3e0
```

## Key Benefits of Refactored Architecture

1. **Modularity**: Each agent has a single responsibility
2. **Testability**: Individual agents can be tested in isolation
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Multiple pipeline implementations
5. **Reusability**: Agents can be used in different workflows
6. **Standardization**: Consistent input/output patterns
7. **Error Handling**: Proper error propagation and logging
8. **Configuration**: Centralized configuration management

## Current Status

- ✅ **SimplifiedTDAPipeline**: Working correctly
- ❌ **TDAPipeline**: Has data flow issues between agents
- ✅ **Individual Agents**: All properly implemented
- ✅ **Data Flow**: Clean and predictable in simplified pipeline
- ✅ **Testing**: Comprehensive test coverage available
- ✅ **Documentation**: Complete architecture documentation
