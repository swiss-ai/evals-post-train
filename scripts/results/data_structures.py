"""
Data structures for evaluation results and samples.
Contains well-defined classes for organizing evaluation data.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass(frozen=True)
class Sample:
    """A single evaluation sample from a task."""
    sample_data: Dict[str, Any]


@dataclass(frozen=True)
class Metric:
    """A single metric with name and score."""
    name: str
    score: float


@dataclass(frozen=True)
class Task:
    """Complete task data including both metrics and samples."""
    task_name: str
    metrics: List[Metric]
    samples: List[Sample]
    
    @property
    def metric_count(self) -> int:
        return len(self.metrics)
    
    @property
    def sample_count(self) -> int:
        return len(self.samples)
    
    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get raw sample data as list of dictionaries."""
        return [sample.sample_data for sample in self.samples]


@dataclass(frozen=True)
class ModelEvaluation:
    """Complete evaluation data for a single model."""
    model_name: str
    tasks: List[Task]
    
    @property
    def total_metrics_count(self) -> int:
        return sum(task.metric_count for task in self.tasks)
    
    @property
    def total_samples_count(self) -> int:
        return sum(task.sample_count for task in self.tasks)
    
    @property
    def task_names(self) -> List[str]:
        return [task.task_name for task in self.tasks]
    
    def get_flattened_metrics(self) -> Dict[str, float]:
        """Get all metrics in a flat dictionary format."""
        flattened = {}
        for task in self.tasks:
            for metric in task.metrics:
                flattened[f"{task.task_name}/{metric.name}"] = metric.score
        return flattened
