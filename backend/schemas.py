"""
schemas.py — Pydantic v2 schemas for Fake Review Detection API
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=3,
        description="The full text of the review to analyze",
        examples=["Great food, friendly staff, and the ambiance was perfect for a date night. Will definitely come back!"]
    )
    rating: float = Field(
        default=4.0,
        ge=1.0,
        le=5.0,
        description="Star rating associated with the review (1.0 to 5.0)",
        examples=[4.0]
    )
    model_name: Literal["xgboost", "linearsvc", "sgd", "ensemble"] = Field(
        default="ensemble",
        description="Model to use for prediction: xgboost, linearsvc, sgd, or ensemble (majority vote)",
        examples=["ensemble"]
    )
    user_id: Optional[int] = Field(
        default=None,
        description="Optional reviewer/user ID. When None, uses safe training median.",
        examples=[None]
    )
    product_id: Optional[int] = Field(
        default=None,
        description="Optional restaurant/product ID. When None, uses safe training median.",
        examples=[None]
    )


class ModelPrediction(BaseModel):
    model_name: str = Field(..., description="Name of the model (XGBoost, LinearSVC, SGD, or Ensemble)")
    prediction: Literal["Fake", "Genuine"] = Field(..., description="Classification result")
    is_spam: bool = Field(..., description="True if classified as Fake / Spam, False if Genuine")
    spam_probability: float = Field(..., description="Estimated probability of being fake/spam [0.0, 1.0]")
    confidence: float = Field(..., description="Confidence score in the prediction [0.0, 1.0]")
    threshold: float = Field(..., description="Decision threshold applied for this model")


class ExplainabilitySignals(BaseModel):
    word_count: int = Field(..., description="Total word count in review")
    char_length: int = Field(..., description="Total character count in review")
    exclamation_count: int = Field(..., description="Number of exclamation marks")
    uppercase_ratio: float = Field(..., description="Ratio of uppercase characters to alphabetic characters")
    sentiment_score: float = Field(..., description="Sentiment compound score (-1.0 to +1.0 or 1.0 to 5.0)")
    sentiment_rating_mismatch: float = Field(..., description="Absolute difference between text sentiment and star rating")
    top_risk_signals: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons explaining why the review looks suspicious or genuine"
    )


class PredictResponse(BaseModel):
    final_verdict: Literal["Fake", "Genuine"] = Field(..., description="Final consensus classification")
    consensus_confidence: float = Field(..., description="Consensus confidence percentage [0.0, 1.0]")
    selected_model: ModelPrediction = Field(..., description="Prediction details for the requested model")
    all_models: List[ModelPrediction] = Field(
        ...,
        description="Side-by-side predictions for XGBoost, LinearSVC, and SGD for live UI comparison"
    )
    signals: ExplainabilitySignals = Field(..., description="Linguistic and behavioral risk signals")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchRowPrediction(BaseModel):
    row_index: int
    review_text: str
    rating: Optional[float] = None
    prediction: Literal["Fake", "Genuine"]
    is_spam: bool
    spam_probability: float
    confidence: float
    key_signals: List[str] = Field(default_factory=list)


class BatchSummary(BaseModel):
    total: int = Field(..., description="Total number of reviews processed")
    fake_count: int = Field(..., description="Number of reviews classified as Fake")
    genuine_count: int = Field(..., description="Number of reviews classified as Genuine")
    fake_percentage: float = Field(..., description="Percentage of reviews flagged as Fake")
    average_confidence: float = Field(..., description="Average confidence score across all reviews")


class BatchPredictResponse(BaseModel):
    summary: BatchSummary
    rows: List[BatchRowPrediction]
    model_used: str
    latency_ms: float
    export_id: Optional[str] = None


class SampleReview(BaseModel):
    id: int
    title: str
    review_text: str
    rating: float
    expected_label: Literal["Fake", "Genuine"]
    category: str
    explanation: str


class SamplesResponse(BaseModel):
    samples: List[SampleReview]


class HealthResponse(BaseModel):
    status: str = "healthy"
    loaded_models: List[str]
    memory_usage_mb: float
    device: str
    artifacts_loaded: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    dataset_metadata: Dict[str, Any]
