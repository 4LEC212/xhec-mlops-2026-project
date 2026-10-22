"""Pydantic models of the inputs and outputs of the web service."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    """Physical measurements of an abalone.

    As in the Kaggle dataset, lengths (mm) and weights (g) are divided by 200.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sex": "M",
                    "length": 0.455,
                    "diameter": 0.365,
                    "height": 0.095,
                    "whole_weight": 0.514,
                    "shucked_weight": 0.2245,
                    "viscera_weight": 0.101,
                    "shell_weight": 0.15,
                }
            ]
        }
    )

    sex: Literal["M", "F", "I"] = Field(description="M (male), F (female) or I (infant)")
    length: float = Field(gt=0, description="Longest shell measurement")
    diameter: float = Field(gt=0, description="Shell diameter, perpendicular to the length")
    height: float = Field(gt=0, description="Height, with the meat in the shell")
    whole_weight: float = Field(gt=0, description="Weight of the whole abalone")
    shucked_weight: float = Field(gt=0, description="Weight of the meat")
    viscera_weight: float = Field(gt=0, description="Gut weight, after bleeding")
    shell_weight: float = Field(gt=0, description="Weight of the shell, after drying")


class PredictionOut(BaseModel):
    """Predicted number of rings of an abalone and the corresponding age."""

    rings: float = Field(description="Predicted number of rings")
    age: float = Field(description="Predicted age in years (rings + 1.5)")
