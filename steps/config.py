from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "LinearRegressionModel"
    # fine_tuning: bool = False
