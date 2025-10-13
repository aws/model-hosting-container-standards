from typing import Optional 
from pydantic import BaseModel, Field, field_validator


class SageMakerRegisterLoRAAdapterRequest(BaseModel):
    name: str
    src: str
    preload: bool = True
    # pin: bool = False
    # prompt_prefix: Optional[str] = Field(default=None)

    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, value: str):
        if value == "":
            raise ValueError("The parameter name cannot be an empty string.")
        return value
    
    @field_validator('src', mode='after')
    @classmethod
    def validate_src(cls, value: str):
        if value == "":
            raise ValueError("The parameter src cannot be an empty string.")
        return value


class SageMakerUpdateLoRAAdapterRequest(BaseModel):
    src: str
    preload: Optional[bool] = Field(default=None)
    pin: Optional[bool] = Field(default=None)
    prompt_prefix: Optional[str] = Field(default=None)


class SageMakerListLoRAAdaptersRequest(BaseModel):
    limit: int = 100
    next_page_token: Optional[int] = Field(default=None)
