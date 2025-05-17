from pydantic import BaseModel
from typing import List, Optional

class DetectionResponse(BaseModel):
    detected_objects: List[dict]
    description: Optional[str] = None
    detected_text: Optional[str] = None