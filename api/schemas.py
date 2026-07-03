from pydantic import BaseModel, Field
from typing import Annotated

class TicketRequest(BaseModel):
    user_input:Annotated[str, Field(..., description='User input to be classified')]

class TicketResponse(BaseModel):
    is_spam: Annotated[bool, Field(..., description='Is the user input a spam?')]
    department: Annotated[str | None, Field(..., description='The department the input mail should be routed to')]
    confidence: Annotated[float | None, Field(..., ge=0, le=100.00, description='Confidence level of the model on classification result')]
    