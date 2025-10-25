import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from solver import Optimizer, EventInstanceEntry, create_sample_data

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-this-in-production")

ALGORITHM = "HS256"

# FastAPI app
app = FastAPI(title="JWT Protected API", version="1.0.0")

# Security scheme
security = HTTPBearer()

# Pydantic models
class TokenData(BaseModel):
    user: int
    role: int
    bodegaId: int
    iat: int
    exp: int

class EchoRequest(BaseModel):
    message: str

class EchoResponse(BaseModel):
    echo: str
    timestamp: str

class OptimizationRequest(BaseModel):
    occurrences: Dict[str, Dict[str, Any]]  # Accept plain dicts, will convert to EventInstanceEntry
    travel_distances: Dict[str, int]  # Accept string keys like "2,4", will convert to tuples
    buffer_minutes: Optional[int] = 60

class OptimizationResponse(BaseModel):
    schedule: Dict[int, List[str]]
    days_used: int
    chosen_occurrences: List[str]
    status: str
    timestamp: str

# JWT functions
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"Payload: {payload}")
        
        # Validate required fields
        if not all(key in payload for key in ["user", "role", "bodegaId", "iat", "exp"]):
            raise credentials_exception
            
        token_data = TokenData(
            user=payload["user"],
            role=payload["role"],
            bodegaId=payload["bodegaId"],
            iat=payload["iat"],
            exp=payload["exp"]
        )
    except JWTError as e:
        print(f"JWTError: {e}")
        raise credentials_exception
    
    return token_data

# Routes
@app.get("/")
async def root():
    return {"message": "JWT Protected API is running"}

@app.post("/echo", response_model=EchoResponse)
async def echo(
    request: EchoRequest,
    # token_data: TokenData = Depends(verify_token)  # JWT auth temporarily disabled
):
    """Echo endpoint that requires JWT authentication"""
    return EchoResponse(
        echo=request.message,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_schedule(
    request: OptimizationRequest,
    # token_data: TokenData = Depends(verify_token)
):
    """Optimize event scheduling using the Optimizer class"""
    try:
        print(f"Request: {request}")
        
        # Convert occurrences from plain dicts to EventInstanceEntry objects
        occurrences = {}
        for key, occurrence_data in request.occurrences.items():
            occurrences[key] = EventInstanceEntry(
                event=occurrence_data["event"],
                day=occurrence_data["day"],
                start=occurrence_data["start"],
                end=occurrence_data["end"],
                loc=occurrence_data["loc"]
            )
        
        # Convert travel_distances from string keys to tuple keys
        travel_distances = {}
        for key, value in request.travel_distances.items():
            # Split string like "2,4" into tuple ("2", "4")
            from_loc, to_loc = key.split(',')
            travel_distances[(from_loc, to_loc)] = value
        
        print(f"Converted occurrences: {occurrences}")
        print(f"Converted travel_distances: {travel_distances}")
        
        optimizer = Optimizer(
            occurrences=occurrences,
            travel_distances=travel_distances,
            buffer_minutes=request.buffer_minutes
        )
        
        result = optimizer.optimize()
        
        return OptimizationResponse(
            schedule=result["schedule"],
            days_used=result["days_used"],
            chosen_occurrences=result["chosen_occurrences"],
            status=result["status"],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )

@app.get("/optimize/sample", response_model=OptimizationResponse)
async def optimize_sample_data(
    # token_data: TokenData = Depends(verify_token)  # JWT auth temporarily disabled
):
    """Optimize using sample data for testing purposes"""
    try:
        occurrences, travel_distances = create_sample_data()
        
        optimizer = Optimizer(
            occurrences=occurrences,
            travel_distances=travel_distances
        )
        
        result = optimizer.optimize()
        
        return OptimizationResponse(
            schedule=result["schedule"],
            days_used=result["days_used"],
            chosen_occurrences=result["chosen_occurrences"],
            status=result["status"],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample optimization failed: {str(e)}"
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7801)

if __name__ == "__main__":
    main()
