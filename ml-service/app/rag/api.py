from __future__ import annotations

import hmac
import os
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from .service import RagService

router = APIRouter(prefix="/rag", tags=["rag"])
_service = RagService()


def get_service() -> RagService:
    return _service


def check_key(x_service_key: str | None = Header(default=None)) -> None:
    """Same shared secret the Express API already uses for the rest of the ML service."""
    expected = os.environ.get("ML_SERVICE_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ML_SERVICE_KEY is not configured")
    if not x_service_key or not hmac.compare_digest(x_service_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="bad service key")


class QueryItem(BaseModel):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9_]+$")
    text: str = Field(min_length=3, max_length=400)
    k: int | None = Field(default=None, ge=1, le=8)


class RetrieveRequest(BaseModel):
    audience: Literal["clinician", "patient"]
    queries: list[QueryItem] = Field(min_length=1, max_length=8)
    topics: list[str] = Field(default_factory=list, max_length=6)
    max_passages: int | None = Field(default=None, ge=1, le=12)

    @field_validator("queries")
    @classmethod
    def _unique_ids(cls, value):
        ids = [q.id for q in value]
        if len(set(ids)) != len(ids):
            raise ValueError("query ids must be unique")
        return value


@router.get("/status", dependencies=[Depends(check_key)])
def rag_status(svc: RagService = Depends(get_service)):
    return svc.status()


@router.post("/retrieve", dependencies=[Depends(check_key)])
def rag_retrieve(req: RetrieveRequest, svc: RagService = Depends(get_service)):
    return svc.retrieve(
        audience=req.audience,
        queries=[q.model_dump() for q in req.queries],
        topics=[t.strip().lower() for t in req.topics],
        max_passages=req.max_passages,
    )


@router.post("/reload", dependencies=[Depends(check_key)])
def rag_reload(svc: RagService = Depends(get_service)):
    svc.reload()
    return svc.status()
