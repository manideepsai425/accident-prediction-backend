from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_data_store
from app.schemas import AnalyticsResponse
from app.services.data_store import DataStore

router = APIRouter(tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics(data_store: DataStore = Depends(get_data_store)) -> AnalyticsResponse:
    return AnalyticsResponse(**data_store.analytics())
