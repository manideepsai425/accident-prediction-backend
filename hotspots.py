from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import get_data_store
from app.schemas import HotspotsResponse
from app.services.data_store import DataStore

router = APIRouter(tags=["hotspots"])


@router.get("/hotspots", response_model=HotspotsResponse)
def hotspots(
    top_n: int = Query(15, ge=1, le=60),
    data_store: DataStore = Depends(get_data_store),
) -> HotspotsResponse:
    spots = data_store.hotspots(top_n=top_n)
    return HotspotsResponse(total_hotspots=len(spots), hotspots=spots)
