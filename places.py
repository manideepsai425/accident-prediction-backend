from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_data_store
from app.schemas import Place, PlacesResponse
from app.services.data_store import DataStore

router = APIRouter(tags=["places"])


@router.get("/places", response_model=PlacesResponse)
def places(data_store: DataStore = Depends(get_data_store)) -> PlacesResponse:
    out = [
        Place(name=name, latitude=info["lat"], longitude=info["lng"])
        for name, info in data_store.places.items()
    ]
    return PlacesResponse(places=out)
