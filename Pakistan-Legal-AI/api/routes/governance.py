from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from auth_middleware import AuthClaims, require_permission
from db.schema import db_schema

router = APIRouter(prefix="/governance", tags=["governance"])

@router.delete("/delete-by-org")
async def delete_by_org(
    claims: AuthClaims = Depends(require_permission("governance:write"))
) -> Dict[str, int]:
    """Delete all logged evaluation data for the authenticated organization."""
    try:
        deleted_count = db_schema.delete_by_org(claims.org_id)
        return {"deleted_records": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-by-user/{user_id}")
async def delete_by_user(
    user_id: str,
    claims: AuthClaims = Depends(require_permission("governance:write"))
) -> Dict[str, int]:
    """Delete all logged evaluation data for a specific user within the organization."""
    try:
        deleted_count = db_schema.delete_by_user(claims.org_id, user_id)
        return {"deleted_records": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
