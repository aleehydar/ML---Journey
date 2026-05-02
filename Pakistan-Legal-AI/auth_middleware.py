import os
from dataclasses import dataclass
from typing import List

import jwt
from fastapi import Depends, Header, HTTPException, status


JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


@dataclass
class AuthClaims:
    sub: str
    org_id: str
    permissions: List[str]


def _extract_bearer_token(authorization: str) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
        )
    return authorization.split(" ", 1)[1].strip()


def _decode_token(token: str) -> dict:
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT is not configured on server",
        )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(exc)}",
        ) from exc
    return payload


def get_auth_claims(authorization: str = Header(default="")) -> AuthClaims:
    token = _extract_bearer_token(authorization)
    payload = _decode_token(token)
    sub = str(payload.get("sub", "")).strip()
    org_id = str(payload.get("org_id", "")).strip()
    permissions = payload.get("permissions", [])
    if not isinstance(permissions, list):
        permissions = []

    if not sub or not org_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims: sub/org_id",
        )
    return AuthClaims(sub=sub, org_id=org_id, permissions=[str(p) for p in permissions])


def require_permission(permission: str):
    def _checker(claims: AuthClaims = Depends(get_auth_claims)) -> AuthClaims:
        if permission not in claims.permissions and "*" not in claims.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission}",
            )
        return claims

    return _checker
