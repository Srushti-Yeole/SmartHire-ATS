"""
Unit and integration tests for User Authentication and Role-Based Access Control.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, User, hash_password, verify_password, create_user, authenticate_user

client = TestClient(app)


def test_password_hashing_and_verification():
    raw_pw = "SuperSecret123!"
    hashed = hash_password(raw_pw)

    assert hashed != raw_pw
    assert "$" in hashed
    assert verify_password(raw_pw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_create_and_authenticate_candidate_in_db():
    db = SessionLocal()
    email = "test.candidate@testdomain.com"
    # Cleanup if exists
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        db.delete(existing)
        db.commit()

    user = create_user(db, email=email, password="password123", role="candidate")
    assert user.id is not None
    assert user.email == email
    assert user.role == "candidate"
    assert user.company_name is None

    # Authenticate success
    auth_user = authenticate_user(db, email=email, password="password123")
    assert auth_user is not None
    assert auth_user.id == user.id

    # Authenticate failure
    assert authenticate_user(db, email=email, password="badpassword") is None
    db.close()


def test_create_recruiter_with_company_name():
    db = SessionLocal()
    email = "recruiter@hiringcorp.com"
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        db.delete(existing)
        db.commit()

    user = create_user(db, email=email, password="recruiterpassword", role="recruiter", company_name="Hiring Corp")
    assert user.role == "recruiter"
    assert user.company_name == "Hiring Corp"
    db.close()


def test_api_auth_register_and_login():
    reg_email = "api_user@test.org"
    reg_payload = {
        "email": reg_email,
        "password": "api_password_99",
        "role": "candidate"
    }

    # Register
    res_reg = client.post("/api/v1/auth/register", json=reg_payload)
    # Could be 200 or 400 if already exists
    if res_reg.status_code == 200:
        data = res_reg.json()
        assert data["email"] == reg_email
        assert data["role"] == "candidate"

    # Login
    login_payload = {
        "email": reg_email,
        "password": "api_password_99"
    }
    res_login = client.post("/api/v1/auth/login", json=login_payload)
    assert res_login.status_code == 200
    data = res_login.json()
    assert data["email"] == reg_email

    # Bad login
    res_bad = client.post("/api/v1/auth/login", json={"email": reg_email, "password": "wrong"})
    assert res_bad.status_code == 401
