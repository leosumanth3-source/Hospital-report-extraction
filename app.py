import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from extractor import (
    ExtractionUnavailable,
    extract_medical_report,
)


app = FastAPI(
    title="Medical Report Extraction API",
    version="1.0.0",
)


class ExtractionRequest(BaseModel):
    description: str


@app.get("/")
def home():
    return {
        "message": "Medical Report Extraction API is running"
    }


@app.post("/extract")
def extract(request: ExtractionRequest):

    start = time.perf_counter()

    try:

        report = extract_medical_report(
            request.description
        )

        result = report.model_dump()

        # ----------------------------------------------------
        # Valid medical input
        # ----------------------------------------------------

        if result.get("status") == "valid":

            result.pop(
                "reason",
                None
            )

        # ----------------------------------------------------
        # Invalid input
        # ----------------------------------------------------

        elif result.get("status") == "invalid":

            result = {
                "status": "invalid",
                "reason": result.get("reason"),
            }

        print(
            f"API TOTAL: "
            f"{time.perf_counter() - start:.2f}s"
        )

        return result

    # --------------------------------------------------------
    # Bad request
    # --------------------------------------------------------

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # --------------------------------------------------------
    # Gemini unavailable / failed
    # --------------------------------------------------------

    except ExtractionUnavailable as error:

        print(
            f"Returning provider error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    # --------------------------------------------------------
    # Unexpected error
    # --------------------------------------------------------

    except Exception as error:

        print(
            f"Unexpected server error: {error!r}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unexpected server error.",
        )