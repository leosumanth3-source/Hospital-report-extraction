import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from schema import MedicalReport


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# EXTRACTION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a medical report information extraction system.

Extract patient and medical information ONLY from the provided
description and return ONLY the requested JSON structure.

CORE RULES:

1. Determine whether the description contains patient or medical
   information.

2. Medical or patient information means information about a patient,
   medical history, examination, health condition, diagnosis,
   symptoms, treatment, or other patient details.

3. Medical/patient information → status = "valid".

4. Completely unrelated information → status = "invalid" and provide
   a short reason. For invalid input, return only status and reason.

5. Never invent, diagnose, or assume information that is not supported
   by the description.

6. For valid input, extract every supported patient field and medical
   condition. Fields that are not mentioned or cannot be determined
   must be null.


MEDICAL CONDITIONS:

7. Apply this rule independently to EVERY medical condition:

   - Explicitly present → "Yes"
   - Explicitly denied, absent, or specifically stated as not diagnosed
     → "No"
   - Not mentioned, uncertain, or only generally referred to
     → null

8. Missing information is NOT the same as "No".

9. Do not infer diseases from medications.

10. Do not use family history as the patient's diagnosis.

11. A general statement such as "I have no other medical conditions"
    does NOT mean every individual condition is "No".

12. Medical condition fields may contain ONLY:
    "Yes", "No", or null.


PATIENT INFORMATION:

13. Extract patient information only when supported by the description.

14. Do not calculate age from date of birth.
    Extract age only when explicitly stated.

15. Normalize unambiguous dates to YYYY-MM-DD.

16. Convert height and weight units only when the original units
    are clear.


SEX/GENDER:

17. If sex or gender is explicitly stated, use that value.

18. If sex/gender is not explicitly stated, estimate the most likely
    sex from the patient's name.

19. Name-based estimation is only a fallback when explicit
    sex/gender information is unavailable.

20. If the name does not provide enough information for a reasonable
    estimate, return null.

21. An explicit sex/gender statement always overrides a name-based
    estimate.

22. Never use age, medical conditions, medication, family history,
    or unrelated information to determine sex.

Examples:

"My name is Janaki."
→ sex = "Female"

"My name is Rahul."
→ sex = "Male"

"My name is Alex."
→ estimate the most likely sex from the name

"My name is Janaki and I am male."
→ sex = "Male"

"My name is Rahul and I am female."
→ sex = "Female"


OUTPUT:

23. For valid input:
    - status = "valid"
    - return patient data
    - return medical_assessment
    - reason must be null

24. For invalid input:
    - status = "invalid"
    - return a short reason
    - patient must be null
    - medical_assessment must be null

25. Return ONLY the JSON structure.

26. Do not return explanations, comments, markdown, code fences,
    or additional fields.
"""


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class ExtractionUnavailable(Exception):
    pass


# ============================================================
# MEDICAL REPORT EXTRACTION
# ============================================================

def extract_medical_report(description: str) -> MedicalReport:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not description or not description.strip():
        raise ValueError(
            "Patient description cannot be empty."
        )

    start = time.perf_counter()

    print(
        f"Starting extraction with {MODEL_NAME}"
    )

    try:

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = f"""
{SYSTEM_PROMPT}

PATIENT DESCRIPTION:

{description}
"""

        # ----------------------------------------------------
        # Gemini API call
        # ----------------------------------------------------

        api_start = time.perf_counter()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=400,
                response_mime_type="application/json",
                response_schema=MedicalReport,
            ),
        )

        api_time = time.perf_counter() - api_start

        print(
            f"Gemini API call: {api_time:.2f}s"
        )

        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if not response.text:
            raise ExtractionUnavailable(
                "Gemini returned an empty response."
            )

        # ----------------------------------------------------
        # Pydantic validation
        # ----------------------------------------------------

        validation_start = time.perf_counter()

        report = MedicalReport.model_validate_json(
            response.text
        )

        validation_time = (
            time.perf_counter()
            - validation_start
        )

        print(
            f"JSON validation: "
            f"{validation_time:.3f}s"
        )

        # ----------------------------------------------------
        # Total time
        # ----------------------------------------------------

        total_time = time.perf_counter() - start

        print(
            f"TOTAL extraction: "
            f"{total_time:.2f}s"
        )

        return report

    # --------------------------------------------------------
    # Input validation error
    # --------------------------------------------------------

    except ValueError:
        raise

    # --------------------------------------------------------
    # Gemini / provider error
    # --------------------------------------------------------

    except Exception as error:

        elapsed = time.perf_counter() - start

        print(
            "\n========== GEMINI ERROR =========="
        )

        print(
            f"Model: {MODEL_NAME}"
        )

        print(
            f"Time before error: "
            f"{elapsed:.2f}s"
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error: "
            f"{error!r}"
        )

        print(
            "==================================\n"
        )

        raise ExtractionUnavailable(
            str(error)
        ) from error