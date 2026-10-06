from typing import Literal

from pydantic import BaseModel


class Patient(BaseModel):

    patient_name: str | None

    date_of_birth: str | None

    age: int | None

    address: str | None

    sex: Literal["Male", "Female"] | None

    marital_status: Literal["Single", "Married"] | None

    blood_type: str | None

    height_cm: float | None

    weight_kg: float | None

    examination_date: str | None


class MedicalAssessment(BaseModel):

    high_blood_pressure: Literal["Yes", "No"] | None

    diabetes: Literal["Yes", "No"] | None

    asthma: Literal["Yes", "No"] | None

    heart_stroke: Literal["Yes", "No"] | None

    kidney_disease: Literal["Yes", "No"] | None

    liver_disease: Literal["Yes", "No"] | None

    thyroid_disorder: Literal["Yes", "No"] | None

    cancer: Literal["Yes", "No"] | None

    anemia: Literal["Yes", "No"] | None

    migraine: Literal["Yes", "No"] | None

    peptic_ulcer: Literal["Yes", "No"] | None

    depression_anxiety: Literal["Yes", "No"] | None

    hair_fall: Literal["Yes", "No"] | None

    vision_problem_blindness: Literal["Yes", "No"] | None

    hearing_loss: Literal["Yes", "No"] | None

    skin_disease: Literal["Yes", "No"] | None

    dental_problem: Literal["Yes", "No"] | None

    obesity: Literal["Yes", "No"] | None

    allergies: Literal["Yes", "No"] | None

    jaundice: Literal["Yes", "No"] | None


class MedicalReport(BaseModel):

    status: Literal["valid", "invalid"]

    reason: str | None

    patient: Patient | None

    medical_assessment: MedicalAssessment | None