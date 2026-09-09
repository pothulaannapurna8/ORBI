"""
Analyst Review Workflow API
Records human confirm/reject decisions and contextual notes in review_log.
Maintains full audit trail of analyst feedback for model improvement and quality assurance.

Endpoints:
- POST /review/{change_id}: Submit analyst review decision (confirmed/rejected) + notes
- GET /review/{change_id}: Retrieve review history for a change result
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.database.db_manager import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["Review"])


class ReviewSubmission(BaseModel):
    """Request body for analyst review submission."""
    decision: Literal["confirmed", "rejected"] = Field(
        ...,
        description="Analyst decision: 'confirmed' (valid change) or 'rejected' (false positive)"
    )
    analyst_note: Optional[str] = Field(
        default="",
        description="Free-text annotation (e.g., reasoning, confidence level, additional context)"
    )
    analyst_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the analyst submitting the review"
    )
    confidence_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional analyst confidence override [0.0, 1.0]"
    )


class ReviewResponse(BaseModel):
    """Response model for review submission."""
    status: str
    review_id: str
    change_id: str
    decision: str
    analyst_note: str
    reviewed_at: str


class ReviewHistoryResponse(BaseModel):
    """Response model for review history."""
    change_id: str
    reviews: List[Dict[str, Any]]
    total_reviews: int
    confirmed_count: int
    rejected_count: int


@router.post("/{change_id}", response_model=ReviewResponse)
async def record_analyst_review(
    change_id: str,
    submission: ReviewSubmission
) -> Dict[str, Any]:
    """
    Record analyst review decision and notes for a change result.
    
    This endpoint:
    1. Validates the change_id exists
    2. Stores the analyst decision (confirmed/rejected)
    3. Saves contextual notes for audit trail
    4. Optionally allows confidence overrides for edge cases
    5. Records timestamp
    
    The review log is used for:
    - Auditing model predictions
    - False positive/negative analysis
    - Model retraining feedback
    - Regulatory compliance (proof-of-review)
    
    Args:
        change_id: UUID of the change result to review
        submission: ReviewSubmission containing:
            - decision: "confirmed" or "rejected"
            - analyst_note: Free-text annotation
            - analyst_id: Optional identifier
            - confidence_override: Optional [0.0, 1.0]
    
    Returns:
        {
            "status": "recorded",
            "review_id": str (UUID),
            "change_id": str,
            "decision": str,
            "analyst_note": str,
            "reviewed_at": str (ISO timestamp)
        }
    
    Raises:
        HTTPException 400: Invalid decision value
        HTTPException 404: Change result not found
        HTTPException 500: Database error
    """
    try:
        # Validate decision
        if submission.decision not in ["confirmed", "rejected"]:
            logger.warning(
                f"Invalid decision '{submission.decision}' for change {change_id}; "
                "must be 'confirmed' or 'rejected'"
            )
            raise HTTPException(
                status_code=400,
                detail="Decision must be 'confirmed' or 'rejected'."
            )
        
        # Verify change_id exists
        change_result = db.get_change_result(change_id)

        if not change_result:
            logger.warning(f"Attempted review of non-existent change: {change_id}")
            raise HTTPException(status_code=404, detail="Change result not found.")
        
        # Insert review into database
        review_id = db.insert_review(
            change_id=change_id,
            decision=submission.decision,
            note=submission.analyst_note or "",
            analyst_id=submission.analyst_id,
            confidence_override=submission.confidence_override
        )
        
        reviewed_at = datetime.now().isoformat()
        
        logger.info(
            f"Review recorded: change_id={change_id}, decision={submission.decision}, "
            f"review_id={review_id}, analyst={submission.analyst_id}"
        )
        
        return {
            "status": "recorded",
            "review_id": review_id,
            "change_id": change_id,
            "decision": submission.decision,
            "analyst_note": submission.analyst_note or "",
            "reviewed_at": reviewed_at
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording review for change {change_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record review")


@router.get("/{change_id}", response_model=ReviewHistoryResponse)
async def get_review_history(change_id: str) -> Dict[str, Any]:
    """
    Retrieve complete review history for a change result.
    
    Shows all analyst decisions and notes, useful for:
    - Audit trails
    - Consensus checking (multiple reviewers)
    - Traceability for regulated workflows
    
    Args:
        change_id: UUID of the change result
    
    Returns:
        {
            "change_id": str,
            "reviews": [
                {
                    "review_id": str,
                    "decision": str ("confirmed" | "rejected"),
                    "analyst_note": str,
                    "analyst_id": str,
                    "reviewed_at": str (ISO timestamp)
                },
                ...
            ],
            "total_reviews": int,
            "confirmed_count": int,
            "rejected_count": int
        }
    
    Raises:
        HTTPException 404: Change result not found
    """
    try:
        # Verify change exists
        change_result = db.get_change_result(change_id)
        if not change_result:
            raise HTTPException(status_code=404, detail="Change result not found.")
        
        reviews = db.get_reviews_for_change(change_id)
        
        if not reviews:
            reviews = []
        
        # Compute statistics
        total_reviews = len(reviews)
        confirmed_count = sum(1 for r in reviews if r.get("decision") == "confirmed")
        rejected_count = sum(1 for r in reviews if r.get("decision") == "rejected")
        
        logger.debug(
            f"Retrieved review history for {change_id}: "
            f"{total_reviews} reviews ({confirmed_count} confirmed, {rejected_count} rejected)"
        )
        
        return {
            "change_id": change_id,
            "reviews": reviews,
            "total_reviews": total_reviews,
            "confirmed_count": confirmed_count,
            "rejected_count": rejected_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving review history for {change_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve review history")


@router.get("/")
async def list_recent_reviews(
    limit: int = 20,
    decision_filter: Optional[Literal["confirmed", "rejected"]] = None
) -> Dict[str, Any]:
    """
    List recent reviews across all changes (for analyst dashboard).
    
    Args:
        limit: Maximum number of reviews to return
        decision_filter: Optional filter ("confirmed" or "rejected")
    
    Returns:
        {
            "total_count": int,
            "reviews": [
                {
                    "review_id": str,
                    "change_id": str,
                    "decision": str,
                    "analyst_note": str,
                    "reviewed_at": str
                },
                ...
            ]
        }
    """
    try:
        reviews = db.get_recent_reviews(limit=limit, decision_filter=decision_filter)
        return {
            "total_count": len(reviews),
            "reviews": reviews if reviews else []
        }
    except Exception as e:
        logger.error(f"Error listing recent reviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reviews")
