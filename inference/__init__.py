"""FastAPI backend for the PitchGPT counterfactual demo.

Wraps the causal layer (``causal.g_computation``, ``causal.positivity``,
``causal.sensitivity``) behind a single HTTP endpoint that the demo frontend
consumes. The point of this layer is to be a *thin* glue — no new science,
no business logic that isn't already in ``causal/``. Schema + serialization
only.

See ``inference.api`` for the FastAPI app and ``inference.schemas`` for the
request/response Pydantic models.
"""
