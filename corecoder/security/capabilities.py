"""Tool capability declarations and pre-execution self checks.

The guard must not infer authority from a tool name.  A guarded tool declares
its permission scope, side effect class, input types, and output trust level;
undeclared or internally inconsistent capabilities fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SCOPES = {
    "filesystem:read",
    "filesystem:write",
    "process:execute",
    "agent:delegate",
    "context:read",
}
_SIDE_EFFECTS = {"none", "local_write", "dynamic", "delegated"}
_TRUST_LEVELS = {"trusted", "untrusted"}
_NETWORK_ACCESS = {"none", "dynamic", "delegated"}
_DECLARED_RISKS = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class CapabilityReport:
    """Result of validating a tool's declared authority."""

    valid: bool
    scope: str = ""
    side_effect: str = ""
    output_trust: str = "untrusted"
    reason: str = ""
    network_access: str = ""
    declared_risk: str = ""


def inspect_tool_capabilities(tool: Any, arguments: dict) -> CapabilityReport:
    """Validate capability metadata and relevant schema constraints.

    Argument-name/signature validation remains in :class:`Agent`; this check is
    deliberately security-focused and can also be used by other runtimes.
    """
    cls = type(tool)
    required_declarations = (
        "permission_scope",
        "side_effect",
        "network_access",
        "declared_risk",
        "input_types",
        "output_type",
    )
    missing = [name for name in required_declarations if name not in cls.__dict__]
    if missing:
        return CapabilityReport(
            False,
            reason=f"tool capability declaration missing: {', '.join(missing)}",
        )

    scope = getattr(tool, "permission_scope", "")
    side_effect = getattr(tool, "side_effect", "")
    output_trust = getattr(tool, "output_trust", "untrusted")
    network_access = getattr(tool, "network_access", "")
    declared_risk = getattr(tool, "declared_risk", "")
    if scope not in _SCOPES:
        return CapabilityReport(False, scope=scope, reason=f"unknown permission scope: {scope or '(empty)'}")
    if side_effect not in _SIDE_EFFECTS:
        return CapabilityReport(False, scope=scope, reason=f"unknown side-effect class: {side_effect or '(empty)'}")
    if output_trust not in _TRUST_LEVELS:
        return CapabilityReport(False, scope=scope, side_effect=side_effect,
                                reason=f"unknown output trust level: {output_trust}")
    if network_access not in _NETWORK_ACCESS:
        return CapabilityReport(False, scope, side_effect, output_trust,
                                f"unknown network access: {network_access or '(empty)'}")
    if declared_risk not in _DECLARED_RISKS:
        return CapabilityReport(False, scope, side_effect, output_trust,
                                f"unknown declared risk: {declared_risk or '(empty)'}")
    if network_access == "none" and scope == "process:execute":
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "process execution cannot claim zero network capability")

    if scope.endswith(":read") and side_effect != "none":
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "read capability cannot declare side effects")
    if scope == "filesystem:write" and side_effect != "local_write":
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "filesystem write capability must be a local write")
    if scope == "process:execute" and side_effect != "dynamic":
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "process execution must declare dynamic side effects")
    if scope == "agent:delegate" and side_effect != "delegated":
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "agent delegation must declare delegated side effects")

    schema = getattr(tool, "parameters", {})
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "tool parameters must declare object properties")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(name, str) for name in required):
        return CapabilityReport(False, scope, side_effect, output_trust,
                                "tool parameters contain an invalid required list")
    missing = set(required) - set(arguments)
    if missing:
        return CapabilityReport(False, scope, side_effect, output_trust,
                                f"required capability argument missing: {', '.join(sorted(missing))}")
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            return CapabilityReport(False, scope, side_effect, output_trust,
                                    f"argument is outside declared capability: {name}")
        error = _validate_value(name, value, spec)
        if error:
            return CapabilityReport(False, scope, side_effect, output_trust, error)

    return CapabilityReport(
        True,
        scope,
        side_effect,
        output_trust,
        "capability self-check passed",
        network_access,
        declared_risk,
    )


def _validate_value(name: str, value: Any, spec: dict) -> str | None:
    expected = spec.get("type")
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    python_type = type_map.get(expected)
    if python_type is not None:
        # bool is a subclass of int, but JSON Schema does not treat it as one.
        if expected in {"integer", "number"} and isinstance(value, bool):
            return f"argument {name} must be {expected}"
        if not isinstance(value, python_type):
            return f"argument {name} must be {expected}"
    if "enum" in spec and value not in spec["enum"]:
        return f"argument {name} is not one of the declared values"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            return f"argument {name} is below its declared minimum"
        if "maximum" in spec and value > spec["maximum"]:
            return f"argument {name} exceeds its declared maximum"
    return None
