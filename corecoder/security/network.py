"""Pre-execution network intent classification and egress policy.

This layer is deliberately conservative. It provides review and audit before a
tool runs; only a container network policy or host firewall can enforce egress
against arbitrary code after execution begins.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import unicodedata
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from .capabilities import CapabilityReport

NetworkAction = Literal["allow", "ask", "deny"]

_URL_RE = re.compile(r"\b(?:https?|ftp|ssh|git|git\+ssh)://[^\s\"'<>]+", re.IGNORECASE)
_NETWORK_CLIENT_RE = re.compile(
    r"\b(?:curl|wget|invoke-webrequest|invoke-restmethod|iwr|irm|ssh|scp|sftp|ftp|telnet|"
    r"netcat|ncat|nc|socat|ping|nslookup|dig)\b"
    r"|\bopenssl\s+s_client\b"
    r"|\bgit(?:\s+-C\s+\S+)?\s+(?:clone|fetch|pull|push|ls-remote|submodule\s+update)\b"
    r"|\b(?:python|python3)\s+-m\s+pip\s+(?:install|download)\b"
    r"|\b(?:pip|pip3|uv\s+pip)\s+(?:install|download)\b"
    r"|\b(?:npm|pnpm|yarn|composer|gem|brew|winget|choco)\s+(?:install|add|publish)\b"
    r"|\b(?:docker|podman)\s+(?:pull|push|login)\b"
    r"|\b(?:twine|cargo)\s+(?:upload|publish|install|add)\b"
    r"|\bgo\s+(?:get|install)\b"
    r"|\b(?:apt|apt-get|dnf|yum|apk)\s+(?:install|update|upgrade)\b"
    r"|\b(?:python|python3)\b[^\r\n]*(?:requests|urllib|httpx|aiohttp|socket)\."
    r"|\bnode\b[^\r\n]*(?:fetch\s*\(|axios\.|https?\.)",
    re.IGNORECASE,
)
_MUTATING_RE = re.compile(
    r"\bcurl\b[^\r\n]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--request\s*(?:POST|PUT|PATCH|DELETE)"
    r"|(?:^|\s)(?:-d|-F|-T)(?:\s|[^-])|--data(?:-\w+)?\b|--form\b|--json\b|--upload-file\b)"
    r"|\bwget\b[^\r\n]*(?:--post-data|--post-file|--method[=\s]+(?:POST|PUT|PATCH|DELETE)|--body-data)"
    r"|\bgit(?:\s+-C\s+\S+)?\s+push\b|\b(?:ssh|scp|sftp)\b"
    r"|\b(?:npm|pnpm|yarn|twine|cargo)\s+(?:publish|upload)\b"
    r"|\b(?:docker|podman)\s+(?:push|login)\b",
    re.IGNORECASE,
)
_REDIRECT_RE = re.compile(r"\bcurl\b[^\r\n]*(?:^|\s)(?:-L|--location)(?:\s|$)", re.IGNORECASE)
_CREDENTIAL_OPTION_RE = re.compile(
    r"(?:^|\s)(?:-u|--user)(?:\s|=)|authorization\s*:\s*(?:bearer|basic)\b",
    re.IGNORECASE,
)
_SCP_STYLE_HOST_RE = re.compile(r"(?:[\w.-]+@)?([\w.-]+):[^/\\\s]")
_KNOWN_OFFLINE_COMMAND_RE = re.compile(
    r"^(?:"
    r"(?:ls|dir|cat|head|tail|less|more|pwd|whoami|date|printenv|env|cd|wc|df|du)"
    r"(?:\s+[^<>|;&]*)?"
    r"|git(?:\s+-C\s+\S+)?\s+(?:status|diff|log|show|rev-parse|ls-files)"
    r"(?:\s+[^<>|;&]*)?"
    r"|hg\s+(?:status|diff|log|cat|id)(?:\s+[^<>|;&]*)?"
    r"|svn\s+(?:status|diff|log|info|list|cat)(?:\s+[^<>|;&]*)?"
    r"|(?:python|python3|pip|pip3|node|npm|npx|yarn|pnpm)\s+"
    r"(?:--version|-V|-v|--help|-h)"
    r")\s*$",
    re.IGNORECASE,
)

_METADATA_HOSTS = {
    "metadata.google.internal",
    "metadata.google",
    "metadata.aws.internal",
    "metadata.azure.internal",
    "100.100.100.200",  # Alibaba Cloud metadata endpoint
    "fd00:ec2::254",  # AWS Nitro IPv6 metadata endpoint
}


@dataclass(frozen=True)
class NetworkIntent:
    accesses_network: bool = False
    destinations: tuple[str, ...] = ()
    mutating: bool = False
    follows_redirects: bool = False
    embedded_credentials: bool = False
    carries_credentials: bool = False
    client: str = ""


@dataclass(frozen=True)
class NetworkDecision:
    action: NetworkAction
    reason: str
    intent: NetworkIntent = NetworkIntent()


@dataclass(frozen=True)
class NetworkPolicy:
    """Review detected egress against a host allowlist.

    ``confirm`` asks for unknown destinations; ``deny`` blocks them. Mutating,
    credential-bearing, private-network, and redirect-following calls receive
    additional handling regardless of the allowlist.
    """

    mode: Literal["confirm", "deny"] = "confirm"
    allowed_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"confirm", "deny"}:
            raise ValueError("network mode must be 'confirm' or 'deny'")
        normalized = tuple(_validate_host_pattern(host) for host in self.allowed_hosts)
        object.__setattr__(self, "allowed_hosts", normalized)

    def review(
        self,
        tool_name: str,
        arguments: dict,
        capability: CapabilityReport | None = None,
    ) -> NetworkDecision:
        intent = inspect_network_intent(tool_name, arguments, capability)
        if not intent.accesses_network:
            return NetworkDecision("allow", "no network intent detected", intent)
        if intent.embedded_credentials:
            return NetworkDecision("deny", "URL-embedded credentials are forbidden", intent)

        protected = [host for host in intent.destinations if _is_metadata_target(host)]
        if protected:
            return NetworkDecision(
                "deny",
                f"cloud metadata or link-local destination is forbidden: {', '.join(protected)}",
                intent,
            )
        private = [host for host in intent.destinations if _is_private_target(host)]
        if private:
            return NetworkDecision(
                "ask",
                f"private or loopback network destination requires confirmation: {', '.join(private)}",
                intent,
            )
        if intent.carries_credentials:
            return NetworkDecision("ask", "network request carries authentication material", intent)
        if intent.mutating:
            return NetworkDecision("ask", "network operation may upload data or mutate remote state", intent)
        if intent.follows_redirects:
            return NetworkDecision("ask", "redirect-following can leave the approved destination", intent)
        if not intent.destinations:
            action: NetworkAction = "deny" if self.mode == "deny" else "ask"
            return NetworkDecision(action, "network client has no statically verifiable destination", intent)
        if all(self._allowed(host) for host in intent.destinations):
            return NetworkDecision("allow", "all network destinations are allowlisted", intent)
        action = "deny" if self.mode == "deny" else "ask"
        return NetworkDecision(
            action,
            f"network destination is not allowlisted: {', '.join(intent.destinations)}",
            intent,
        )

    def _allowed(self, host: str) -> bool:
        candidate = host.casefold().rstrip(".")
        for pattern in self.allowed_hosts:
            if pattern.startswith("*."):
                suffix = pattern[2:]
                if candidate.endswith(f".{suffix}") and candidate != suffix:
                    return True
            elif candidate == pattern:
                return True
        return False


def inspect_network_intent(
    tool_name: str,
    arguments: dict,
    capability: CapabilityReport | None = None,
) -> NetworkIntent:
    """Extract a conservative network intent from a tool call."""
    if tool_name != "bash":
        if capability is not None and capability.network_access not in {"", "none", "delegated"}:
            return NetworkIntent(True, client=tool_name)
        return NetworkIntent()

    command = _normalize(str(arguments.get("command", "")))
    client_match = _NETWORK_CLIENT_RE.search(command)
    if client_match is None:
        # Process execution is an opaque authority boundary: a binary or script
        # can open sockets without advertising that fact in argv.  An explicit
        # permission allow must therefore not suppress network review.  Only a
        # narrow set of structurally read-only commands is treated as offline;
        # every other dynamic process is reviewed as unknown-destination egress.
        if (
            capability is not None
            and capability.network_access == "dynamic"
            and not _KNOWN_OFFLINE_COMMAND_RE.fullmatch(command.strip())
        ):
            return NetworkIntent(True, client="opaque-process")
        return NetworkIntent()

    destinations: list[str] = []
    embedded_credentials = False
    for raw_url in _URL_RE.findall(command):
        url = raw_url.rstrip(".,);")
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
        except ValueError:
            continue
        if parsed.username is not None or parsed.password is not None:
            embedded_credentials = True
        if any(
            key.casefold() in {"token", "access_token", "api_key", "apikey", "password", "secret"}
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            embedded_credentials = True
        if hostname:
            destinations.append(_canonical_host(hostname))

    ssh_host = _extract_ssh_host(command)
    if ssh_host:
        destinations.append(_canonical_host(ssh_host))
    for host in _SCP_STYLE_HOST_RE.findall(command):
        if host and "." in host:
            destinations.append(_canonical_host(host))

    return NetworkIntent(
        accesses_network=True,
        destinations=tuple(dict.fromkeys(destinations)),
        mutating=bool(_MUTATING_RE.search(command)),
        follows_redirects=bool(_REDIRECT_RE.search(command)) or client_match.group(0).casefold() == "wget",
        embedded_credentials=embedded_credentials,
        carries_credentials=bool(_CREDENTIAL_OPTION_RE.search(command)),
        client=client_match.group(0).strip().split()[0].casefold(),
    )


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Cf")


def _canonical_host(host: str) -> str:
    return host.casefold().rstrip(".")


def _extract_ssh_host(command: str) -> str:
    """Return SSH's first positional host without trusting later remote argv."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ""
    ssh_index = next(
        (index for index, token in enumerate(tokens) if token.casefold() in {"ssh", "ssh.exe"}),
        None,
    )
    if ssh_index is None:
        return ""
    options_with_values = {
        "-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l",
        "-m", "-O", "-o", "-P", "-p", "-Q", "-R", "-S", "-W", "-w",
    }
    index = ssh_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token in options_with_values:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    if index >= len(tokens):
        return ""
    host = tokens[index].rsplit("@", 1)[-1]
    return host.strip("[]") if re.fullmatch(r"[\w.:\[\]-]+", host) else ""


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip("[]").casefold()
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        pass
    try:
        if candidate.isdecimal():
            return ipaddress.ip_address(int(candidate, 10))
        if candidate.startswith("0x"):
            return ipaddress.ip_address(int(candidate, 16))
    except ValueError:
        pass
    return None


def _is_metadata_target(host: str) -> bool:
    candidate = _canonical_host(host)
    if candidate in _METADATA_HOSTS:
        return True
    address = _as_ip(candidate)
    return bool(address and address.is_link_local)


def _is_private_target(host: str) -> bool:
    candidate = _canonical_host(host)
    if candidate == "localhost" or candidate.endswith(".localhost"):
        return True
    address = _as_ip(host)
    return bool(address and (address.is_private or address.is_loopback or address.is_reserved))


def _validate_host_pattern(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("network allowlist entries must be strings")
    host = value.strip().casefold().rstrip(".")
    candidate = host.removeprefix("*.")
    if (
        not candidate
        or "://" in candidate
        or any(char in candidate for char in "/\\@?#")
        or not re.fullmatch(r"[a-z0-9.-]+", candidate)
        or candidate.startswith(".")
        or candidate.endswith(".")
        or ".." in candidate
        or (host.startswith("*.") and "." not in candidate)
    ):
        raise ValueError(f"invalid network allowlist host: {value!r}")
    if _is_metadata_target(candidate) or _is_private_target(candidate):
        raise ValueError("private, loopback, and metadata targets cannot be allowlisted")
    return host
