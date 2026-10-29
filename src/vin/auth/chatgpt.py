import jwt
import json
import webbrowser
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPResponse
from secrets import token_urlsafe
from threading import Event, Lock, Thread
from time import sleep, monotonic
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen
from jwt import PyJWKClient
from jwt.algorithms import AllowedPublicKeys
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from vin.types import (
    AccountRecord, AuthorizationCallback, Credentials, Registration, ExternalRecord, Identity, LoginAttempt,
    is_list, is_record,
)

from .credentials import (
    ACTIVE_SESSION_PATH,
    OPENAI_ISSUER,
    build_credential_record,
    get_host_id,
    is_access_token_expired,
    load_active_session,
    save_credential_record,
    save_private_json,
)

JWKS_CLIENT = PyJWKClient(
    "https://auth.openai.com/.well-known/jwks.json",
    timeout=10,
)


class TokenExchangeError(RuntimeError):
    """Safe token-endpoint diagnostics without provider bodies or credentials."""

    LOGIN_CODES = frozenset({
        "invalid_grant", "invalid_refresh_token", "token_expired",
        "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
    })

    def __init__(self, status_code: int, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code if code in self.LOGIN_CODES else None
        self.requires_login = self.code is not None
        message = f"Token exchange failed (HTTP {status_code})"
        if self.code:
            message += f"; code={self.code}. Sign in again with /login."
        super().__init__(message)


def prepare_login_attempt() -> LoginAttempt:
    state = token_urlsafe(32)
    nonce = token_urlsafe(32)
    code_verifier = token_urlsafe(32)
    digest = sha256(code_verifier.encode("ascii")).digest()
    code_challenge = urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return {
        "state": state,
        "nonce": nonce,
        "code_verifier": code_verifier,
        "code_challenge": code_challenge,
    }


def build_authorization_url(
    attempt: LoginAttempt,
    host_id: str,
    redirect_uri: str,
    registration: AccountRecord | None = None,
) -> str:
    params = {
        "client_id": "dynamic_agent_client",
        "agent_name_hint": "Vin",
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
        "resource": "https://api.openai.com/v1",
        "state": attempt["state"],
        "nonce": attempt["nonce"],
        "code_challenge_method": "S256",
        "code_challenge": attempt["code_challenge"],
    }

    if registration is not None:
        params["client_id"] = registration["client_id"]
        del params["agent_name_hint"]

    return "https://auth.openai.com/api/accounts/authorize?" + urlencode(params)


def parse_authorization_callback(
    callback_url: str,
    expected_state: str,
    expected_client_id: str | None = None,
) -> AuthorizationCallback:
    params = parse_qs(urlsplit(callback_url).query, keep_blank_values=True)

    states = params.get("state", [])

    if (
        len(states) != 1
        or not states[0]
        or not compare_digest(states[0].encode("utf-8"), expected_state.encode("utf-8"))
    ):
        raise ValueError("Sign-in state could not be verified")
    if "error" in params:
        raise RuntimeError("Sign-in was declined or failed")

    if expected_client_id is not None:
        supplied = params.get("client_id")
        if supplied is not None and supplied != [expected_client_id]:
            raise ValueError("Callback client ID did not match")
        params["client_id"] = [expected_client_id]

    result: dict[str, str] = {}
    for field in ("code", "client_id"):
        values = params.get(field, [])
        if len(values) != 1 or not values[0]:
            raise ValueError(f"Expected exactly one nonempty {field}")
        result[field] = values[0]

    if result["client_id"] == "dynamic_agent_client":
        raise ValueError("Expected an issued client ID")

    return {"code": result["code"], "client_id": result["client_id"]}


def request_tokens(form: Mapping[str, str]) -> ExternalRecord:
    request = Request(
        "https://auth.openai.com/api/accounts/oauth/token",
        data=urlencode(form).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with cast(HTTPResponse, urlopen(request, timeout=30)) as response:
            if response.status != 200:
                raise RuntimeError("Unexpected token response status")
            tokens = cast(object, json.load(response))
    except HTTPError as error:
        status = error.code
        code = None
        try:
            payload = cast(object, json.loads(error.read(8192)))
            if is_record(payload):
                detail = payload.get("error")
                candidate = detail.get("code") if is_record(detail) else detail
                if isinstance(candidate, str) and candidate in TokenExchangeError.LOGIN_CODES:
                    code = candidate
        except (OSError, ValueError):
            pass
        finally:
            error.close()
        raise TokenExchangeError(status, code) from None
    except OSError:
        raise RuntimeError("Token exchange request failed") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("Token endpoint returned invalid JSON") from None

    if not is_record(tokens):
        raise RuntimeError("Expected a JSON object from the token endpoint")
    return tokens


def exchange_authorization_code(
    callback: AuthorizationCallback, attempt: LoginAttempt, redirect_uri: str
) -> ExternalRecord:
    form = {
        "grant_type": "authorization_code",
        "client_id": callback["client_id"],
        "code": callback["code"],
        "code_verifier": attempt["code_verifier"],
        "redirect_uri": redirect_uri,
        "resource": "https://api.openai.com/v1",
    }
    return request_tokens(form)


def exchange_refresh_token(credentials: ExternalRecord) -> ExternalRecord:
    client_id = credentials.get("client_id")
    refresh_token = credentials.get("refresh_token")

    if (
        not isinstance(client_id, str)
        or not client_id
        or client_id == "dynamic_agent_client"
    ):
        raise RuntimeError("Refresh requires an issued client ID")

    if not isinstance(refresh_token, str) or not refresh_token:
        raise TokenExchangeError(401, "invalid_refresh_token")

    form = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
        "resource": "https://api.openai.com/v1",
    }
    return request_tokens(form)


def revoke_refresh_token(credentials: ExternalRecord) -> bool:
    refresh_token = credentials.get("refresh_token")
    client_id = credentials.get("client_id")
    if (
        not isinstance(refresh_token, str)
        or not refresh_token
        or not isinstance(client_id, str)
        or not client_id
        or client_id == "dynamic_agent_client"
    ):
        return False

    for attempt in range(2):
        try:
            with cast(HTTPResponse, urlopen(
                OPENAI_ISSUER + "/.well-known/openid-configuration",
                timeout=10,
            )) as response:
                metadata = cast(object, json.load(response))

            endpoint = (
                metadata.get("revocation_endpoint")
                if is_record(metadata)
                else None
            )

            if not isinstance(endpoint, str) or not endpoint.startswith(
                OPENAI_ISSUER + "/"
            ):
                return False

            request = Request(
                endpoint,
                data=urlencode(
                    {
                        "token": refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": client_id,
                    }
                ).encode("ascii"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with cast(HTTPResponse, urlopen(request, timeout=10)) as response:
                return response.status == 200
        except HTTPError as error:
            status = error.code
            error.close()
            if not 500 <= status < 600:
                return False
        except OSError:
            pass
        except (ValueError, UnicodeDecodeError):
            return False

        if attempt == 0:
            sleep(1)
    return False


def validate_id_token(id_token: object, client_id: str, expected_nonce: str) -> Identity:
    if not isinstance(id_token, str) or not id_token:
        raise RuntimeError("Token response did not contain an ID token")

    try:
        signing_key = JWKS_CLIENT.get_signing_key_from_jwt(id_token)
        claims = cast(object, jwt.decode(
            id_token,
            cast(AllowedPublicKeys | str | bytes, signing_key.key),
            algorithms=["RS256"],
            issuer=OPENAI_ISSUER,
            audience=client_id,
            options={"require": ["iss", "aud", "exp", "iat", "sub", "nonce"]},
            leeway=5,
        ))
    except (jwt.PyJWTError, ValueError, OSError):
        raise RuntimeError("ID token could not be verified") from None
    assert is_record(claims)
    nonce = claims["nonce"]
    if not isinstance(nonce, str) or not compare_digest(
        nonce.encode("utf-8"), expected_nonce.encode("utf-8")
    ):
        raise RuntimeError("ID token nonce did not match")

    if not isinstance(claims["sub"], str) or not claims["sub"]:
        raise RuntimeError("ID token did not contain a valid object")

    audience = claims["aud"]
    if is_list(audience) and len(audience) > 1:
        if claims.get("azp") != client_id:
            raise RuntimeError("ID token authorized party did not match")
    elif "azp" in claims and claims["azp"] != client_id:
        raise RuntimeError("ID token authorized party did not match")

    # PyJWT verifies the issuer; the checks above verify the subject and nonce.
    return cast(Identity, claims)


def refresh_credentials(credentials: Credentials) -> Credentials:
    if not is_access_token_expired(credentials):
        return credentials

    tokens = exchange_refresh_token(credentials)
    received_at = datetime.now(timezone.utc)

    refresh_token = tokens.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError(
            "Refresh response did not contain a replacement refresh token"
        )

    assert "scopes" in credentials
    assert "id_token" in credentials
    scope_value = tokens.get("scope", " ".join(credentials["scopes"]))
    if not isinstance(scope_value, str):
        raise RuntimeError("Refresh response contained invalid scopes")

    identity: Identity = {
        "iss": credentials["issuer"],
        "sub": credentials["subject"],
        "email": credentials.get("email"),
    }

    updated = build_credential_record(
        {**tokens, "id_token": credentials["id_token"]},
        identity,
        credentials["client_id"],
        credentials["ext_agent_host_id"],
        set(scope_value.split()),
        received_at,
    )

    save_credential_record(updated)
    return updated


class CallbackServerState(Protocol):
    server_port: int
    expected_state: str
    expected_client_id: str | None
    callback_result: AuthorizationCallback | None
    callback_error: RuntimeError | None
    callback_done: Event
    callback_lock: Lock

    def serve_forever(self, poll_interval: float = 0.5) -> None: ...
    def shutdown(self) -> None: ...


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/auth/callback":
            self.send_error(404)
            return

        # login_with_chatgpt initializes this state before starting the listener.
        server = cast(CallbackServerState, self.server)
        with server.callback_lock:
            if server.callback_done.is_set():
                self.send_error(409, "Sign-in attempt already finished")
                return

            try:
                callback = parse_authorization_callback(
                    self.path,
                    server.expected_state,
                    server.expected_client_id,
                )
            except ValueError:
                self.send_error(400, "Invalid sign-in callback")
                return
            except RuntimeError as error:
                server.callback_error = error
                try:
                    self.send_error(400, "Sign-in was declined or failed")
                finally:
                    server.callback_done.set()
                return
            server.callback_result = callback

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Callback received. Return to Vin.")
            finally:
                server.callback_done.set()

    def log_message(self, format: str, *args: object) -> None:
        pass


def login_with_chatgpt(
    registration: AccountRecord | None = None,
    *,
    cancel_event: Event | None = None,
    notify: Callable[[str], object] = print,
) -> Credentials | None:
    if cancel_event is not None and cancel_event.is_set():
        notify("Sign-in cancelled")
        return None
    attempt = prepare_login_attempt()
    host_id = get_host_id()

    if registration is not None and registration["ext_agent_host_id"] != host_id:
        raise RuntimeError("Saved registration belongs to a different host")

    with ThreadingHTTPServer(("127.0.0.1", 0), CallbackHandler) as http_server:
        server = cast(CallbackServerState, http_server)
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        server.expected_state = attempt["state"]
        server.expected_client_id = (
            registration["client_id"] if registration is not None else None
        )
        server.callback_result = None
        server.callback_error = None
        server.callback_done = Event()
        server.callback_lock = Lock()
        url = build_authorization_url(attempt, host_id, redirect_uri, registration)

        listener = Thread(target=server.serve_forever, daemon=True)
        listener.start()
        try:
            if not webbrowser.open(url):
                raise RuntimeError("Could not open the sign-in browser")
            deadline = monotonic() + 300
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    notify("Sign-in cancelled")
                    return None
                if server.callback_done.wait(0.1):
                    break
                if monotonic() >= deadline:
                    raise TimeoutError("Sign-in timed out")
            if server.callback_error is not None:
                raise server.callback_error
            callback = server.callback_result
            assert callback is not None
            tokens = exchange_authorization_code(
                callback, attempt, redirect_uri
            )
            if cancel_event is not None and cancel_event.is_set():
                notify("Sign-in cancelled")
                return None
            received_at = datetime.now(timezone.utc)
            identity = validate_id_token(
                tokens.get("id_token"),
                callback["client_id"],
                attempt["nonce"],
            )
            if registration is not None and (
                identity["iss"] != registration["issuer"]
                or identity["sub"] != registration["subject"]
            ):
                raise RuntimeError("Sign-in returned a different account")
            scope_value = tokens.get("scope", "")
            if not isinstance(scope_value, str):
                raise RuntimeError("Token response contained invalid scopes")

            scopes = set(scope_value.split())
            subscription_enabled = "chatgpt.tokens.use.direct" in scopes
            credentials = build_credential_record(
                tokens,
                identity,
                callback["client_id"],
                host_id,
                scopes,
                received_at,
            )
            if cancel_event is not None and cancel_event.is_set():
                notify("Sign-in cancelled")
                return None
            credential_path = save_credential_record(credentials)
            save_private_json(
                ACTIVE_SESSION_PATH, {"account_file": credential_path.name}
            )
            if subscription_enabled:
                notify("Identity verified. ChatGPT plan permission granted.")
            else:
                notify("Identity verified. ChatGPT plan permission was not granted.")
            return credentials
        except KeyboardInterrupt:
            notify("Sign-in cancelled")
        except (RuntimeError, TimeoutError) as error:
            notify(str(error))
        finally:
            server.callback_done.set()
            server.shutdown()
            listener.join()


def logout_chatgpt(
    credentials: AccountRecord | None, *, notify: Callable[[str], object] = print
) -> Registration | None:
    if credentials is None:
        return None

    remote_confirmed = True
    if credentials.get("refresh_token"):
        remote_confirmed = revoke_refresh_token(credentials)

    registration: Registration = {
        "issuer": credentials["issuer"],
        "subject": credentials["subject"],
        "client_id": credentials["client_id"],
        "ext_agent_host_id": credentials["ext_agent_host_id"],
    }

    registration["email"] = credentials.get("email")

    # Preserve the caller's dictionary identity while removing token fields.
    mutable_credentials = cast(dict[str, object], credentials)
    mutable_credentials.clear()
    mutable_credentials.update(registration)
    save_credential_record(registration)

    if not remote_confirmed:
        notify(
            "Signed out locally. Remote revocation was not confirmed; "
            "you can disconnect Vin in ChatGPT Settings."
        )
    return registration


if __name__ == "__main__":
    login_with_chatgpt(load_active_session())
