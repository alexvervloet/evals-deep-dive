"""
Setup check: run this first.

    secrun python check_setup.py

Checks your Python version, the installed packages, your chosen PROVIDER, and the
API key that provider needs, and tells you exactly what to fix. Makes NO API
calls, so it costs nothing. Uses only the standard library, so it runs even before
`pip install` and can report missing dependencies instead of crashing.
"""

import importlib.util
import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def ok(msg):
    print(f"  {_c('✓', '32')} {msg}")


def warn(msg):
    print(f"  {_c('!', '33')} {msg}")


def fail(msg):
    print(f"  {_c('✗', '31')} {msg}")


HERE = os.path.dirname(os.path.abspath(__file__))


def _read_env_file():
    env_path = os.path.join(HERE, ".env")
    values = {}
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _get(env, name):
    return os.getenv(name) or (env or {}).get(name, "")


ALWAYS = [
    ("dotenv", "python-dotenv", "loads PROVIDER/config from .env"),
    ("rich", "rich", "score tables in the eval_run.py capstone"),
]
PROVIDER_DEPS = {
    "openai": [("openai", "openai", "OpenAI chat")],
    "claude": [("anthropic", "anthropic", "Claude chat")],
}
PROVIDER_KEYS = {
    "openai": [("OPENAI_API_KEY", "sk-", "sk-your-openai-key-here")],
    "claude": [("ANTHROPIC_API_KEY", "sk-ant-", "sk-ant-your-key-here")],
}


def check_python():
    print("Python version")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        ok(f"Python {major}.{minor} (3.10+ required)")
        return True
    fail(f"Python {major}.{minor}: this repo needs Python 3.10 or newer.")
    print("    Install a newer Python from https://www.python.org/downloads/")
    return False


def check_provider(env):
    print("\nProvider")
    provider = (_get(env, "PROVIDER") or "openai").strip().lower()
    if provider in PROVIDER_DEPS:
        ok(f"PROVIDER = {provider}")
        return provider
    fail(f"PROVIDER = {provider!r} is not recognized.")
    print("    Set PROVIDER=openai or PROVIDER=claude in .env.")
    return None


def check_dependencies(provider):
    print("\nDependencies")
    needed = ALWAYS + PROVIDER_DEPS.get(provider, [])
    missing = []
    for import_name, pip_name, purpose in needed:
        if importlib.util.find_spec(import_name) is not None:
            ok(f"{pip_name}: {purpose}")
        else:
            fail(f"{pip_name} MISSING: {purpose}")
            missing.append(pip_name)
    if missing:
        print("\n    Install everything with:")
        print("        pip install -r requirements.txt")
    return not missing


def check_keys(env, provider):
    print("\nAPI key")
    if env is None:
        fail(".env file not found.")
        print("    Create it with:  cp .env.example .env")
        return False
    all_ok = True
    for name, prefix, placeholder in PROVIDER_KEYS.get(provider, []):
        value = _get(env, name)
        if not value or value == placeholder:
            fail(f"{name} is not set.")
            print("    Store it in your OS keychain and run `secrun python check_setup.py` . See SECRETS.md.")
            all_ok = False
        elif not value.startswith(prefix):
            warn(f"{name} is set but doesn't start with '{prefix}'. Double-check it.")
        else:
            ok(f"{name} is set and looks right.")
    return all_ok


def main():
    print(_c("Checking your setup for the Evals deep dive...\n", "1"))
    env = _read_env_file()
    py = check_python()
    provider = check_provider(env)
    if provider is None:
        print(_c("\nFix PROVIDER in .env, then run this again.", "1;31"))
        return 1
    deps = check_dependencies(provider)
    keys = check_keys(env, provider)

    print()
    if py and deps and keys:
        print(_c("All set! 🎉", "1;32"))
        print("Start here:  python examples/01_anatomy.py")
        print("(Examples 01–04 are offline and need no key.)")
        return 0
    print(_c("Not ready yet. Fix the ✗ items above, then run this again.", "1;31"))
    print("(Examples 01–04 are offline, so you can start those without a key.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
