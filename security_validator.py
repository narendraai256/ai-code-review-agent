
import ast
import re


# Names commonly used for passwords, API keys, and secrets.
SECRET_NAMES = {
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "client_secret",
    "access_token",
    "auth_token",
    "private_key",
    "db_password",
    "database_password",
}


# Checks whether a variable name represents a sensitive credential.
def is_secret_name(name: str) -> bool:
    normalized = name.lower()

    return (
        normalized in SECRET_NAMES
        or normalized.endswith("_password")
        or normalized.endswith("_api_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
    )


# Checks whether a value is a nonempty string literal.
def is_string_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(node.value.strip())
    )


# Extracts the variable names from an assignment target.
def get_assignment_names(target: ast.AST) -> list[str]:
    names = []

    if isinstance(target, ast.Name):
        names.append(target.id)

    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            names.extend(get_assignment_names(item))

    return names


# Detects hardcoded credentials assigned to variables.
def contains_hardcoded_password(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):

        # Example: password = "admin123"
        if isinstance(node, ast.Assign):
            if is_string_literal(node.value):
                for target in node.targets:
                    for name in get_assignment_names(target):
                        if is_secret_name(name):
                            return True

        # Example: password: str = "admin123"
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                if is_string_literal(node.value):
                    for name in get_assignment_names(node.target):
                        if is_secret_name(name):
                            return True

        # Example: config = {"password": "admin123"}
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if is_string_literal(key) and is_string_literal(value):
                    if is_secret_name(key.value):
                        return True

    return False


# Detects hardcoded fallback credentials in environment-variable calls.
def contains_insecure_password_fallback(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if not isinstance(function, ast.Attribute):
            continue

        # Detect os.getenv("PASSWORD", "admin123")
        is_os_getenv = (
            function.attr == "getenv"
            and isinstance(function.value, ast.Name)
            and function.value.id == "os"
        )

        # Detect os.environ.get("PASSWORD", "admin123")
        is_os_environ_get = (
            function.attr == "get"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "environ"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "os"
        )

        if not (is_os_getenv or is_os_environ_get):
            continue

        if not node.args:
            continue

        # Only flag fallback credentials for secret-related keys.
        key = node.args[0]

        if not is_string_literal(key):
            continue

        if not is_secret_name(key.value):
            continue

        # Check positional fallback values.
        if len(node.args) >= 2:
            if is_string_literal(node.args[1]):
                return True

        # Check keyword fallback values.
        for keyword in node.keywords:
            if keyword.arg == "default":
                if is_string_literal(keyword.value):
                    return True

    return False


# Detects direct printing of sensitive credential variables.
def contains_password_print(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        if node.func.id != "print":
            continue

        for argument in node.args:
            for child in ast.walk(argument):
                if isinstance(child, ast.Name):
                    if is_secret_name(child.id):
                        return True

    return False


# Extracts Python code blocks from Markdown text.
def extract_python_code_blocks(review_text: str) -> list[str]:
    pattern = r"```(?:python|py)\s*\n(.*?)```"

    return re.findall(
        pattern,
        review_text,
        flags=re.DOTALL | re.IGNORECASE,
    )


# Returns security findings from Python source code.
def analyze_source_security(code: str) -> list[str]:
    findings = []

    if contains_hardcoded_password(code):
        findings.append(
            "HARDCODED SECRET: A password, API key, token, "
            "or other credential appears to be assigned a "
            "literal value. Remove the embedded credential "
            "and load it from a secure secrets manager or "
            "a required environment variable."
        )

    if contains_insecure_password_fallback(code):
        findings.append(
            "INSECURE CREDENTIAL FALLBACK: The code uses "
            "a hardcoded fallback credential. Remove the "
            "fallback and fail safely when the required "
            "credential is unavailable."
        )

    if contains_password_print(code):
        findings.append(
            "SENSITIVE DATA EXPOSURE: The code directly "
            "prints a credential variable. Remove the "
            "print statement to avoid exposing secrets "
            "in logs or terminal output."
        )

    return findings
