
from security_validator import (
    contains_hardcoded_password,
    contains_insecure_password_fallback,
    extract_python_code_blocks
)


# TEST 1: Detect an insecure hardcoded password fallback.
unsafe_code = '''
import os

password = os.getenv("ADMIN_PASSWORD", "default_password")
'''

unsafe_result = contains_insecure_password_fallback(unsafe_code)

print("Unsafe code detected:", unsafe_result)

assert unsafe_result is True


# TEST 2: Check a required environment variable without a fallback.
safe_code = '''
import os

password = os.environ["ADMIN_PASSWORD"]
'''

safe_result = contains_insecure_password_fallback(safe_code)

print("Safe code detected as insecure:", safe_result)

assert safe_result is False

print("All security validator tests passed!")


# TEST 3: Extract Python code from an AI-generated review.
review_text = '''
The password should be stored in an environment variable.

```python
import os

password = os.getenv("ADMIN_PASSWORD", "default_password")
```

This prevents credentials from being written directly in the source code.
'''

code_blocks = extract_python_code_blocks(review_text)

print("Extracted Python code blocks:", len(code_blocks))

assert len(code_blocks) == 1

insecure_result = contains_insecure_password_fallback(
    code_blocks[0]
)

print("Insecure AI recommendation detected:", insecure_result)

assert insecure_result is True

print("Python code extraction test passed!")


# TEST 4: Detect a hardcoded password with a type annotation.
annotated_code = 'password: str = "admin123"'

annotated_result = contains_hardcoded_password(annotated_code)

print("Annotated password detected:", annotated_result)

assert annotated_result is True

print("Annotated password test passed!")

# TEST 5: Detect a hardcoded password inside a dictionary.
dictionary_code = 'config = {"password": "admin123"}'

dictionary_result = contains_hardcoded_password(dictionary_code)

print("Dictionary password detected:", dictionary_result)

assert dictionary_result is True

print("Dictionary password test passed!")


# ============================================================
# TESTS 6–22: Additional AST security coverage
# ============================================================

from security_validator import (
    contains_password_print,
    analyze_source_security
)


# TEST 6: Detect a secret in a tuple assignment.
# A tuple allows multiple values to be assigned together.
tuple_code = 'password, username = "admin123"'

# The validator detects a hardcoded credential assigned
# to a sensitive variable inside a tuple target.
assert contains_hardcoded_password(tuple_code) is True


# TEST 7: Detect a hardcoded password with invalid Python syntax.
invalid_code = 'password = '

assert contains_hardcoded_password(invalid_code) is False


# TEST 8: Invalid syntax in environment-variable analysis.
assert contains_insecure_password_fallback(invalid_code) is False


# TEST 9: Ignore a normal function call.
normal_call_code = 'get_password("ADMIN_PASSWORD")'

assert contains_insecure_password_fallback(normal_call_code) is False


# TEST 10: Ignore unrelated attribute calls.
unrelated_call_code = 'config.get("ADMIN_PASSWORD", "admin123")'

assert contains_insecure_password_fallback(unrelated_call_code) is False


# TEST 11: Ignore os.getenv() without arguments.
empty_getenv_code = '''
import os
value = os.getenv()
'''

assert contains_insecure_password_fallback(empty_getenv_code) is False


# TEST 12: Ignore environment-variable names that are not string literals.
dynamic_key_code = '''
import os
key = "ADMIN_PASSWORD"
value = os.getenv(key, "admin123")
'''

assert contains_insecure_password_fallback(dynamic_key_code) is False


# TEST 13: Ignore hardcoded fallback values for nonsensitive keys.
normal_environment_code = '''
import os
region = os.getenv("REGION", "us-east-1")
'''

assert contains_insecure_password_fallback(normal_environment_code) is False


# TEST 14: Detect a keyword-based fallback credential.
keyword_fallback_code = '''
import os
password = os.getenv("DB_PASSWORD", default="admin123")
'''

assert contains_insecure_password_fallback(keyword_fallback_code) is True


# TEST 15: Ignore non-string keyword fallback values.
numeric_fallback_code = '''
import os
password = os.getenv("DB_PASSWORD", default=123)
'''

assert contains_insecure_password_fallback(numeric_fallback_code) is False


# TEST 16: Invalid syntax in password-print analysis.
assert contains_password_print(invalid_code) is False


# TEST 17: Ignore calls to functions other than print.
other_function_code = 'log(password)'

assert contains_password_print(other_function_code) is False


# TEST 18: Ignore attribute-based print calls.
attribute_print_code = 'logger.print(password)'

assert contains_password_print(attribute_print_code) is False


# TEST 19: Ignore printing nonsensitive variables.
normal_print_code = 'print(username)'

assert contains_password_print(normal_print_code) is False


# TEST 20: Detect direct printing of a password.
password_print_code = 'print(password)'

assert contains_password_print(password_print_code) is True


# TEST 21: Detect printing a password inside an expression.
nested_print_code = 'print("Password:", str(password))'

assert contains_password_print(nested_print_code) is True


# TEST 22: Verify the combined security analyzer reports exposure.
security_findings = analyze_source_security(
    'print(password)'
)

assert any(
    "SENSITIVE DATA EXPOSURE" in finding
    for finding in security_findings
)


print("All 22 security validation checks passed!")
