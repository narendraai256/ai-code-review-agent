
from security_validator import (
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

# Extract Python code blocks from the AI review.
code_blocks = extract_python_code_blocks(review_text)

print("Extracted Python code blocks:", len(code_blocks))

assert len(code_blocks) == 1


# Validate the extracted Python code.
insecure_result = contains_insecure_password_fallback(
    code_blocks[0]
)

print("Insecure AI recommendation detected:", insecure_result)

assert insecure_result is True

print("Python code extraction test passed!")
