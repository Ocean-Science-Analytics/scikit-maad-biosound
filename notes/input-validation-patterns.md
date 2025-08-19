# Input Validation Patterns: Educational Overview

## What is Input Validation?

Input validation is the practice of checking user-provided data before using it in your application. It serves as the first line of defense against crashes, security vulnerabilities, and unexpected behavior.

## Why Input Validation Matters

1. **Prevents Crashes**: Invalid data can cause runtime errors that crash your application
2. **Improves User Experience**: Clear error messages help users fix problems quickly
3. **Security**: Validates data hasn't been tampered with or contains malicious content
4. **Data Integrity**: Ensures your application processes only meaningful data
5. **Debugging**: Makes it easier to identify where problems originate

## When to Use Input Validation

Input validation should be applied at **every entry point** where external data enters your system:

- User interface inputs (text fields, dropdowns, file uploads)
- Command-line arguments
- Configuration files
- API endpoints
- File imports
- Database queries

**Golden Rule**: Never trust external input - always validate it.

## Common Validation Patterns

### 1. Existence and Type Checking
```python
def validate_required_string(value, field_name):
    if not value:
        raise ValidationError(f"{field_name} cannot be empty")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    return value.strip()
```

### 2. Range and Boundary Validation
```python
def validate_numeric_range(value, min_val, max_val, field_name):
    if value < min_val or value > max_val:
        raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")
    return value
```

### 3. Format Validation (Regex)
```python
def validate_email_format(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")
    return email
```

### 4. File System Validation
```python
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"File not readable: {file_path}")
    return os.path.abspath(file_path)
```

### 5. Business Logic Validation
```python
def validate_frequency_ranges(low_range, high_range):
    # Check individual ranges first
    validate_frequency_range(low_range)
    validate_frequency_range(high_range)
    
    # Check business rule: ranges shouldn't overlap significantly
    if low_range[1] > high_range[0]:
        overlap = low_range[1] - high_range[0]
        if overlap > (high_range[1] - high_range[0]) * 0.2:
            raise ValidationError("Frequency ranges overlap too much")
```

## Error Handling Strategies

### 1. Custom Exception Classes
```python
class ValidationError(Exception):
    """Clear, specific exception for validation failures"""
    pass

class ConfigurationError(ValidationError):
    """More specific validation error for config issues"""
    pass
```

### 2. Detailed Error Messages
```python
# Bad: Generic error
raise ValidationError("Invalid input")

# Good: Specific, actionable error
raise ValidationError(
    f"Frequency range maximum ({freq_max}) must be greater than "
    f"minimum ({freq_min}). Please check your frequency settings."
)
```

### 3. Validation Result Objects
```python
class ValidationResult:
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
    
    def add_error(self, message):
        self.is_valid = False
        self.errors.append(message)
    
    def add_warning(self, message):
        self.warnings.append(message)
```

## Validation Module Architecture

A well-structured validation module typically includes:

### 1. Core Validation Functions
- Basic type checking (string, number, boolean)
- Range validation (min/max bounds)
- Format validation (regex patterns)
- Existence validation (files, directories)

### 2. Domain-Specific Validators
- Business rule validation
- Cross-field validation
- Complex format validation

### 3. Utility Functions
- Error message formatting
- Conversion helpers (string to number with validation)
- Batch validation for multiple inputs

### 4. Integration Points
- GUI input validation
- CLI argument validation  
- Configuration file validation
- API parameter validation

## Best Practices

### 1. Fail Fast
Validate inputs as early as possible in your application flow.

### 2. Be Specific
Provide clear, actionable error messages that tell users exactly what's wrong and how to fix it.

### 3. Validate at Boundaries
Always validate data when it crosses system boundaries (user input, file input, network input).

### 4. Separate Validation from Business Logic
Keep validation code separate from your core application logic for better maintainability.

### 5. Use Consistent Patterns
Establish consistent validation patterns across your application.

### 6. Test Your Validators
Write comprehensive tests for your validation functions, including edge cases.

## Example: Comprehensive Parameter Validation

```python
def validate_marine_acoustic_parameters(params):
    """
    Validates a complete set of marine acoustic parameters.
    Demonstrates multiple validation patterns working together.
    """
    result = {}
    errors = []
    
    # 1. Required field validation
    required_fields = ['input_directory', 'flim_low', 'flim_mid', 'sensitivity']
    for field in required_fields:
        if field not in params or not params[field]:
            errors.append(f"Required field '{field}' is missing")
    
    if errors:
        raise ValidationError("Missing required parameters:\n" + "\n".join(errors))
    
    # 2. Directory validation
    try:
        result['input_directory'] = validate_directory_exists(params['input_directory'])
    except ValidationError as e:
        errors.append(str(e))
    
    # 3. Frequency range validation
    try:
        result['flim_low'] = validate_frequency_range(params['flim_low'])
        result['flim_mid'] = validate_frequency_range(params['flim_mid'])
        
        # 4. Cross-field business logic validation
        if result['flim_low'][1] >= result['flim_mid'][0]:
            errors.append("Low frequency range maximum must be less than mid frequency range minimum")
    except ValidationError as e:
        errors.append(str(e))
    
    # 5. Numeric parameter validation
    try:
        result['sensitivity'] = validate_parameter_value(
            params['sensitivity'], 'Sensitivity', min_val=-200, max_val=0
        )
    except ValidationError as e:
        errors.append(str(e))
    
    # 6. Collect and report all errors
    if errors:
        raise ValidationError("Parameter validation failed:\n" + "\n".join(errors))
    
    return result
```

## Integration with Applications

### GUI Applications
- Validate inputs on field change or form submission
- Show validation errors near the relevant input fields
- Disable submit buttons until all validation passes

### CLI Applications  
- Validate command-line arguments before processing
- Show usage help when validation fails
- Exit gracefully with meaningful error codes

### APIs
- Validate request parameters before processing
- Return structured error responses (HTTP 400 with details)
- Log validation failures for monitoring

## Testing Validation Logic

```python
def test_frequency_range_validation():
    # Test valid input
    result = validate_frequency_range([0, 1000])
    assert result == [0.0, 1000.0]
    
    # Test invalid inputs
    with pytest.raises(ValidationError, match="minimum frequency"):
        validate_frequency_range([-100, 1000])
    
    with pytest.raises(ValidationError, match="must be less than maximum"):
        validate_frequency_range([1000, 500])
    
    # Test edge cases
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_frequency_range("")
```

## Conclusion

Input validation is a critical foundation for robust applications. By implementing consistent validation patterns, providing clear error messages, and validating at system boundaries, you can prevent crashes, improve user experience, and make your applications more maintainable.

The key is to be proactive: validate early, fail fast, and provide actionable feedback to help users correct their input.