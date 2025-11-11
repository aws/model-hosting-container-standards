"""Tests for AppendOperation model."""

from model_hosting_container_standards.sagemaker.lora.models import AppendOperation


class TestAppendOperation:
    """Test AppendOperation model functionality."""

    def test_jmespath_compilation(self):
        """Test AppendOperation automatically compiles JMESPath expressions."""
        op = AppendOperation(separator=":", expression='headers."test"')
        assert op.compiled_expression is not None
        assert hasattr(op.compiled_expression, "search")

    def test_default_operation_field(self):
        """Test AppendOperation has default operation field."""
        op = AppendOperation(separator=":", expression='headers."test"')
        assert op.operation == "append"

    def test_custom_separator(self):
        """Test AppendOperation accepts custom separators."""
        op = AppendOperation(separator="-", expression='headers."test"')
        assert op.separator == "-"

    def test_empty_separator(self):
        """Test AppendOperation accepts empty separator."""
        op = AppendOperation(separator="", expression='headers."test"')
        assert op.separator == ""
