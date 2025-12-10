"""Unit tests for common.transforms.utils module."""

import pytest

from model_hosting_container_standards.common.transforms.utils import (  # _set_value_with_parent_creation,
    set_value,
)


class TestSetValue:
    """Test set_value function."""

    def test_set_value_simple_key(self):
        """Test setting a simple top-level key."""
        obj = {}
        result = set_value(obj, "key", "value")
        assert result == {"key": "value"}
        assert obj == {"key": "value"}

    def test_set_value_nested_existing_path(self):
        """Test setting a value on an existing nested path."""
        obj = {"parent": {"child": "old_value"}}
        result = set_value(obj, "parent.child", "new_value")
        assert result == {"parent": {"child": "new_value"}}

    def test_set_value_deep_existing_path(self):
        """Test setting a value on a deep existing path."""
        obj = {"a": {"b": {"c": {"d": "old"}}}}
        result = set_value(obj, "a.b.c.d", "new")
        assert result == {"a": {"b": {"c": {"d": "new"}}}}

    def test_set_value_missing_parent_raises_error(self):
        """Test that missing parent raises KeyError when create_parent=False."""
        obj = {"parent": {}}
        with pytest.raises(KeyError, match="Parent path 'parent.missing' not found"):
            set_value(obj, "parent.missing.child", "value")

    def test_set_value_with_create_parent_simple(self):
        """Test creating a simple nested structure."""
        obj = {}
        result = set_value(obj, "parent.child", "value", create_parent=True)
        assert result == {"parent": {"child": "value"}}

    def test_set_value_with_create_parent_deep(self):
        """Test creating a deep nested structure."""
        obj = {}
        result = set_value(
            obj, "a.b.c.d", "value", create_parent=True, max_create_depth=4
        )
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_create_parent_partial_existing(self):
        """Test creating parents when some already exist."""
        obj = {"a": {"b": {}}}
        result = set_value(
            obj, "a.b.c.d", "value", create_parent=True, max_create_depth=4
        )
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_create_parent_preserves_siblings(self):
        """Test that creating nested paths preserves existing sibling keys."""
        obj = {"a": {"b": {"existing_key": "existing_value", "another": 123}}}
        result = set_value(
            obj, "a.b.c.d", "new_value", create_parent=True, max_create_depth=4
        )
        assert result == {
            "a": {
                "b": {
                    "existing_key": "existing_value",
                    "another": 123,
                    "c": {"d": "new_value"},
                }
            }
        }
        # Verify existing keys are preserved
        assert result["a"]["b"]["existing_key"] == "existing_value"
        assert result["a"]["b"]["another"] == 123

    def test_set_value_preserves_sibling_at_target_level(self):
        """Test that updating a value preserves sibling keys at the same level."""
        obj = {"a": {"b": {"c": {"d": "old_value", "e": "other_value"}}}}
        result = set_value(obj, "a.b.c.d", "new_value", create_parent=True)
        assert result == {"a": {"b": {"c": {"d": "new_value", "e": "other_value"}}}}
        # Verify sibling key 'e' is preserved
        assert result["a"]["b"]["c"]["e"] == "other_value"

    def test_set_value_with_create_parent_max_depth_allowed(self):
        """Test that paths within max_create_depth are allowed."""
        obj = {}
        result = set_value(
            obj, "a.b.c", "value", create_parent=True, max_create_depth=3
        )
        assert result == {"a": {"b": {"c": "value"}}}

    def test_set_value_with_create_parent_max_depth_exceeded(self):
        """Test that paths exceeding max_create_depth raise KeyError."""
        obj = {}
        with pytest.raises(KeyError, match="exceeds max depth of 3"):
            set_value(obj, "a.b.c.d", "value", create_parent=True, max_create_depth=3)

    def test_set_value_with_create_parent_default_depth_exceeded(self):
        """Test that paths exceeding DEFAULT_MAX_DEPTH_TO_CREATE raise KeyError."""
        obj = {}
        # Default is 2, so depth 3 should fail
        with pytest.raises(KeyError, match="exceeds max depth of 2"):
            set_value(obj, "a.b.c", "value", create_parent=True)

        # Depth 4 should also fail
        with pytest.raises(KeyError, match="exceeds max depth of 2"):
            set_value(obj, "a.b.c.d", "value", create_parent=True)

    def test_set_value_with_create_parent_max_depth_existing_path_allowed(self):
        """Test that existing deep paths work regardless of max_create_depth."""
        obj = {"a": {"b": {"c": {"d": {"e": {}}}}}}
        result = set_value(
            obj, "a.b.c.d.e.f", "value", create_parent=True, max_create_depth=3
        )
        assert result == {"a": {"b": {"c": {"d": {"e": {"f": "value"}}}}}}

    def test_set_value_without_create_parent_existing_deep_path(self):
        """Test that existing deep paths work without create_parent."""
        obj = {"a": {"b": {"c": {"d": {}}}}}
        result = set_value(obj, "a.b.c.d.e", "value", create_parent=False)
        assert result == {"a": {"b": {"c": {"d": {"e": "value"}}}}}
