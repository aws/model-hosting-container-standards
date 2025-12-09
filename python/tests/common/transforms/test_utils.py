"""Unit tests for common.transforms.utils module."""

import pytest

from model_hosting_container_standards.common.transforms.utils import (
    _set_value_with_parent_creation,
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
        result = set_value(obj, "a.b.c.d", "value", create_parent=True)
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_create_parent_partial_existing(self):
        """Test creating parents when some already exist."""
        obj = {"a": {"b": {}}}
        result = set_value(obj, "a.b.c.d", "value", create_parent=True)
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

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


class TestSetValueWithParentCreation:
    """Test _set_value_with_parent_creation function."""

    def test_set_value_with_parent_creation_no_parents(self):
        """Test setting value with no parent parts."""
        obj = {}
        result = _set_value_with_parent_creation(obj, [], "key", "value")
        assert result == {"key": "value"}

    def test_set_value_with_parent_creation_single_parent(self):
        """Test creating a single parent level."""
        obj = {}
        result = _set_value_with_parent_creation(obj, ["parent"], "child", "value")
        assert result == {"parent": {"child": "value"}}

    def test_set_value_with_parent_creation_multiple_parents(self):
        """Test creating multiple parent levels."""
        obj = {}
        result = _set_value_with_parent_creation(obj, ["a", "b", "c"], "d", "value")
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_parent_creation_partial_existing(self):
        """Test creating parents when some already exist."""
        obj = {"a": {"b": {}}}
        result = _set_value_with_parent_creation(obj, ["a", "b", "c"], "d", "value")
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_parent_creation_all_existing(self):
        """Test setting value when all parents exist."""
        obj = {"a": {"b": {"c": {}}}}
        result = _set_value_with_parent_creation(obj, ["a", "b", "c"], "d", "value")
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    def test_set_value_with_parent_creation_max_depth_at_limit(self):
        """Test that depth exactly at limit is allowed."""
        obj = {}
        result = _set_value_with_parent_creation(
            obj, ["a", "b"], "c", "value", max_create_depth=3
        )
        assert result == {"a": {"b": {"c": "value"}}}

    def test_set_value_with_parent_creation_max_depth_exceeded(self):
        """Test that exceeding max depth raises KeyError."""
        obj = {}
        with pytest.raises(KeyError, match="exceeds max depth of 3"):
            _set_value_with_parent_creation(
                obj, ["a", "b", "c"], "d", "value", max_create_depth=3
            )

    def test_set_value_with_parent_creation_max_depth_with_existing_parents(self):
        """Test that max_depth only applies when creating, not for existing paths."""
        obj = {"a": {"b": {"c": {"d": {}}}}}
        result = _set_value_with_parent_creation(
            obj, ["a", "b", "c", "d"], "e", "value", max_create_depth=2
        )
        assert result == {"a": {"b": {"c": {"d": {"e": "value"}}}}}

    def test_set_value_with_parent_creation_max_depth_partial_existing(self):
        """Test max_depth when some parents exist and some need creation."""
        obj = {"a": {"b": {}}}
        # Path is a.b.c.d.e (depth 5), but a.b already exists
        # So we're only creating c.d.e (depth 5 total)
        with pytest.raises(KeyError, match="exceeds max depth of 4"):
            _set_value_with_parent_creation(
                obj, ["a", "b", "c", "d"], "e", "value", max_create_depth=4
            )

    def test_set_value_with_parent_creation_no_max_depth(self):
        """Test that very deep paths work when max_depth is None."""
        obj = {}
        result = _set_value_with_parent_creation(
            obj, ["a", "b", "c", "d", "e", "f"], "g", "value", max_create_depth=None
        )
        assert result == {"a": {"b": {"c": {"d": {"e": {"f": {"g": "value"}}}}}}}

    def test_set_value_with_parent_creation_overwrite_existing_value(self):
        """Test that existing values are overwritten."""
        obj = {"a": {"b": {"c": "old_value"}}}
        result = _set_value_with_parent_creation(obj, ["a", "b"], "c", "new_value")
        assert result == {"a": {"b": {"c": "new_value"}}}
