"""Small, ROS-independent gate for the shared perception control command."""
import json


class MissionControl:
    def __init__(self, enabled, object_type, modes):
        self.enabled = bool(enabled)
        self.object_type = object_type
        self.modes = frozenset(modes)
        self.mode = "OFF"
        self.object_types = ()

    @property
    def active(self):
        return not self.enabled or (
            self.mode in self.modes and self.object_type in self.object_types)

    def update(self, command):
        """Apply the mission node's {mode, object_types} command."""
        if not self.enabled:
            return False
        data = json.loads(command)
        mode = data["mode"]
        objects = tuple(sorted(data["object_types"])) if mode != "OFF" else ()
        changed = (mode, objects) != (self.mode, self.object_types)
        self.mode, self.object_types = mode, objects
        return changed
