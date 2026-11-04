"""
ExamGuard - Exam Hall Policy & Item Permission Engine
Allows exam administrators and invigilators to configure allowed vs. prohibited items
per examination hall / session (e.g. Scientific Calculators, Reference Books, Laptops).
Dynamically adjusts HUD colors (Mint Green for Allowed, Crimson Red for Prohibited)
and controls whether AI triggers security alerts.
"""

from typing import Dict, Any, Optional

# Master item definitions with display labels, categories, and icon identifiers
POLICY_ITEMS: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "label": "Scientific / Basic Calculator",
        "category": "Electronics",
        "icon": "ph-calculator",
        "description": "Scientific calculators (e.g. Casio fx-991, TI-30X) or standard desk calculators",
        "aliases": ["calculator", "remote"]
    },
    "cell phone": {
        "label": "Mobile Phone / Smartphone",
        "category": "Electronics",
        "icon": "ph-device-mobile",
        "description": "Smartphones, cellular devices, feature phones",
        "aliases": ["cell phone", "mobile_phone", "phone"]
    },
    "book": {
        "label": "Books / Notes / Cheat Paper",
        "category": "Exam Material",
        "icon": "ph-book-open-text",
        "description": "Physical textbooks, reference notebooks, unauthorized paper sheets",
        "aliases": ["book", "paper"]
    },
    "laptop": {
        "label": "Laptop / Workstation",
        "category": "Electronics",
        "icon": "ph-laptop",
        "description": "Laptops, netbooks, keyboards, computer mice",
        "aliases": ["laptop", "computer", "keyboard", "mouse"]
    },
    "tablet": {
        "label": "Tablet / iPad / Screen",
        "category": "Electronics",
        "icon": "ph-device-tablet-camera",
        "description": "Portable tablets, e-readers, external monitors",
        "aliases": ["tablet", "tv", "monitor"]
    },
    "clock": {
        "label": "Smartwatch / Wearable",
        "category": "Wearables",
        "icon": "ph-watch",
        "description": "Smartwatches, fitness trackers, programmable digital timers",
        "aliases": ["clock", "watch"]
    },
    "backpack": {
        "label": "Backpack / Handbag / Pouch",
        "category": "Storage",
        "icon": "ph-backpack",
        "description": "Bags, pouches, or cases near the candidate desk",
        "aliases": ["backpack", "handbag", "suitcase"]
    },
    "headphones": {
        "label": "Headphones / Earbuds",
        "category": "Electronics",
        "icon": "ph-headphones",
        "description": "In-ear earbuds, wireless pods, over-ear headphones",
        "aliases": ["headphones"]
    },
    "pencil": {
        "label": "Pencils & Pens",
        "category": "Stationery",
        "icon": "ph-pencil-simple",
        "description": "Writing instruments (pencils, ballpoint pens)",
        "aliases": ["pencil", "pen"]
    },
    "eraser": {
        "label": "Eraser & Sharpener",
        "category": "Stationery",
        "icon": "ph-eraser",
        "description": "Standard exam stationery (erasers, sharpeners)",
        "aliases": ["eraser", "sharpener"]
    },
    "ruler": {
        "label": "Ruler / Scale",
        "category": "Stationery",
        "icon": "ph-ruler",
        "description": "Plastic or wooden measuring rulers",
        "aliases": ["ruler"]
    },
    "scissors": {
        "label": "Scissors / Cutting Tools",
        "category": "Tools",
        "icon": "ph-scissors",
        "description": "Scissors, utility blades, craft cutters",
        "aliases": ["scissors"]
    }
}

# Standard Exam Hall Presets
EXAM_PRESETS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "id": "standard",
        "name": "Standard Theory Exam (Strict)",
        "badge": "Strict Closed-Book",
        "description": "All electronic devices, calculators, notes, and bags strictly prohibited. Only standard writing stationery allowed.",
        "rules": {
            "cell phone": {"allowed": False, "enabled": True},
            "calculator": {"allowed": False, "enabled": True},
            "book": {"allowed": False, "enabled": True},
            "laptop": {"allowed": False, "enabled": True},
            "tablet": {"allowed": False, "enabled": True},
            "clock": {"allowed": False, "enabled": True},
            "backpack": {"allowed": False, "enabled": True},
            "headphones": {"allowed": False, "enabled": True},
            "pencil": {"allowed": True, "enabled": True},
            "eraser": {"allowed": True, "enabled": True},
            "ruler": {"allowed": True, "enabled": True},
            "scissors": {"allowed": False, "enabled": True},
        }
    },
    "scientific": {
        "id": "scientific",
        "name": "Engineering & Science Exam (Calculators Allowed)",
        "badge": "Calculators Allowed",
        "description": "Scientific calculators and standard stationery are explicitly authorized. Mobile phones, books, and laptops remain prohibited.",
        "rules": {
            "cell phone": {"allowed": False, "enabled": True},
            "calculator": {"allowed": True, "enabled": True},
            "book": {"allowed": False, "enabled": True},
            "laptop": {"allowed": False, "enabled": True},
            "tablet": {"allowed": False, "enabled": True},
            "clock": {"allowed": False, "enabled": True},
            "backpack": {"allowed": False, "enabled": True},
            "headphones": {"allowed": False, "enabled": True},
            "pencil": {"allowed": True, "enabled": True},
            "eraser": {"allowed": True, "enabled": True},
            "ruler": {"allowed": True, "enabled": True},
            "scissors": {"allowed": False, "enabled": True},
        }
    },
    "open_book": {
        "id": "open_book",
        "name": "Open-Book Exam (Notes & Calculators Allowed)",
        "badge": "Open Book & Notes",
        "description": "Textbooks, reference notes, and calculators permitted. Communication devices (phones, laptops, smartwatches) remain strictly prohibited.",
        "rules": {
            "cell phone": {"allowed": False, "enabled": True},
            "calculator": {"allowed": True, "enabled": True},
            "book": {"allowed": True, "enabled": True},
            "laptop": {"allowed": False, "enabled": True},
            "tablet": {"allowed": False, "enabled": True},
            "clock": {"allowed": False, "enabled": True},
            "backpack": {"allowed": False, "enabled": True},
            "headphones": {"allowed": False, "enabled": True},
            "pencil": {"allowed": True, "enabled": True},
            "eraser": {"allowed": True, "enabled": True},
            "ruler": {"allowed": True, "enabled": True},
            "scissors": {"allowed": False, "enabled": True},
        }
    },
    "lab_exam": {
        "id": "lab_exam",
        "name": "Computer Practical / Coding Lab Exam",
        "badge": "Lab Workstations Permitted",
        "description": "Laptops, desktop computers, and calculators authorized for lab evaluations. Personal mobile phones and earphones prohibited.",
        "rules": {
            "cell phone": {"allowed": False, "enabled": True},
            "calculator": {"allowed": True, "enabled": True},
            "book": {"allowed": True, "enabled": True},
            "laptop": {"allowed": True, "enabled": True},
            "tablet": {"allowed": True, "enabled": True},
            "clock": {"allowed": False, "enabled": True},
            "backpack": {"allowed": False, "enabled": True},
            "headphones": {"allowed": False, "enabled": True},
            "pencil": {"allowed": True, "enabled": True},
            "eraser": {"allowed": True, "enabled": True},
            "ruler": {"allowed": True, "enabled": True},
            "scissors": {"allowed": False, "enabled": True},
        }
    }
}


class PolicyManager:
    """Manages dynamic exam hall permissions, presets, and item classification rules."""
    
    def __init__(self):
        self.active_preset_id = "standard"
        # Deep copy default standard rules
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.apply_preset("standard")

    def _build_alias_map(self) -> Dict[str, str]:
        mapping = {}
        for primary_key, meta in POLICY_ITEMS.items():
            mapping[primary_key] = primary_key
            for alias in meta.get("aliases", []):
                mapping[alias] = primary_key
        return mapping

    def normalize_class(self, raw_class_name: str) -> str:
        """Maps any detector/tracker class alias to canonical policy item key."""
        clean = raw_class_name.lower().strip()
        alias_map = self._build_alias_map()
        return alias_map.get(clean, clean)

    def apply_preset(self, preset_id: str) -> bool:
        """Applies a preset configuration to active rules."""
        if preset_id not in EXAM_PRESETS:
            return False
        
        self.active_preset_id = preset_id
        preset = EXAM_PRESETS[preset_id]
        self.rules = {}
        for item_key, item_meta in POLICY_ITEMS.items():
            preset_rule = preset["rules"].get(item_key, {"allowed": False, "enabled": True})
            self.rules[item_key] = {
                "key": item_key,
                "label": item_meta["label"],
                "category": item_meta["category"],
                "icon": item_meta["icon"],
                "description": item_meta["description"],
                "allowed": bool(preset_rule.get("allowed", False)),
                "enabled": bool(preset_rule.get("enabled", True)),
            }
        return True

    def set_item_rule(self, item_key: str, allowed: Optional[bool] = None, enabled: Optional[bool] = None) -> bool:
        """Updates permissions or enabled status for a specific item."""
        canonical = self.normalize_class(item_key)
        if canonical not in self.rules:
            if canonical not in POLICY_ITEMS:
                return False
            meta = POLICY_ITEMS[canonical]
            self.rules[canonical] = {
                "key": canonical,
                "label": meta["label"],
                "category": meta["category"],
                "icon": meta["icon"],
                "description": meta["description"],
                "allowed": False,
                "enabled": True,
            }

        if allowed is not None:
            self.rules[canonical]["allowed"] = bool(allowed)
        if enabled is not None:
            self.rules[canonical]["enabled"] = bool(enabled)
            
        self.active_preset_id = "custom"
        return True

    def evaluate_item(self, class_name: str) -> Dict[str, Any]:
        """
        Evaluates a detected object class against active exam hall policy.
        Returns:
            - enabled: bool (whether to track/look for this item)
            - allowed: bool (whether permitted in this hall)
            - severity: str ('ALLOWED' vs 'HIGH' / 'MEDIUM')
            - color: tuple BGR (Mint Green if allowed, Crimson Red/Amber if prohibited)
            - status_badge: str ('ALLOWED' vs 'PROHIBITED')
            - label: str (Human-readable label)
        """
        canonical = self.normalize_class(class_name)
        rule = self.rules.get(canonical)

        if not rule:
            # Fallback for unrecognized classes: treat as prohibited if unknown electronic, or neutral
            return {
                "key": canonical,
                "enabled": True,
                "allowed": False,
                "severity": "HIGH",
                "color": (0, 0, 255),  # Red
                "status_badge": "PROHIBITED",
                "label": class_name.title()
            }

        enabled = rule["enabled"]
        allowed = rule["allowed"]

        if allowed:
            # Mint Green highlight for authorized items (e.g. calculator in science exam)
            color = (60, 214, 63)  # Bright mint-green in BGR
            severity = "ALLOWED"
            status_badge = "ALLOWED"
        else:
            # Crimson Red highlight for prohibited contraband
            color = (0, 0, 255) if canonical in ["cell phone", "laptop", "tablet", "clock"] else (0, 140, 255)
            severity = "HIGH" if canonical in ["cell phone", "laptop", "tablet", "clock", "backpack"] else "MEDIUM"
            status_badge = "PROHIBITED"

        return {
            "key": canonical,
            "enabled": enabled,
            "allowed": allowed,
            "severity": severity,
            "color": color,
            "status_badge": status_badge,
            "label": rule["label"]
        }

    def get_policy_state(self) -> Dict[str, Any]:
        """Returns the full state for frontend dashboard consumption."""
        presets_summary = [
            {
                "id": p_id,
                "name": p_data["name"],
                "badge": p_data["badge"],
                "description": p_data["description"],
                "is_active": (p_id == self.active_preset_id)
            }
            for p_id, p_data in EXAM_PRESETS.items()
        ]

        active_preset_info = EXAM_PRESETS.get(self.active_preset_id, {
            "name": "Custom Exam Hall Policy",
            "badge": "Custom Rules",
            "description": "Customized item rules for this examination session."
        })

        items_list = list(self.rules.values())

        return {
            "active_preset_id": self.active_preset_id,
            "active_preset_name": active_preset_info["name"],
            "active_preset_badge": active_preset_info["badge"],
            "description": active_preset_info["description"],
            "presets": presets_summary,
            "rules": self.rules,
            "items": items_list
        }


# Global Singleton Instance
policy_manager = PolicyManager()
